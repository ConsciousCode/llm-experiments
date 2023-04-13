print("import torch")
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

print("import lightning")
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor, Callback
from lightning.pytorch import Trainer, LightningModule

print("import transformers")
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config

print("import datasets")
from datasets import load_dataset

from collections import OrderedDict
import os
import re

print("import model")
import model
import knn
print("Done")

TEACHER = "gpt2"
CACHE_FILE = "dataloader-map.cache"

os.environ["TOKENIZERS_PARALLELISM"] = "true"

torch.set_float32_matmul_precision('medium')
pl.seed_everything(42)

def map_gpt2_to_mine(teacher):
	if 'wpe' in teacher:
		return None
	
	student = (teacher
		.replace("transformer.", "lm.")
		.replace(".c_attn.", ".attn.qkv_proj.")
		.replace(".c_proj.", ".attn.out_proj.")
		.replace("wte.", "embed.")
	)

	return re.sub(r"lm\.h\.(\d+)\.", r"lm.\1.MyModelBlock.", student)


print("Loading teacher")

teacher = GPT2LMHeadModel.from_pretrained(TEACHER)
tokenizer = GPT2TokenizerFast.from_pretrained(TEACHER)
tokenizer.pad_token = tokenizer.eos_token
config = teacher.config
#print(list(teacher.state_dict().keys()))
#exit()
#print("Config:", config)
print("Done")

print("Building student")

student = model.MyGPT2Model(config)
student = model.LanguageModel(config.vocab_size, config.n_embd, student)
#student = model.TransformersWrapper(student)
#print(student)
#exit()
memory = knn.TestDatabase(config.n_embd, "test.db")

#print("Config:", config)
print("Done")

print("Transfer weights")

student_state = OrderedDict()

for teacher, teacher_weight in teacher.state_dict().items():
	student_key = map_gpt2_to_mine(teacher)
	if student_key is None:
		continue
	if 'proj' in student_key:
		print(teacher_weight.shape)
		teacher_weight = teacher_weight.T
	student_state[student_key] = teacher_weight

student.load_state_dict(student_state, strict=False)
print("Done")

class KnowledgeDistillationModel(pl.LightningModule):
	def __init__(self,
		  teacher, student, memory, tokenizer, dataset,
		  *,
		  temperature=2.0,
		  batch_size=8,
		  max_length=1024
		):
		super().__init__()
		self.teacher = teacher
		self.student = student
		self.memory = memory
		self.tokenizer = tokenizer
		self.dataset = dataset
		self.feedback = None
		self.batch_size = batch_size
		self.max_length = max_length
		self.temperature = temperature
		self.distill_loss = nn.KLDivLoss(reduction="batchmean")
		self.ce_loss = nn.CrossEntropyLoss()

	def training_step(self, batch, batch_idx):
		input_ids, attention_mask = batch
		input_ids = input_ids.reshape(self.batch_size, self.max_length)
		attention_mask = attention_mask.reshape(self.batch_size, self.max_length)
		labels = input_ids
		print("input", type(input_ids), input_ids)
		#input_ids = torch.stack(input_ids, dim=0)
		#attention_mask = torch.stack(attention_mask, dim=0)

		print("input_ids shape:", input_ids.shape, "type:", input_ids.dtype)
		print("attention_mask shape:", attention_mask.shape, "type:", attention_mask.dtype)
		
		# Teacher model output
		with torch.no_grad():
			teacher_logits = self.teacher(
				input_ids=input_ids,
				attention_mask=attention_mask
			).logits

		# Student model output
		student_logits, hidden = self.student(
			input_ids,
			attention_mask=attention_mask,
			memory=self.memory,
			feedback=self.feedback
		)
		print("hidden shape", hidden.shape)
		self.feedback = hidden[1:] + [None]

		# Calculate distillation loss
		distill_loss = self.distill_loss(
			F.log_softmax(student_logits / self.temperature, dim=-1),
			F.softmax(teacher_logits / self.temperature, dim=-1)
		)
		student_loss = self.ce_loss(student_logits, labels)
		teacher_loss = self.ce_loss(teacher_logits, labels)
		
		loss = distill_loss + student_loss
		
		self.log("train_loss", {
			"combined": loss,
			"distill": distill_loss,
			"ce_teacher": teacher_loss,
			"ce_student": student_loss
		})
		return loss

	def configure_optimizers(self):
		return optim.Adam(self.student.parameters(), lr=1e-4)

	def _collate(self, batch):
		'''
		Collates the inputs as (seq*batch,) tensors because of weirdness with
		how torch handles parallelism, which can exhaust file descriptors:
		https://github.com/pytorch/pytorch/issues/65198
		'''
		input_ids = [torch.tensor(item['input_ids']) for item in batch]
		attention_mask = [torch.tensor(item['attention_mask']) for item in batch]

		input_ids = torch.stack(input_ids, dim=0)
		attention_mask = torch.stack(attention_mask, dim=0)
		
		return input_ids, attention_mask

	def _dataloader(self, split):
		def tokenize(dataset):
			return self.tokenizer(dataset['text'],
				truncation=True,
				padding="max_length",
				max_length=self.max_length
			)
		dataset = self.dataset[split].map(tokenize, batched=True, cache_file_name=CACHE_FILE)
		return DataLoader(dataset, self.batch_size, num_workers=8, collate_fn=self._collate)
	
	def train_dataloader(self):
		return self._dataloader("train")

	def val_dataloader(self):
		return self._dataloader("validation")

	def test_dataloader(self):
		return self._dataloader("test")

class FrequentCheckpoint(Callback):
	def __init__(self, save_steps: int, output_dir: str):
		super().__init__()
		self.save_steps = save_steps
		self.output_dir = output_dir

	def on_batch_end(self, trainer, pl_module):
		global_step = trainer.global_step
		if global_step % self.save_steps == 0:
			ckpt_path = os.path.join(self.output_dir, f"checkpoint_step_{global_step}.ckpt")
			trainer.save_checkpoint(ckpt_path)
			print(f"Checkpoint saved at step {global_step}: {ckpt_path}")

block_size = config.n_positions
dataset = load_dataset("ag_news")

memory = knn.KNNMemory("orin", config.n_embd, config.n_embd)
print("Done")

print("Begin training")

# Train the model
model = KnowledgeDistillationModel(
	teacher, student, memory, tokenizer, dataset,
	max_length=config.n_positions,
	batch_size=8
)
trainer = pl.Trainer(accelerator="gpu", max_epochs=3, log_every_n_steps=8, callbacks=[
	LearningRateMonitor(logging_interval='step'),
	FrequentCheckpoint(save_steps=1000, output_dir="checkpoints")
])
trainer.fit(model)