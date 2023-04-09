import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data import RandomUTF8Dataset
from model import Orin
import torchinfo
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# Define hyperparameters
batch_size = 4
num_epochs = 10
learning_rate = 0.001

samples = 4096

class Validate(pl.pytorch.Callback):
	def on_train_epoch_start(self, trainer, plm):
		size = 1000
		result = trainer.model.generate("x", size)
		correct = 0
		while result:
			try:
				correct += len(result.decode('utf-8'))
				break
			except UnicodeDecodeError as e:
				correct += e.start
				result = result[e.start + 1:]
		
		self.log("val_loss", correct/size)
	
	def on_train_epoch_end(self, trainer, plm):
		trainer.model.save()

class LightningOrin(pl.LightningModule):
	def __init__(self, model):
		super().__init__()
		
		self.model = model
	
	def training_step(self, batch, batch_idx):
		loss = self.model.forward(batch, return_loss=True)
		self.log("train_loss", loss)
		return loss
	
	def configure_optimizers(self):
		opt = optim.Adam(self.model.parameters(), learning_rate)
		sched = optim.lr_scheduler.LinearLR(opt, 0.5, 0.1)
		return {
			"optimizer": opt,
			"lr_scheduler": sched,
			"monitor": "train_loss"
		}
	
	def generate(self, *args):
		return self.model.generate(*args)
	
	def save(self, *args):
		return self.model.save(*args)

def train(model, epochs):
	trainer = pl.Trainer(
		accelerator='gpu',
		max_epochs = epochs,
		min_epochs = 5,
		callbacks = [
			ModelCheckpoint("checkpoint", monitor="val_loss", mode="min"),
			LearningRateMonitor(),
			Validate()
		]
	)
	train_loader = DataLoader(
		RandomUTF8Dataset(samples), batch_size,
		shuffle=False,
		num_workers=8
	)
	print("Begin fit")
	trainer.fit(model=model, train_dataloaders=train_loader)

def main(file, epochs):
	torch.set_float32_matmul_precision('medium')
	print("Load module")
	model = Orin.load(file).to('cuda' if torch.cuda.is_available() else 'cpu')
	model = LightningOrin(model)
	torchinfo.summary(model, depth=10)
	train(model, epochs)
