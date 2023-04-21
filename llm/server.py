from concurrent import futures
import grpc
from build import llm_pb2, llm_pb2_grpc
import os
import sys
import torch
import torch.nn.functional as F
from transformers.models.auto import AutoTokenizer, AutoModelForCausalLM
import traceback as tb

MODEL = "databricks/dolly-v2-7b"
MODEL = "gpt2"
PORT = f'unix://{os.getcwd()}llm.sock'
WORKERS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = True

def repetition_penalty(logits, input_ids, penalty):
	if abs(1 - penalty) < 1e-5:
		return logits
	
	unique = torch.unique(input_ids, dim=-1).cuda()
	for token_id in unique:
		logits[0, token_id] /= penalty
	return logits

def top_kp(
	logits: torch.Tensor,
	top_k: int = 0,
	top_p: float = 1.0,
	filter_value: float = -float("Inf"),
	min_tokens_to_keep: int = 1,
) -> torch.Tensor:
	"""Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
	Args:
		logits: logits distribution shape (batch size, vocabulary size)
		if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
		if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
			Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
		Make sure we keep at least min_tokens_to_keep per batch example in the output
	From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
	"""
	if top_k > 0:
		top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
		# Remove all tokens with a probability less than the last token of the top-k
		indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
		logits[indices_to_remove] = filter_value

	if top_p < 1.0:
		sorted_logits, sorted_indices = torch.sort(logits, descending=True)
		cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

		# Remove tokens with cumulative probability above the threshold (token with 0 are kept)
		sorted_indices_to_remove = cumulative_probs > top_p
		if min_tokens_to_keep > 1:
			# Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
			sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
		# Shift the indices to the right to keep also the first token above the threshold
		sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
		sorted_indices_to_remove[..., 0] = 0

		# scatter sorted tensors to original indexing
		indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
		logits[indices_to_remove] = filter_value
	return logits

def clamp(x, lo, hi):
	return max(lo, min(x, hi))

class LLMService(llm_pb2_grpc.LLM):
	def __init__(self, model=MODEL, device=DEVICE):
		super().__init__()
		print("Loading model", model)
		self.model = AutoModelForCausalLM.from_pretrained(model).to(device)
		self.tokenizer = AutoTokenizer.from_pretrained(model)
		self.device = device

	def Complete(self, request_iterator, context):
		try:
			for req in request_iterator:
				top_k = clamp(int(req.top_k or 0), 0, self.model.config.vocab_size)
				top_p = clamp(float(req.top_p or 0), 0, 1)
				temperature = clamp(float(req.temperature or 1), 0, 1)
				penalty = clamp(float(req.repetition_penalty or 1), 0, 1)
				stop_tokens = req.stop_tokens or []
				prompt = req.prompt or "\n"
				length = clamp(int(req.length or 1), 1, self.model.config.n_positions)
				
				if DEBUG:
					print("Request:", prompt)
					print("  length =", length)
					print("  temperature =", temperature)
					print("  top_k =", top_k)
					print("  top_p =", top_p)
					print("  repetition_penalty =", penalty)
					print("  stop_tokens =", stop_tokens)
				
				# Convert stop_tokens to corresponding token IDs
				stop_token_ids = list(map(self.tokenizer.convert_tokens_to_ids, stop_tokens))
				input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

				past_kv = None
				for _ in range(length):
					output = self.model(
						input_ids,
						past_key_values=past_kv,
						use_cache=True
					)
					logits = output.logits[:, -1, :]
					if past_kv is not None:
						past_kv += output.past_key_values
					
					logits = repetition_penalty(logits, input_ids, penalty)
					logits /= temperature
					logits = top_kp(logits, top_k, top_p)
					P = torch.softmax(logits, dim=-1)
					next_token_id = torch.multinomial(P, num_samples=1)
					next_token_str = self.tokenizer.decode(next_token_id[0])
					
					if DEBUG:
						print(next_token_str, end ="")
						sys.stdout.flush()
						
					if next_token_id.item() in stop_token_ids:
						break
					
					yield llm_pb2.CompletionResponse(response=next_token_str)
					input_ids = torch.cat((input_ids, next_token_id), dim=1)
				
				if DEBUG:
					print()
		except Exception as e:
			tb.print_exception(e)
			raise e
	
	def Complete_0(self, request_iterator, context):
		try:
			for req in request_iterator:
				input_ids = self.tokenizer.encode(req.prompt, return_tensors ="pt").to(self.device)
				stop_tokens = list(map(self.tokenizer.convert_tokens_to_ids, req.stop_tokens))
				#if not stop_tokens:
				#	stop_tokens = None
				
				eos_token = self.tokenizer.eos_token_id
				pad_token = self.tokenizer.pad_token_id or eos_token
				
				past_kv = None
				for _ in range(req.length):
					output = self.model(
						input_ids,
						past_key_values=past_kv,
						use_cache=True,
					)
					output = self.model.generate(
						input_ids,
						do_sample=True,
						max_length=input_ids.shape[1] + 1,
						temperature=req.temperature,
						top_p=req.top_p,
						repetition_penalty=req.repetition_penalty,
						eos_token_id=eos_token,
						pad_token_id=pad_token,
						num_return_sequences=1,
						use_cache=True,
						past_key_values=past_kv,
						return_dict_in_generate=True
					)
					print(output.past_key_values)
					next_token = output[:, -1]
					next_token_str = self.tokenizer.decode(next_token)
					if next_token in stop_tokens:
						break
					
					past_kv = output.past_key_values
					
					yield llm_pb2.CompletionResponse(response=next_token_str)
					input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
		except Exception as e:
			tb.print_exception(e)
			raise e

	def Embed(self, request, context):
		try:
			# Use transformers library to generate sentence embeddings
			embed = self.model(request.prompt)
			return llm_pb2.EmbedResponse(embed=embed)
		except Exception as e:
			tb.print_exception(e)
			raise e

def serve():
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=WORKERS))
	llm_pb2_grpc.add_LLMServicer_to_server(LLMService(), server)
	server.add_insecure_port(PORT)
	print("Listening...")
	server.start()
	server.wait_for_termination()

if __name__ == '__main__':
	serve()
