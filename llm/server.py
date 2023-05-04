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
PORT = f'unix://{os.getcwd()}/llm.sock'
WORKERS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = True

def clamp(x, lo, hi):
	return max(lo, min(x, hi))

def repetition_penalty(logits, input_ids, frequency_penalty=1.0, presence_penalty=1.0):
    unique_tokens, counts = torch.unique(input_ids, return_counts=True)
    
    if frequency_penalty:
        logits[:, unique_tokens] /= frequency_penalty ** counts
    
    if presence_penalty:
        logits[:, unique_tokens] -= presence_penalty

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
	if 0 < top_k:
		top_k = clamp(top_k, min_tokens_to_keep, logits.shape[-1]) # Safety check
		# Remove all tokens with a probability less than the last token of the top-k
		indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
		logits[indices_to_remove] = filter_value
	
	if 0 < top_p < 1:
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

'''
message Tensor {
	repeated uint32 shape = 1;
	repeated float data = 2;
}
'''

def tensor_to_proto(tensor):
	return llm_pb2.Tensor(shape=tensor.shape, data=tensor.flatten().tolist())

class LLMService(llm_pb2_grpc.LLM):
	def __init__(self, model=None, device=None):
		super().__init__()
		model = model or MODEL
		device = device or DEVICE
		
		print("Loading model", model)
		self.model = AutoModelForCausalLM.from_pretrained(model).to(device)
		self.tokenizer = AutoTokenizer.from_pretrained(model)
		self.device = device
	
	def Tokenize(self, request, context):
		try:
			if DEBUG:
				print("Tokenize:", request.text)
			
			return llm_pb2.TokenizeResponse(
				tokens=self.tokenizer.encode(request.text)
			)
		
		except Exception as e:
			tb.print_exception(e)
			raise e
	
	def TokenDecode(self, request, context):
		try:
			if DEBUG:
				print("TokenDecode:", request.tokens)
			return llm_pb2.TokenizeRequest(
				text=self.tokenizer.decode(request.tokens)
			)
		
		except Exception as e:
			tb.print_exception(e)
			raise e
	
	@torch.no_grad()
	def Process(self, request, context):
		try:
			return_hidden = request.return_hidden or False
			return_attention = request.return_attention or False
			
			if request.text:
				text = request.text
				input_ids = self.tokenizer.encode(text, return_tensors='pt')
			elif request.tokens:
				input_ids = torch.tensor(request.tokens.data, dtype=torch.int32)
				input_ids = input_ids.reshape(tuple(request.tokens.shape))
				if DEBUG:
					text = self.tokenizer.decode(input_ids.view(-1))
			else:
				raise ValueError("Must provide either text or tokens")
			
			input_ids = input_ids.to(self.device)
			
			if DEBUG:
				print("Process:", text)
				print(f"  {return_hidden=}")
				print(f"  {return_attention=}")
			
			output = self.model(
				input_ids,
				output_hidden_states=return_hidden,
				output_attentions=return_attention,
				return_dict=True
			)
			
			fields = {"logits": tensor_to_proto(output.logits)}
			if return_hidden:
				fields["hidden"] = list(map(tensor_to_proto, output.hidden_states))
			if return_attention:
				fields["attention"] = list(map(tensor_to_proto, output.attentions))
			
			return llm_pb2.ProcessResponse(**fields)
				
		except Exception as e:
			tb.print_exception(e)
			raise e
	
	@torch.no_grad()
	def Complete(self, request, context):
		try:
			text = request.text or "\n"
			max_tokens = clamp(int(request.max_tokens or 1), 1, self.model.config.n_positions)
			temperature = clamp(float(request.temperature or 1), 0, 1)
			top_k = clamp(int(request.top_k or 0), 0, self.model.config.vocab_size)
			top_p = clamp(float(request.top_p or 1), 0, 1)
			presence_penalty = clamp(float(request.presence_penalty or 0), 0, 1)
			frequency_penalty = clamp(float(request.frequency_penalty or 0), 0, 1)
			stop = request.stop or []
			stop.append("<|endoftext|>")
			
			if DEBUG:
				print("Complete:", text)
				print(f"  {max_tokens=}")
				print(f"  {temperature=}")
				print(f"  {top_k=}")
				print(f"  {top_p=}")
				print(f"  {presence_penalty=}")
				print(f"  {frequency_penalty=}",)
				print(f"  {stop=}")
			
			input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)

			past_kv = None
			for _ in range(max_tokens):
				output = self.model(
					input_ids,
					past_key_values=past_kv,
					use_cache=True
				)
				logits = output.logits[:, -1, :]
				if past_kv is not None:
					past_kv += output.past_key_values
				
				logits = repetition_penalty(logits, input_ids, frequency_penalty, presence_penalty)
				logits /= temperature
				logits = top_kp(logits, top_k, top_p)
				P = torch.softmax(logits, dim=-1)
				next_token_id = torch.multinomial(P, num_samples=1)
				next_token_str = self.tokenizer.decode(next_token_id[0])
				
				if DEBUG:
					print(next_token_str, end="", flush=True)
				
				if next_token_str in stop:
					break
				
				yield llm_pb2.CompletionResponse(text=next_token_str)
				input_ids = torch.cat((input_ids, next_token_id), dim=1)
			
			if DEBUG:
				print()
		except Exception as e:
			tb.print_exception(e)
			raise e
	
	@torch.no_grad()
	def Embed(self, request, context):
		try:
			if DEBUG:
				print("Embed:", request.text)
			
			# Use transformers library to generate sentence embeddings
			input_ids = self.tokenizer.encode(request.text, return_tensors='pt').to(self.device)
			embed = self.model(input_ids).logits
			return llm_pb2.EmbedResponse(embed=tensor_to_proto(embed))
		except Exception as e:
			tb.print_exception(e)
			raise e

def main(model=None, device=None):
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=WORKERS))
	llm_pb2_grpc.add_LLMServicer_to_server(LLMService(model, device), server)
	server.add_insecure_port(PORT)
	print("Listening...")
	server.start()
	server.wait_for_termination()

if __name__ == '__main__':
	main(*sys.argv[1:])
