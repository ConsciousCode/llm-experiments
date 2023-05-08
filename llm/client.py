import grpc
from build import llm_pb2, llm_pb2_grpc
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, Iterable
import torch
import torch.nn as nn
import tensor

PORT = f"unix://{os.getcwd()}/llm.sock"

@dataclass
class ForwardResponse:
	logits: np.ndarray
	hidden: list[np.ndarray]
	attention: list[np.ndarray]

class GRPCModel(nn.Module):
	def __init__(self, port=None):
		super().__init__()
		self.port = port or PORT
	
	def encode(self, text):
		'''Tokenize text into a list of tokens ids.'''
		
		with grpc.insecure_channel(PORT) as channel:
			stub = llm_pb2_grpc.LLMStub(channel)
			request = llm_pb2.Decoding(text=text)
			response = stub.Encode(request)
			return list(response.tokens)
	tokenize = encode
	
	def decode(self, tokens):
		'''Decode a list of token ids into text.'''
		
		with grpc.insecure_channel(PORT) as channel:
			stub = llm_pb2_grpc.LLMStub(channel)
			request = llm_pb2.Encoding(tokens=tokens)
			response = stub.Decode(request)
			return response.text

	def forward(self,
	    	input_ids: str|np.ndarray,
			attention_mask: Optional[np.ndarray]=None,
			return_hidden=False,
			return_attention=False
		):
		'''
		Convert text or a batch of token ids into logits, hidden states, and attention.
		
		Parameters:
			input_ids: Input text or pre-tokenized input ids.
			attention_mask: Optional attention mask.
			return_hidden: Return hidden states.
			return_attention: Return attention.
		'''
		
		if isinstance(input_ids, str):
			input = {"text": input_ids}
		elif isinstance(input_ids, (list, np.ndarray, torch.Tensor)):
			input = {"tokens": tensor.encode(input_ids, 'i')}
		else:
			raise TypeError(f"text must be str|list[int]|np.ndarray, got {type(input_ids)}")
		
		with grpc.insecure_channel(PORT) as channel:
			stub = llm_pb2_grpc.LLMStub(channel)
			request = llm_pb2.ForwardRequest(**input,
				attention_mask=tensor.encode(attention_mask, 'b'),
				return_hidden=return_hidden,
				return_attention=return_attention
			)
			response = stub.Forward(request)
			
			logits = tensor.encode(response.logits, 'f')
			hidden = response.hidden and [tensor.decode(h) for h in response.hidden]
			attention = response.attention and [tensor.decode(a) for a in response.attention]
			
			return ForwardResponse(logits, hidden, attention)

	def complete(self,
			text,
			max_tokens=10,
			temperature=1.0,
			*,
			top_k=0, top_p=0.9,
			stop: Optional[str|Iterable[str]]=None,
			presence_penalty=0,
			frequency_penalty=0,
			stream=True
		):
		'''
		Given a prompt, generate a completion.
		
		Parameters:
			text: The prompt to complete
			max_tokens: The maximum number of tokens to generate
			temperature: The temperature to use for sampling
			top_k: The number of tokens to consider for top-k sampling
			top_p: The cumulative probability to consider for top-p sampling
			stop: A token or list of tokens to stop at
			presence_penalty: The presence penalty to use
			frequency_penalty: The frequency penalty to use
			stream: Whether to stream the response or return it all at once
		'''
		
		if isinstance(stop, str):
			stop = [stop]
		
		with grpc.insecure_channel(PORT) as channel:
			stub = llm_pb2_grpc.LLMStub(channel)
			
			request = llm_pb2.CompletionRequest(
				text=text,
				max_tokens=max_tokens,
				temperature=temperature,
				top_k=top_k,
				top_p=top_p,
				presence_penalty=presence_penalty,
				frequency_penalty=frequency_penalty,
				stop=stop,
			)
			completion = (x.text for x in stub.Complete(request))
			return completion if stream else ''.join(completion)

	def embed(self, text):
		with grpc.insecure_channel(PORT) as channel:
			stub = llm_pb2_grpc.LLMStub(channel)
			request = llm_pb2.EmbedRequest(text=text)
			response = stub.Embed(request)
			return tensor.decode(response.embed)
