import grpc
from build import llm_pb2, llm_pb2_grpc
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, Iterable

PORT = f"unix://{os.getcwd()}/llm.sock"

def tokenize(text):
	'''Tokenize text into a list of tokens ids.'''
	
	with grpc.insecure_channel(PORT) as channel:
		stub = llm_pb2_grpc.LLMStub(channel)
		request = llm_pb2.TokenizeRequest(text=text)
		response = stub.Tokenize(request)
		return list(response.tokens)

def tokendecode(tokens):
	'''Decode a list of token ids into text.'''
	
	with grpc.insecure_channel(PORT) as channel:
		stub = llm_pb2_grpc.LLMStub(channel)
		request = llm_pb2.TokenizeResponse(tokens=tokens)
		response = stub.TokenDecode(request)
		return response.text

@dataclass
class ProcessResponse:
	logits: np.ndarray
	hidden: list[np.ndarray]
	attention: list[np.ndarray]

def proto_to_tensor(proto):
	return np.array(proto.data).reshape(proto.shape)

def process(text: str|np.ndarray, return_hidden=False, return_attention=False):
	'''Convert text or a batch of token ids into logits, hidden states, and attention.'''
	
	if isinstance(text, str):
		input = {"text": text}
	elif isinstance(text, list):
		input = {"tokens": llm_pb2.Tensor(data=text, shape=(len(text),))}
	elif isinstance(text, np.ndarray):
		input = {"tokens": llm_pb2.Tensor(data=text.flatten(), shape=text.shape)}
	else:
		raise TypeError(f"text must be str|list[int]|np.ndarray, got {type(text)}")
	
	with grpc.insecure_channel(PORT) as channel:
		stub = llm_pb2_grpc.LLMStub(channel)
		request = llm_pb2.ProcessRequest(**input, return_hidden=return_hidden, return_attention=return_attention)
		response = stub.Process(request)
		
		logits = proto_to_tensor(response.logits)
		hidden = [proto_to_tensor(p) for p in response.hidden]
		attention = [proto_to_tensor(p) for p in response.attention]
		
		return ProcessResponse(logits, hidden, attention)

def complete(
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

def embed(text):
	with grpc.insecure_channel(PORT) as channel:
		stub = llm_pb2_grpc.LLMStub(channel)
		request = llm_pb2.EmbedRequest(text=text)
		response = stub.Embed(request)
		return proto_to_tensor(response.embed)
