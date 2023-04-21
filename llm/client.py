import grpc
from build import llm_pb2, llm_pb2_grpc
import os

PORT = f"unix://{os.getcwd()}llm.sock"

def complete(prompt, length=10, temperature=1.0, *, top_k=0, top_p=0.9, stop_tokens=None, repetition_penalty=1.0):
	with grpc.insecure_channel(PORT) as channel:
		stub = llm_pb2_grpc.LLMStub(channel)

		request = llm_pb2.CompletionRequest(
			prompt=prompt,
			length=length,
			temperature=temperature,
			top_k=top_k,
			top_p=top_p,
			stop_tokens=stop_tokens,
			repetition_penalty=repetition_penalty
		)
		completion_responses = stub.Complete(iter([request]))

		for response in completion_responses:
			yield response.response

def embed(prompt):
	with grpc.insecure_channel(PORT) as channel:
		stub = llm_pb2_grpc.LLMStub(channel)
		request = llm_pb2.EmbedRequest(prompt=prompt)
		response = stub.Embed(request)
		return response.embed
