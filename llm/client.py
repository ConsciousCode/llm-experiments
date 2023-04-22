import grpc
from build import llm_pb2, llm_pb2_grpc
import os

PORT = f"unix://{os.getcwd()}llm.sock"

def complete(prompt, max_tokens=10, temperature=1.0, *, top_k=0, top_p=0.9, stop=None, presence_penalty=0, frequency_penalty=0, stream=True):
	def complete_stream():
		with grpc.insecure_channel(PORT) as channel:
			stub = llm_pb2_grpc.LLMStub(channel)

			request = llm_pb2.CompletionRequest(
				prompt=prompt,
				max_tokens=max_tokens,
				temperature=temperature,
				top_k=top_k,
				top_p=top_p,
				presence_penalty=presence_penalty,
				frequency_penalty=frequency_penalty,
				stop=stop,
			)
			completion_responses = stub.Complete(iter([request]))

			for response in completion_responses:
				yield response.response
	
	cs = complete_stream()
	if not stream:
		cs = ''.join(cs)
	return cs

def embed(prompt):
	with grpc.insecure_channel(PORT) as channel:
		stub = llm_pb2_grpc.LLMStub(channel)
		request = llm_pb2.EmbedRequest(prompt=prompt)
		response = stub.Embed(request)
		return response.embed
