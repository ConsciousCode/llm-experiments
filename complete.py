import llm.client as client
import os

import openai
openai.api_key = os.getenv("API_KEY")

def chatgpt(prompt, *, engine=None, max_tokens=100, temperature=0.9, top_p=0.9, frequency_penalty=0.0, presence_penalty=0.0, stop=None):
	stop = stop or []
	stop.append("\n")
	engine = engine or LLM_ENGINE
	
	response = openai.Completion.create(
		engine=engine,
		prompt=prompt,
		max_tokens=max_tokens,
		temperature=temperature,
		top_p=top_p,
		frequency_penalty=frequency_penalty,
		presence_penalty=presence_penalty,
		stop=stop,
		stream=True
	)
	
	for token in response:
		x = token.choices[0].text
		yield x

def grpc(prompt, *, max_tokens=100, temperature=0.9, top_k=0, top_p=0.9, frequency_penalty=0.0, presence_penalty=0.0, stop=None):
	stop = stop or []
	stop.append("\n")
	
	response = client.complete(
		prompt=prompt,
		max_tokens=max_tokens,
		temperature=temperature,
		top_k=top_k,
		top_p=top_p,
		frequency_penalty=frequency_penalty,
		presence_penalty=presence_penalty,
		stop=stop
	)
	
	yield from response

LLM_ENGINE = os.getenv("ENGINE") or "text-davinci-003"
if LLM_ENGINE.lower() == "grpc":
	complete = grpc
else:
	complete = chatgpt