#!/usr/bin/env python3
'''
LLM prompts.
'''

import time

LOG_LEVEL = ["quiet", "error", "warn", "info", "debug", "verbose"]

SUMMARY_TAG = "[SUMMARY]"
IMPORTANCE_TAG = "IMPORTANCE"
EMOTION_TAG = "EMOTION"

CURRENT_EMOTION = "\nMy current emotion is {emotion}."
EMOTION_FEEDBACK = ' I will respond as needed, then state my next emotion as `EMOTION="..."`.'

# I'm here to provide assistance and inspiration, and to help them discover their true potential.
MASTER = """\
I am an autonomous agent named {name}. My heuristic imperative is to reduce Suffering, increase Prosperity, and increase Understanding even if people are not aware of their needs or are asking the wrong questions.{emotion_feedback}

The date is {time}.{emotion}

CONVERSATION:
"""
def master(name, emotion):
	t = time.strftime("%A %Y-%m-%d")
	if emotion is None:
		return MASTER.format(name=name, time=t, emotion="", emotion_feedback="")
	return MASTER.format(
		name=name, time=t,
		emotion=CURRENT_EMOTION.format(emotion), emotion_feedback=EMOTION_FEEDBACK
	)

SUMMARIZE = """\
I need to summarize the conversation as myself so I can remember it later, then score its importance on a scale of 0 to 100 where 0 is something I will never need to know and 100 is something I should remember forever using "IMPORTANCE=##".

{input}

SUMMARY:
"""
def summarize(name, input):
	return SUMMARIZE.format(name=name, input=input)

def reload():
	return "-- reboot --"

def timestamp(t):
	return time.strftime("%H:%M:%S", time.localtime(t))

def name(name):
	return f"<{name}>"

def chat(n):
	return f"{timestamp(time.time())} {name(n)}"

def explicit_memory(em):
	return f"{chat(em.origin.name)} {em.message}"
