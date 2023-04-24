#!/usr/bin/python env
'''
Pretty print with format functions.
'''

import pprint
from prompt_toolkit import print_formatted_text as printf, HTML
from prompt_toolkit.formatted_text import PygmentsTokens
from prompt_toolkit.styles import Style
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers import JsonLexer
import pygments
import prompt

import json as lib_json

LOG_COLORS = ["", "bold red", "yellow", "forestgreen", "teal", "magenta"]

log_style = Style.from_dict({
    level.lower(): color for level, color in zip(prompt.LOG_LEVEL, LOG_COLORS)
})

json_style = Style.from_dict({
    "pygments.keyword": '#ff6600',
	"pygments.operator": '#ff66ff',
	"pygments.punctuation": '#cccccc',
	"pygments.number": '#00ffff',
	"pygments.string": '#00ff00',
	"pygments.whitepsace": '#bbbbbb',
})

__all__ = ["json", "log"]

def json(obj):
	obj = lib_json.dumps(obj, indent=2).replace("<", "&lt;").replace(">", "&gt;")
	obj = PygmentsTokens(list(pygments.lex(obj, JsonLexer())))
	printf(obj, style=json_style)

def log(level: int, msg: str):
	msg = msg.replace("<", "&lt;").replace(">", "&gt;")
	level = prompt.LOG_LEVEL[level]
	printf(HTML(f"<{level}>[{level[0].upper()}] {msg}</{level}>"), style=log_style)