#!/usr/bin/env python3

from typing import TypeVar, Optional, Callable

# Defaults
NORM = 1e-5
DROP = 0.1
BIAS = True

# Helper functions

T = TypeVar("T")
def default(x: Optional[T], y: T|Callable[[], T]) -> T:
	'''Extensible defaults for function arguments.'''
	return x if x is not None else y() if callable(y) else y