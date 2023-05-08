#!/usr/bin/env python3
'''
Common methods for both client and server.
'''

from build import llm_pb2
import torch

def dtype_proto(dtype):
	match dtype:
		case 'b': return llm_pb2.BoolTensor
		case 'i': return llm_pb2.IntTensor
		case 'f': return llm_pb2.FloatTensor
		case _: raise ValueError(f"Unknown gRPC proto dtype: {dtype}")

def proto_dtype(proto):
	match proto:
		case llm_pb2.BoolTensor(): return torch.bool
		case llm_pb2.IntTensor(): return torch.int32
		case llm_pb2.FloatTensor(): return torch.float32
		case _: raise ValueError(f"Unknown gRPC proto: {type(proto)}")

def encode(tensor, dtype):
	if tensor is None:
		return None
	
	if isinstance(tensor, list):
		shape = (len(tensor),)
	else:
		shape = tensor.shape
		tensor = tensor.flatten()
	return dtype_proto(dtype)(data=tensor, shape=shape)

def decode(proto):
	if proto is None:
		return None
	return torch.tensor(proto.data, dtype=proto_dtype(proto)).reshape(tuple(proto.shape))
