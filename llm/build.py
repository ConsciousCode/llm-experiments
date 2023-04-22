# Importing this lets us "import" the proto file
import grpc_tools.protoc
import sys

try:
	import llm_pb2_grpc, llm_pb2
except ImportError:
	grpc_tools.protoc.main([
		"grpc_tools.protoc",
		"-I.",
		"--python_out=.",
		"--grpc_python_out=.",
		"llm.proto"
	])
	# Reload
	import llm_pb2_grpc, llm_pb2
