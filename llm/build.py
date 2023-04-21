try:
	import llm_pb2_grpc, llm_pb2
except ImportError:
	import grpc_tools.protoc
	grpc_tools.protoc.main([
		"grpc_tools.protoc",
		"-I.",
		"--python_out=.",
		"--grpc_python_out=.",
		"llm.proto"
	])
	# Reload
	import llm_pb2_grpc, llm_pb2
