#!/usr/bin/bash

IN=llm.proto
OUT=.

python -m grpc_tools.protoc -I. --python_out=$OUT --grpc_python_out=$OUT $IN
