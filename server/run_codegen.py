"""Runs protoc with the gRPC plugin to generate messages and gRPC stubs."""

from grpc_tools import protoc
from pathlib import Path

pdir = Path(__file__).parent / "proto"

print(pdir)

#python -m grpc_tools.protoc -I ./proto --python_out=. --pyi_out=. --grpc_python_out=. trdg.proto
protoc.main((
    '',
    f"-I {str(pdir)}",
    f"--python_out={str(pdir.parent)}",
    f"--grpc_python_out={str(pdir.parent)}",
    "trdg.proto",
))
