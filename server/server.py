from concurrent import futures
import logging
import math
import time

import grpc
import trdg_pb2
import trdg_pb2_grpc

import io
import numpy as np

class TrdgServicer(trdg_pb2_grpc.TrdgServicer):
    def __init__(self, gen):
        self.gen = gen

    def GetText(self, request, context):
        img, text = next(self.gen)
        print(f"[server][get_text] called. got new item: {text}")
        buff = io.BytesIO()
        np.save(buff, np.asarray(img))
        response = trdg_pb2.TextData(text=text, image=buff.getvalue())
        return response


def serve(gen, port: int = 50051, max_workers: int = 1):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    trdg_pb2_grpc.add_TrdgServicer_to_server(
        TrdgServicer(gen), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    server.wait_for_termination()
    