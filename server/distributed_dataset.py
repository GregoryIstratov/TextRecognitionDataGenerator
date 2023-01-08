import logging
import random

import grpc
import trdg_pb2
import trdg_pb2_grpc

import numpy as np
import cv2
import io

def get_text_and_show(stub: trdg_pb2_grpc.TrdgStub):
    text_data: trdg_pb2.TextData = stub.GetText(trdg_pb2.TextRequest())
    
    buff = io.BytesIO(text_data.image)
    img = np.load(buff)
    
    print(f"Get new text: {text_data.text}")
    cv2.imshow("main", img)
    cv2.waitKey()
    

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('172.25.0.86:50051') as channel:
        stub = trdg_pb2_grpc.TrdgStub(channel)
        while True:
            get_text_and_show(stub)
        


if __name__ == '__main__':
    logging.basicConfig()
    run()