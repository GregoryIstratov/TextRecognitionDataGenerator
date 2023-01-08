import logging
import random

import grpc
import trdg_pb2
import trdg_pb2_grpc

import numpy as np
import cv2
import io
from PIL import Image

class Client:
    addr: str
    channel: grpc.Channel
    stub: trdg_pb2_grpc.TrdgStub
    
    def __init__(self, addr: str) -> None:
        self.addr = addr
        
    def __del__(self):
        if self.channel is not None:
            self.channel.close()
        
    def connect(self):
        self.channel = grpc.insecure_channel(self.addr)
        self.stub = trdg_pb2_grpc.TrdgStub(self.channel)
        
    def __next__(self) -> tuple[Image, str]:
        text_data: trdg_pb2.TextData = self.stub.GetText(trdg_pb2.TextRequest())
        
        buff = io.BytesIO(text_data.image)
        img = np.load(buff)
        image = Image.fromarray(img)
        return image, text_data.text
        
def get_text_and_show(client: Client):
    img, text = next(client)
    
    print(f"Get new text: {text}")
    cv2.imshow("main", np.asarray(img))
    cv2.waitKey()
    

def run():
    client = Client("172.25.0.86:50051")
    client.connect()
    #while True:
    for _ in range(5):
        get_text_and_show(client)
        


if __name__ == '__main__':
    logging.basicConfig()
    run()