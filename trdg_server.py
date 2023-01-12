import sys
from pathlib import Path
cur_dir = str(Path(__name__).parent / "server")
sys.path.insert(0, cur_dir)

from queue import Queue
from server import serve
from generate import Generator
from multiprocessing import JoinableQueue, Process
from threading import Thread
import yaml

def consumer(q: JoinableQueue):
    while True:
        res = q.get()
        #print(f'Consume {res}')
        q.task_done()


def producer(i, q: JoinableQueue, opt: dict):
    gen = Generator(opt)
    while True:
        img, text = next(gen)
        #print(f'[P{i}] Produce text: {text}')
        q.put((img, text))
    q.join()

class QeuGen:
    q: Queue
    mpq: JoinableQueue
    t: Thread
    
    def __init__(self, mpq: JoinableQueue) -> None:
        self.q = Queue(maxsize=32)
        self.mpq = mpq
        self.t = Thread(target=self.__consumer)
        self.t.start()
        
    def __consumer(self):
        while True:
            img, text = self.mpq.get()
            #print(f"[consumer] got new item: {text}")
            self.q.put((img, text))
            self.mpq.task_done()
    
    def __next__(self):
        return self.q.get()
        

def launch_trdg_server(opt: dict, workers: int = 1, max_que_size: int = 8):
    mpq = JoinableQueue(maxsize=max_que_size)

    producers = [
        Process(target=producer, args=(i, mpq, opt))
        for i in range(workers)
    ]

    for p in producers:
        p.start()

    serve(QeuGen(mpq), max_workers=workers)
    
    for p in producers:
        p.join()

# if __name__ == "__main__":   
#     with open("../config_files/resnet_lstm_attn.yaml", 'r', encoding="utf8") as stream:
#         opt = yaml.safe_load(stream)
#     #opt = AttrDict(opt)
    
#     aug_opts = opt['tg_augs']
#     launch_trdg_server()
    