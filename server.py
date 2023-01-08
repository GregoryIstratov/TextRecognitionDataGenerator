from queue import Queue
from server.server import serve
from generate import Generator
import logging
from multiprocessing import JoinableQueue, Process
from threading import Thread
import yaml

def consumer(q: JoinableQueue):
    while True:
        res = q.get()
        print(f'Consume {res}')
        q.task_done()


def producer(i, q: JoinableQueue, aug_opts):
    gen = Generator(aug_opts=aug_opts)
    while True:
        img, text = next(gen)
        print(f'[P{i}] Produce text: {text}')
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
            print(f"[consumer] got new item: {text}")
            self.q.put((img, text))
            self.mpq.task_done()
    
    def __next__(self):
        return self.q.get()
        

if __name__ == "__main__":
    logging.basicConfig() 
    
    with open("../config_files/resnet_lstm_attn.yaml", 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    #opt = AttrDict(opt)
    
    aug_opts = opt['tg_augs']
    jobs = 2
    mpq = JoinableQueue(maxsize=32)

    producers = [
        Process(target=producer, args=(i, mpq, aug_opts))
        for i in range(jobs)
    ]

    # + order here doesn't matter
    for p in producers:
        p.start()

    serve(QeuGen(mpq))
    
    for p in producers:
        p.join()
    