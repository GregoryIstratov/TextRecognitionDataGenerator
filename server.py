from queue import Queue
from server.server import serve
from generate import Generator
import logging
from multiprocessing import JoinableQueue, Process
from threading import Thread


def consumer(q: JoinableQueue):
    while True:
        res = q.get()
        print(f'Consume {res}')
        q.task_done()


def producer(i, q: JoinableQueue):
    gen = Generator()
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
        self.q = Queue(maxsize=64)
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
    jobs = 12
    mpq = JoinableQueue(maxsize=64)

    producers = [
        Process(target=producer, args=(i, mpq))
        for i in range(jobs)
    ]

    # + order here doesn't matter
    for p in producers:
        p.start()

    serve(QeuGen(mpq))
    
    for p in producers:
        p.join()
    