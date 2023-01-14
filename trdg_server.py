import string
import sys
from pathlib import Path
cur_dir = str(Path(__name__).parent / "server")
sys.path.insert(0, cur_dir)

from queue import Queue
from server import serve
from generate import Generator
from multiprocessing import JoinableQueue, Process, Value
from threading import Thread
import yaml
import socket
import json
import signal
from functools import partial

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


class SocketServer:
    s: socket.socket
    t: dict[str, Process]
    is_exit: bool = False
    
    def __init__(self) -> None:
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(('', 50333))
        self.s.listen(1)
        self.t = dict[str, Process]()
        pass
    
    
    def __del__(self):
        if not self.is_exit:
            self.shutdown()

    def shutdown(self):
        self.is_exit = True
        print("[SocketServer] shutting down socket...")
        self.s.shutdown(socket.SHUT_RDWR)
        print("[SocketServer] closing socket...")
        self.s.close()
        print("[SocketServer] all clear")
    
    def create_trdg_process(self, opt: dict, workers: int = 1, max_que_size: int = 8):
        p = Process(target=launch_trdg_server, args=(opt, workers, max_que_size,))
        p.start()
        return p
    
    def create_trdg_override(self, name: str, opts: dict):
        if name in self.t.keys():
            print(f"Killing existing trdg server: {name}")
            self.t[name].kill()
            
        print(f"Creating new trdg server: {name} opts:\n{opts}")            
        self.t[name] = self.create_trdg_process(opt=opts, workers=1, max_que_size=512)
            
    
    def handle(self, conn, addr):
        with conn:
            print(f"Connected by {addr}")
            data:bytes = conn.recv(4096)
            if not data:
                print("Empty data")
                return
            
            sz = int.from_bytes(data[0:4], "big")
            print(f"Got payload of size: {sz}")
            payload = data[4:4+sz].decode('utf-8')
            
            print(f"payload: {payload}")
            
            request = json.loads(payload)
            
            if request['op'] == 'create': 
                name = request['name']
                opt = dict(request['opt'])
                self.create_trdg_override(name, opt)
            
            resp = b'OK'
            conn.sendall(resp)   
    
    def run(self):
        while not self.is_exit:
            try:
                conn, addr = self.s.accept()
                self.handle(conn, addr)
            except Exception:
                pass
        print("Exit listener")
    
    
def sigterm_handler(ss: SocketServer, _signo, _stack_frame):
    print("sig int handler")
    ss.shutdown()
            
if __name__ == "__main__":
    ss = SocketServer()
        
    signal.signal(signal.SIGINT, partial(sigterm_handler, ss))
    ss.run()
    # with open("../config_files/resnet_lstm_attn.yaml", 'r', encoding="utf8") as stream:
    #     opt = yaml.safe_load(stream)
    # #opt = AttrDict(opt)
    
    # aug_opts = opt['tg_augs']
    # launch_trdg_server()
    