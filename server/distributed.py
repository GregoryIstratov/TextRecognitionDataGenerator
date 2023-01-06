import random as rnd
from multiprocessing import JoinableQueue, Process

def consumer(q: JoinableQueue):
    while True:
        res = q.get()
        print(f'Consume {res}')
        q.task_done()


def producer(q: JoinableQueue, food):
    for i in range(2):
        res = f'{food} {i}'
        print(f'Produce {res}')
        q.put(res)
    q.join()


if __name__ == "__main__":
    foods = ['apple', 'banana', 'melon', 'salad']
    jobs = 2
    q = JoinableQueue()

    producers = [
        Process(target=producer, args=(q, random.choice(foods)))
        for _ in range(jobs)
    ]

    # daemon=True is important here
    consumers = [
        Process(target=consumer, args=(q, ), daemon=True)
        for _ in range(jobs * 2)
    ]

    # + order here doesn't matter
    for p in consumers + producers:
        p.start()

    for p in producers:
        p.join()