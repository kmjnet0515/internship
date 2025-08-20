from multiprocessing import Process, Queue
import time
import random

def producer(q, name):
    for i in range(5):
        time.sleep(random.uniform(0.1, 0.5))  # 임의 작업 지연
        item = f'{name}-item-{i}'
        q.put(item)  # 큐에 데이터 넣기 (락 발생 가능 구간)
        print(f'{name} produced {item}')

def consumer(q, name):
    while True:
        try:
            item = q.get(timeout=1)  # 큐에서 데이터 꺼내기 (락 발생 가능 구간)
            print(f'{name} consumed {item}')
            time.sleep(random.uniform(0.1, 0.3))  # 임의 작업 지연
        except:
            # 1초 동안 대기 후 큐가 비어있으면 종료
            print(f'{name} finished consuming')
            break

if __name__ == '__main__':
    q = Queue()

    p1 = Process(target=producer, args=(q, 'Producer1'))
    p2 = Process(target=producer, args=(q, 'Producer2'))
    c1 = Process(target=consumer, args=(q, 'Consumer1'))

    p1.start()
    p2.start()
    c1.start()

    p1.join()
    p2.join()
    c1.join()
