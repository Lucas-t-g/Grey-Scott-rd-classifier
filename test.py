from multiprocessing import Process
import os

def sum_v(v, s, e):
    print('module name:', __name__, os.getppid(), os.getpid())

    for i in range(s, e, 1):
        v[i] *= 10
    print(v)


if __name__ == '__main__':
    v = list(range(10))
    p = Process(target=sum_v, args=(v, 0, 9))
    p.start()
    p.join()
    print(v)
