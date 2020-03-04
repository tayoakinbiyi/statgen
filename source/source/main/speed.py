from multiprocessing import Process, Queue, managers
import time

def worker(id, data, queue):
    tmp1 = time.time()
    queue.put([data[x]*2 for x in range(len(data))])

def workerShared(id, data, queue, *args):
    tmp1 = time.time()
    for i in range(args[0], args[1]):
        data[i] *= 2

def without_shared_memory():
    print("Without shared memory")
    iterations = 6
    for i in range(2, 6):
        start_time = time.time()
        num_procs = 4
        data = list(range(1, 10**i))
        chunk_size = len(data) // num_procs
        for _ in range(iterations):
            queue = Queue()
            procs = [Process(target=worker,
                             args=(j, data[j*chunk_size:(j+1)*chunk_size],queue)) for j in range(num_procs)]
            for p in procs:
                p.start()

            tmp = 0
            for _ in range(num_procs):
                tmp += sum(queue.get())

            for p in procs:
                p.join()

        end_time = time.time()
        secs_per_iteration = (end_time - start_time) / iterations
        print("data {0:>10,} ints : {1:>6.6f} secs per iteration"
              .format(len(data), secs_per_iteration))


def with_shared_memory():
    print("With shared memory")
    iterations = 6
    for i in range(2, 6):
        num_procs = 4
        with managers.SharedMemoryManager() as smm:
            start_time = time.time()
            shm = smm.SharedMemory(create=True, size=a.nbytes)
            data = np.ndarray((10**i-1,), dtype='float64', buffer=shm.buf)

            chunk_size = len(data) // num_procs
            for _ in range(iterations):
                queue = Queue()
                procs = [Process(target=workerShared,
                                 args=(j, data, j*chunk_size,(j+1)*chunk_size)) for j in range(num_procs)]
                for p in procs:
                    p.start()

                for p in procs:
                    p.join()

                tmp = sum(data)

            end_time = time.time()
            secs_per_iteration = (end_time - start_time) / iterations
            print("data {0:>10,} ints : {1:>6.6f} secs per iteration"
                  .format(len(data), secs_per_iteration))


without_shared_memory()
with_shared_memory()