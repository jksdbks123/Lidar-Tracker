import numpy as np
import time
import multiprocessing as mp
from multiprocessing import shared_memory, Semaphore
import psutil

def producer_shared_memory(shm_name, array_shape, num_arrays, sem_producer, sem_consumer, producer_id):
    shm = shared_memory.SharedMemory(name=shm_name)
    for i in range(num_arrays):
        array = np.random.rand(*array_shape).astype(np.float32)
        shared_array = np.ndarray(array_shape, dtype=np.float32, buffer=shm.buf)
        np.copyto(shared_array, array)
        sem_producer.release()
        sem_consumer.acquire()
    print(f"Producer {producer_id} finished")

def consumer_shared_memory(shm_name, array_shape, num_arrays, sem_producer, sem_consumer, consumer_id):
    shm = shared_memory.SharedMemory(name=shm_name)
    for i in range(num_arrays):
        sem_producer.acquire()
        shared_array = np.ndarray(array_shape, dtype=np.float32, buffer=shm.buf)
        array = np.array(shared_array)  # Make a copy to simulate full data usage
        sem_consumer.release()
    print(f"Consumer {consumer_id} finished")

def test_multiple_shared_memory(array_shape, num_arrays, num_pairs):
    shared_memories = []
    semaphores = []
    processes = []

    start_time = time.time()

    for i in range(num_pairs):
        array_size = np.prod(array_shape) * 4  # 4 bytes per float32
        shm = shared_memory.SharedMemory(create=True, size=array_size)
        shared_memories.append(shm)

        sem_producer = Semaphore(0)
        sem_consumer = Semaphore(0)
        semaphores.append((sem_producer, sem_consumer))

        producer_process = mp.Process(target=producer_shared_memory, 
                                      args=(shm.name, array_shape, num_arrays, sem_producer, sem_consumer, i))
        consumer_process = mp.Process(target=consumer_shared_memory, 
                                      args=(shm.name, array_shape, num_arrays, sem_producer, sem_consumer, i))
        
        processes.extend([producer_process, consumer_process])
        producer_process.start()
        consumer_process.start()

    for process in processes:
        process.join()

    end_time = time.time()

    for shm in shared_memories:
        shm.close()
        shm.unlink()

    return end_time - start_time

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

if __name__ == "__main__":
    array_shapes = [(32, 1800)]
    num_arrays = 1000
    num_pairs = 4  # Number of producer-consumer pairs

    print("Testing Multiple Shared Memory Segments:")
    for shape in array_shapes:
        print(f"\nArray shape: {shape}")
        
        start_memory = get_memory_usage()
        shared_memory_time = test_multiple_shared_memory(shape, num_arrays, num_pairs)
        end_memory = get_memory_usage()
        shared_memory_memory = end_memory - start_memory
        
        print(f"Total Time: {shared_memory_time:.4f} seconds")
        print(f"Total Memory Usage: {shared_memory_memory:.2f} MB")
        print(f"Average Time per Pair: {shared_memory_time/num_pairs:.4f} seconds")
        print(f"Average Memory Usage per Pair: {shared_memory_memory/num_pairs:.2f} MB")