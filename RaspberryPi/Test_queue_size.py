import numpy as np
import sys

def calculate_array_memory(shape, dtype):
    # Create a sample array
    sample_array = np.zeros(shape, dtype=dtype)
    
    # Calculate the memory usage
    memory_bytes = sample_array.nbytes
    memory_mb = memory_bytes / (1024 * 1024)
    
    return memory_bytes, memory_mb

def calculate_queue_memory(array_shape, array_dtype, queue_size):
    array_bytes, array_mb = calculate_array_memory(array_shape, array_dtype)
    queue_bytes = array_bytes * queue_size
    queue_mb = array_mb * queue_size
    
    return queue_bytes, queue_mb

# Example usage
array_shape = (32, 1800)
array_dtype = np.float16
queue_sizes = [100, 500, 1000, 5000, 10000]

print(f"Memory usage for a single array of shape {array_shape} with dtype {array_dtype}:")
single_array_bytes, single_array_mb = calculate_array_memory(array_shape, array_dtype)
print(f"  {single_array_bytes} bytes ({single_array_mb:.2f} MB)")

print("\nMemory usage for queues of different sizes:")
for size in queue_sizes:
    queue_bytes, queue_mb = calculate_queue_memory(array_shape, array_dtype, size)
    print(f"  Queue size {size}: {queue_bytes} bytes ({queue_mb:.2f} MB)")

print("\nRecommended queue sizes based on available memory:")
available_memory_options = [1024, 4096, 8192, 16384]  # in MB
for memory in available_memory_options:
    max_queue_size = memory / single_array_mb
    print(f"  For {memory} MB of available memory: approximately {int(max_queue_size)} items")