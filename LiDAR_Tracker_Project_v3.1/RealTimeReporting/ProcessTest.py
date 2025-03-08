import time
import multiprocessing

def worker(process_id, shared_queue, stop_event):
    """Simple worker function that puts numbers in the queue"""
    while not stop_event.is_set():  # Check if stop signal is received
        shared_queue.put(f"Process {process_id}: {time.time()}")
        time.sleep(1)  # Simulate work

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # Ensures compatibility in Ubuntu

    try:
        with multiprocessing.Manager() as manager:
            shared_queue = manager.Queue()
            stop_event = manager.Event()

            # Start 3 worker processes
            processes = [
                multiprocessing.Process(target=worker, args=(i, shared_queue, stop_event))
                for i in range(3)
            ]

            for p in processes:
                p.start()

            # Run for 5 seconds, then stop
            start_time = time.time()
            while time.time() - start_time < 5:
                if not shared_queue.empty():
                    print(shared_queue.get())  # Print messages from workers

            # Signal workers to stop
            stop_event.set()

            # Cleanup
            for p in processes:
                p.join()

            print("Multiprocessing test complete!")

    except Exception as e:
        print("Error:", e)
