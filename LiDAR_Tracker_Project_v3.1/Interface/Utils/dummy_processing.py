from threading import Thread, Event
from functools import partial
from p_tqdm import p_umap
import time

class DummyProcessor:
    def __init__(self):
        self.tasks = []
        self.termination_event = Event()  # Event to signal termination

    def generate_dummy_tasks(self, num_tasks):
        """Generate dummy tasks for demonstration."""
        self.tasks = [{"task_id": i, "data": i * 10} for i in range(num_tasks)]

    def process_task(self, task, multiplier):
        """Simulate processing a single task."""
        for _ in range(10):  # Simulate a long computation in chunks
            if self.termination_event.is_set():
                print(f"Task {task['task_id']} terminated.")
                return None
            time.sleep(0.1)  # Simulate computation delay
        result = task["data"] * multiplier
        print(f"Processed Task {task['task_id']} with result: {result}")
        return result

    def run_tasks(self, num_tasks, multiplier):
        """Run tasks using a separate thread and p_umap."""
        self.generate_dummy_tasks(num_tasks)
        print("Starting multi-task processing...")

        self.termination_event.clear()  # Ensure the event is not set
        self.processing_thread = Thread(
            target=self._process_tasks_in_thread,
            args=(multiplier,)
        )
        self.processing_thread.start()

    def _process_tasks_in_thread(self, multiplier):
        """Process tasks in a separate thread."""
        print("Processing tasks...")
        try:
            results = p_umap(partial(self.process_task, multiplier=multiplier), self.tasks)
            print("All tasks processed. Results:", results)
        except Exception as e:
            print(f"Processing interrupted: {e}")

    def terminate_tasks(self):
        """Signal termination of the processing tasks."""
        print("Termination signal sent. Stopping tasks...")
        self.termination_event.set()  # Set the event to stop ongoing tasks
        if self.processing_thread.is_alive():
            self.processing_thread.join()  # Wait for the thread to finish
        print("All tasks terminated.")
