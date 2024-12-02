

def terminate_process(termination_event, progress_var,progress_queue):
    """
    Signal termination of the batch process.
    """
    termination_event.set()  # Signal to stop processing
    progress_queue.put(None)  # Stop the queue listener
    progress_var.set(0)  # Reset progress bar
    print("Termination signal sent.")

