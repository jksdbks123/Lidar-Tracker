import time
import sys

def print_progress_bar(progress):
    # progress is an integer from 0 to 100
    bar_length = 50  # length of the progress bar
    filled_length = int(bar_length * progress // 100)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    # Carriage return to overwrite the same line
    sys.stdout.write(f"\rProgress: [{bar}] {progress}%")
    sys.stdout.flush()

def main():
    cycle_duration = 2 * 60 * 60  # 2 hours in seconds
    steps = 100
    step_duration = cycle_duration / steps

    while True:  # Loop indefinitely
        for i in range(0, steps + 1):
            print_progress_bar(i)
            time.sleep(step_duration)
        # Once done, it will loop again, simulating a new run

if __name__ == "__main__":
    main()
