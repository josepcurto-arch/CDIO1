import threading
import multiprocessing
import numpy as np
import pickle
import time

DATA = np.random.randn(10000, 10000)


def dummy_task(name):
    print(f"{name} started")
    time.sleep(20)
    print(f"{name} finished")


def input_output_intensive_task(name):
    print(f"{name} started")
    with open("data", "wb") as f:
        pickle.dump(DATA, f)
    print(f"{name} finished")


def cpu_intensive_task(name):
    print(f"{name} started")
    _ = np.sum(DATA)
    print(f"{name} finished")


def baseline_example(task):
    print("Starting single process/thread...")
    for i in range(4):
        if task == "dummy":
            dummy_task(f"Baseline-{i + 1}")
        elif task == "input_output":
            input_output_intensive_task(f"Baseline-{i + 1}")
        elif task == "cpu_intensive":
            cpu_intensive_task(f"Baseline-{i + 1}")
    print("Single process/thread completed.")


# Using threading
def threading_example(task):
    print("Starting threads...")
    threads = []
    for i in range(4):
        if task == "dummy":
            thread = threading.Thread(target=dummy_task, args=(f"Thread-{i + 1}",))
            threads.append(thread)
            thread.start()
        elif task == "input_output":
            thread = threading.Thread(target=input_output_intensive_task, args=(f"Thread-{i + 1}",))
            threads.append(thread)
            thread.start()
        elif task == "cpu_intensive":
            thread = threading.Thread(target=cpu_intensive_task, args=(f"Thread-{i + 1}",))
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join()
    print("Threads completed.")


# Using multiprocessing
def multiprocessing_example(task):
    print("Starting processes...")
    processes = []
    for i in range(4):
        if task == "dummy":
            process = multiprocessing.Process(target=dummy_task, args=(f"Process-{i + 1}",))
            processes.append(process)
            process.start()
        elif task == "input_output":
            process = multiprocessing.Process(target=input_output_intensive_task, args=(f"Process-{i + 1}",))
            processes.append(process)
            process.start()
        elif task == "cpu_intensive":
            process = multiprocessing.Process(target=cpu_intensive_task, args=(f"Process-{i + 1}",))
            processes.append(process)
            process.start()

    for process in processes:
        process.join()
    print("Processes completed.")


def benchmark(func, task_type):
    start = time.time()
    func(task_type)
    end = time.time()
    return end - start

def format_row(row):
    return " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))


if __name__ == "__main__":
    task_types = ['dummy', 'input_output', 'cpu_intensive']
    methods = [
        ("Baseline Example (no threading, no multiprocessing)", baseline_example, "Baseline"),
        ("Threading Example", threading_example, "Threading"),
        ("Multiprocessing Example", multiprocessing_example, "Multiprocessing"),
    ]

    results = []  # store rows for the table

    for task_type in task_types:
        print(f"\n--- Task type: {task_type} ---")
        row = [task_type]
        for desc, func, label in methods:
            print(f"\n--- {desc} ---")
            elapsed = benchmark(func, task_type)
            print(f"{label} execution time: {elapsed:.2f} seconds")
            row.append(f"{elapsed:.2f}")
        results.append(row)

    # Build summary table manually
    headers = ["Task Type", "Baseline (s)", "Threading (s)", "Multiprocessing (s)"]
    all_rows = [headers] + results

    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in all_rows) for i in range(len(headers))]

    print("\n=== Summary Table ===")
    print(format_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in results:
        print(format_row(row))
