import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103303
NUM_THREADS = 4
TOTAL_POINTS = 40_000_000
POINTS_PER_THREAD = TOTAL_POINTS // NUM_THREADS


def shared_prng():
    rng = random.Random(STUDENT_ID)
    lock = threading.Lock()

    def worker(thread_index):
        inside = 0

        for _ in range(POINTS_PER_THREAD):
            with lock:
                x = rng.random()
                y = rng.random()

            if x * x + y * y <= 1.0:
                inside += 1

        return inside

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(worker, range(NUM_THREADS)))

    elapsed_ms = (time.perf_counter() - start) * 1000
    pi_value = 4.0 * sum(results) / TOTAL_POINTS

    return pi_value, abs(pi_value - math.pi), elapsed_ms


def independent_prng():

    def worker(thread_index):
        seed = STUDENT_ID * 10 + thread_index
        rng = random.Random(seed)

        inside = 0

        for _ in range(POINTS_PER_THREAD):
            x = rng.random()
            y = rng.random()

            if x * x + y * y <= 1.0:
                inside += 1

        return inside

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(worker, range(NUM_THREADS)))

    elapsed_ms = (time.perf_counter() - start) * 1000
    pi_value = 4.0 * sum(results) / TOTAL_POINTS

    return pi_value, abs(pi_value - math.pi), elapsed_ms


if __name__ == "__main__":
    print("TASK 2 - PRNG State Bottleneck")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Threads: {NUM_THREADS}")
    print(f"Total Samples: {TOTAL_POINTS:,}")
    print("-" * 45)

    print("Running Shared PRNG...")
    shared_pi, shared_error, shared_time = shared_prng()

    print("Running Independent PRNG...")
    independent_pi, independent_error, independent_time = independent_prng()

    speedup = shared_time / independent_time

    output = f"""
TASK 2 - PRNG State Bottleneck
Student ID: {STUDENT_ID}
Threads: {NUM_THREADS}
Total Samples: {TOTAL_POINTS:,}

Shared PRNG:
Computed PI: {shared_pi:.9f}
Absolute Error: {shared_error:.9f}
Wall Time: {shared_time:.3f} ms

Independent PRNG:
Computed PI: {independent_pi:.9f}
Absolute Error: {independent_error:.9f}
Wall Time: {independent_time:.3f} ms

Speedup Factor: {speedup:.3f}x
"""

    print(output)

    with open("task2_output.txt", "w", encoding="utf-8") as file:
        file.write(output)

    print("Results saved to task2_output.txt")