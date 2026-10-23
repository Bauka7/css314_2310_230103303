import os
import time
import sys

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

STUDENT_ID = 230103303
ARRAY_SIZE = 2_000_000

# Generate array using Student ID seed
rng = np.random.default_rng(STUDENT_ID)
data = rng.integers(1, 11, size=ARRAY_SIZE, dtype=np.int64)

# Use available CPU threads
NUM_THREADS = min(os.cpu_count() or 4, 8)
set_num_threads(NUM_THREADS)


# MODE A: Serial inclusive prefix scan
@njit
def serial_scan(arr):
    n = len(arr)
    result = np.empty(n, dtype=np.int64)

    total = 0
    for i in range(n):
        total += arr[i]
        result[i] = total

    return result


# MODE B: Parallel chunked prefix scan
@njit(parallel=True)
def parallel_scan(arr, num_threads):
    n = len(arr)
    result = np.empty(n, dtype=np.int64)

    chunk_size = (n + num_threads - 1) // num_threads
    chunk_sums = np.zeros(num_threads, dtype=np.int64)

    # Pass 1: Calculate local prefix sums
    for t in prange(num_threads):
        start = t * chunk_size
        end = min(start + chunk_size, n)

        local_sum = 0

        for i in range(start, end):
            local_sum += arr[i]
            result[i] = local_sum

        chunk_sums[t] = local_sum

    # Calculate offsets
    offsets = np.zeros(num_threads, dtype=np.int64)

    running_sum = 0
    for t in range(num_threads):
        offsets[t] = running_sum
        running_sum += chunk_sums[t]

    # Pass 2: Apply offsets in parallel
    for t in prange(num_threads):
        start = t * chunk_size
        end = min(start + chunk_size, n)

        offset = offsets[t]

        for i in range(start, end):
            result[i] += offset

    return result


def main():
    print("TASK 3 - Parallel Prefix Scan")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Array Size: {ARRAY_SIZE:,}")
    print(f"Threads: {get_num_threads()}")
    print("-" * 50)

    # Warm-up: exclude JIT compilation time
    serial_scan(data[:100])
    parallel_scan(data[:100], NUM_THREADS)

    # Serial benchmark
    start = time.perf_counter()
    serial_output = serial_scan(data)
    serial_time = (time.perf_counter() - start) * 1000

    # Parallel benchmark
    start = time.perf_counter()
    parallel_output = parallel_scan(data, NUM_THREADS)
    parallel_time = (time.perf_counter() - start) * 1000

    # Verification
    try:
        assert parallel_output[-1] == serial_output[-1]
        status = "PASS"
    except AssertionError:
        status = "FAIL"

    speedup = serial_time / parallel_time if parallel_time > 0 else 0

    output = f"""
TASK 3 - Parallel Prefix Scan
Student ID: {STUDENT_ID}
Array Size: {ARRAY_SIZE:,}
Threads: {get_num_threads()}

Serial Time: {serial_time:.3f} ms
Parallel Time: {parallel_time:.3f} ms
Speedup: {speedup:.3f}x
Verification Status: {status}

Serial Final Sum: {serial_output[-1]}
Parallel Final Sum: {parallel_output[-1]}
"""

    print(output)

    with open("task3_output.txt", "w", encoding="utf-8") as file:
        file.write(output)

    if status == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()