import os
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103303
TOTAL_TASKS = 10_000
HASH_REPEATS = 500

CORES = os.cpu_count() or 4

POOL_SIZES = sorted(set([
    1,
    max(1, CORES // 2),
    CORES,
    CORES * 2,
    CORES * 4,
    CORES * 16,
    256
]))

DATA = (
    str(STUDENT_ID).encode() * 1024
)[:1024]


def cpu_task(task_id):
    data = DATA + task_id.to_bytes(4, "little")

    for _ in range(HASH_REPEATS):
        data = hashlib.sha256(data).digest()

    return data


def benchmark(pool_size):
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        results = list(
            executor.map(cpu_task, range(TOTAL_TASKS))
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    assert len(results) == TOTAL_TASKS

    return elapsed_ms


def main():
    print("TASK 5 - Thread Oversubscription Threshold")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical Cores: {CORES}")
    print(f"Total Tasks: {TOTAL_TASKS:,}")
    print(f"Hash Repeats: {HASH_REPEATS}")
    print("-" * 75)

    results = []

    cpu_task(0)

    for pool_size in POOL_SIZES:
        print(f"Testing pool size {pool_size}...")

        elapsed = benchmark(pool_size)
        results.append((pool_size, elapsed))

    baseline = results[0][1]

    lines = [
        "TASK 5 - Thread Oversubscription Threshold",
        f"Student ID: {STUDENT_ID}",
        f"Logical Cores: {CORES}",
        f"Total Tasks: {TOTAL_TASKS:,}",
        f"Hash Repeats: {HASH_REPEATS}",
        "",
        f"{'Pool Size':>12} | "
        f"{'Execution Time (ms)':>20} | "
        f"{'Speedup':>10} | Status",
        "-" * 75
    ]

    previous_time = None
    threshold = None

    for pool_size, elapsed in results:
        speedup = baseline / elapsed

        if previous_time is None:
            status = "BASELINE"

        elif elapsed > previous_time:
            if threshold is None:
                threshold = pool_size
                status = "DEGRADATION START"
            else:
                status = "SLOWER"

        else:
            status = "OK"

        lines.append(
            f"{pool_size:>12} | "
            f"{elapsed:>20.3f} | "
            f"{speedup:>9.3f}x | {status}"
        )

        previous_time = elapsed

    lines.append("")

    if threshold is not None:
        lines.append(
            f"First observed degradation at pool size: {threshold}"
        )
    else:
        lines.append(
            "No degradation detected in tested pool sizes."
        )

    lines.append(
        "Note: Python GIL may limit CPU-bound thread parallelism."
    )

    output = "\n".join(lines)

    print("\n" + output)

    with open(
        "task5_output.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(output + "\n")

    print("\nResults saved to task5_output.txt")


if __name__ == "__main__":
    main()