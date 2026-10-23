import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time

STUDENT_ID = 230103303
SIZES = [100_000, 1_000_000, 10_000_000]
REPEATS = 3


def pipe_worker(conn):
    try:
        while True:
            data = conn.recv()

            if data is None:
                break

            conn.send((len(data), int(data.sum())))
    finally:
        conn.close()


def shared_worker(conn):
    try:
        while True:
            message = conn.recv()

            if message is None:
                break

            name, shape, dtype_str = message

            shm = shared_memory.SharedMemory(name=name)

            try:
                arr = np.ndarray(
                    shape, dtype=np.dtype(dtype_str), buffer=shm.buf
                )

                result = (len(arr), int(arr.sum()))
            finally:
                del arr
                shm.close()

            conn.send(result)
    finally:
        conn.close()


def benchmark_pipe(data):
    parent, child = mp.Pipe()
    process = mp.Process(target=pipe_worker, args=(child,))
    process.start()
    child.close()

    times = []

    try:
        for _ in range(REPEATS):
            start = time.perf_counter()

            parent.send(data)
            result = parent.recv()

            elapsed = (time.perf_counter() - start) * 1000

            assert result == (len(data), int(data.sum()))
            times.append(elapsed)
    finally:
        parent.send(None)
        parent.close()
        process.join()

    return min(times)


def benchmark_shared(data):
    shm = shared_memory.SharedMemory(
        create=True, size=data.nbytes
    )

    shared_array = np.ndarray(
        data.shape, dtype=data.dtype, buffer=shm.buf
    )

    shared_array[:] = data

    parent, child = mp.Pipe()
    process = mp.Process(target=shared_worker, args=(child,))
    process.start()
    child.close()

    times = []

    try:
        for _ in range(REPEATS):
            start = time.perf_counter()

            parent.send((
                shm.name,
                data.shape,
                data.dtype.str
            ))

            result = parent.recv()

            elapsed = (time.perf_counter() - start) * 1000

            assert result == (len(data), int(data.sum()))
            times.append(elapsed)

    finally:
        parent.send(None)
        parent.close()
        process.join()

        del shared_array
        shm.close()
        shm.unlink()

    return min(times)


def main():
    mp.freeze_support()

    rng = np.random.default_rng(STUDENT_ID)

    lines = [
        "TASK 4 - IPC vs Shared Memory",
        f"Student ID: {STUDENT_ID}",
        f"Repeats: {REPEATS}",
        "",
        f"{'N':>12} | {'Pipe (ms)':>12} | "
        f"{'Shared (ms)':>12} | {'Speedup':>10}",
        "-" * 57
    ]

    for n in SIZES:
        data = rng.integers(
            1, 11, size=n, dtype=np.int64
        )

        pipe_time = benchmark_pipe(data)
        shared_time = benchmark_shared(data)

        speedup = pipe_time / shared_time

        lines.append(
            f"{n:>12,} | "
            f"{pipe_time:>12.3f} | "
            f"{shared_time:>12.3f} | "
            f"{speedup:>9.2f}x"
        )

    lines.extend([
        "",
        "Path A: Pipe with Pickle serialization",
        "Path B: Zero-copy shared memory access",
        "Timing: Request-response including array sum.",
        "Shared memory initialization excluded.",
        "Process startup excluded.",
        "Benchmark Status: PASS"
    ])

    output = "\n".join(lines)

    print(output)

    with open(
        "task4_output.txt", "w", encoding="utf-8"
    ) as file:
        file.write(output + "\n")

    print("\nResults saved to task4_output.txt")


if __name__ == "__main__":
    main()