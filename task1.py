"""Task 1 — Parallel spatial-cache profiling (row-major vs column-major)."""
import argparse
import os
import time
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

try:
    import psutil
except ImportError:
    psutil = None


@njit(parallel=True)
def sum_row_major(a):
    total = 0.0
    for i in prange(a.shape[0]):
        partial = 0.0
        for j in range(a.shape[1]):
            partial += a[i, j]
        total += partial
    return total


@njit(parallel=True)
def sum_column_major(a):
    total = 0.0
    for j in prange(a.shape[1]):
        partial = 0.0
        for i in range(a.shape[0]):
            partial += a[i, j]
        total += partial
    return total


def benchmark(fn, matrix, repeats):
    times = []
    results = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(matrix)
        times.append((time.perf_counter() - start) * 1000)
        results.append(result)
    return min(times), results[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student-id', type=int, required=True, help='Your real numeric Student ID')
    parser.add_argument('--size', type=int, default=8000, help='Matrix dimension (default 8000)')
    parser.add_argument('--repeats', type=int, default=3, help='Timed runs per traversal')
    parser.add_argument('--output', default='task1_output.txt', help='Output text file')
    args = parser.parse_args()
    if args.size <= 0 or args.repeats <= 0:
        parser.error('--size and --repeats must be positive')

    logical = os.cpu_count() or 1
    physical = (psutil.cpu_count(logical=False) if psutil else None) or logical
    # Numba's configured maximum can be less than detected physical cores.
    from numba import config
    num_threads = min(physical, config.NUMBA_NUM_THREADS)
    set_num_threads(num_threads)

    print('Allocating the matrix (~{:.1f} MiB)...'.format(args.size**2 * 8 / 2**20), flush=True)
    rng = np.random.default_rng(args.student_id)
    matrix = rng.random((args.size, args.size), dtype=np.float64)
    assert matrix.flags.c_contiguous

    # Compile the two JIT kernels before timing to exclude compilation overhead.
    sum_row_major(matrix[:2, :2])
    sum_column_major(matrix[:2, :2])

    a_ms, a_sum = benchmark(sum_row_major, matrix, args.repeats)
    b_ms, b_sum = benchmark(sum_column_major, matrix, args.repeats)
    if not np.isclose(a_sum, b_sum, rtol=1e-10, atol=1e-6):
        raise RuntimeError(f'Mode sums disagree: {a_sum} vs {b_sum}')

    lines = [
        'TASK 1 — ROW-MAJOR VS COLUMN-MAJOR PARALLEL CACHE TRAVERSAL',
        f'Student ID: {args.student_id}',
        f'Logical core count: {logical}',
        f'Physical core count: {physical}',
        f'Parallel worker threads: {get_num_threads()}',
        f'Matrix: {args.size} x {args.size} (float64, C-contiguous)',
        f'Timed repetitions per mode: {args.repeats} (best/minimum reported)',
        f'Mode A (Row-Major) runtime: {a_ms:.3f} ms',
        f'Mode B (Column-Major / Strided) runtime: {b_ms:.3f} ms',
        f'Cache penalty factor (B / A): {b_ms / a_ms:.3f}x',
        f'Mode A sum: {a_sum:.8f}',
        f'Mode B sum: {b_sum:.8f}',
        'SUM CHECK: PASSED',
    ]
    output = '\n'.join(lines) + '\n'
    print('\n' + output)
    Path(args.output).write_text(output, encoding='utf-8')
    print(f'Saved: {Path(args.output).resolve()}')


if __name__ == '__main__':
    main()
