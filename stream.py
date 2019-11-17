#!/usr/bin/env python3
import logging
import importlib

import infra
import numpy as np

all_benchmarks = [
	"classic",
	"numpy_colon",
	"numpy_builtin",
	"numba_jit",
	"numba_guvectorize",
	"numba_vectorize_cuda",
	"numba_parallel",
	"numba_pparallel",
	"numba_cuda"
]
def main(arrayLength=100000, innerRepeat=10, outerRepeat=3, bench=["all"]):
	dtype = infra.real_t
	assembled = {}

	for benchmark in bench:
		if benchmark is "all" or benchmark in all_benchmarks:
			benchmark_module = importlib.import_module(benchmark)
			assembled[benchmark] = benchmark_module.bench(arrayLength, dtype, innerRepeat, outerRepeat)

	print("### STREAM ###")
	print("# Array length: {}".format(arrayLength))
	print("# Inner repeat: {}; Outer repeat {}.".format(innerRepeat, outerRepeat))

	array_size = ((arrayLength * np.dtype(dtype).itemsize ) / (2**20), "MiB")

	bandwidth_conversion_factor = {
		"copy":  2,
		"scale": 2,
		"add":   3,
		"triad": 3
	}
	for name, benchclass in assembled.items():
		print("## Benchmark {}".format(name))
		for (bench, time) in benchclass:
			print("Time (for {} repeats) {:>8} (/ s): {:.4}".format(innerRepeat, bench, infra.my_stat_array(time)))
		for (bench, time) in benchclass:
			print("{}\tBandwidth (single) / {}: {:.6}".format(
				bench.upper(),
				"GiB/s",
				infra.my_stat_array(
					[(array_size[0] * bandwidth_conversion_factor[bench.lower()] / 2**10) / (_t / innerRepeat)
					for _t 
					in time]
					)
				))

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description="Python Numba STREAM Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("--length", "-l", type=int, help="Length of STREAM arrays", default=100000)
	parser.add_argument("--repeat", "-r", type=int, help="Repeat experiment this often", default=10)
	parser.add_argument("--outer-repeat", type=int, help="Repeat repeating this often, minimum is taken", default=3)

	parser.add_argument("-v", "--verbose", action='count', help="Increase verbose output; use multiple -v to increase verbosity. Might not be available for all sub-benchmarks", default=0)

	parser.add_argument("bench", default="all", nargs='*', choices=all_benchmarks + ["all"], help="Implementation to run")

	args = parser.parse_args()

	log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
	log_level = log_levels[min(len(log_levels) - 1, args.verbose)]

	logging.basicConfig(level=log_level)

	main(arrayLength=args.length, innerRepeat=args.repeat, outerRepeat=args.outer_repeat, bench=args.bench)
