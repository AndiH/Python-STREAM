# from infra import real_t
import inspect

from infra import state_o, benchmark

import classic

from numba import vectorize

@vectorize('float64(float64)', target="cuda")
def copy(rhs_i):
	return rhs_i

@vectorize('float64(float64, float64)', target="cuda")
def scale(rhs_i, alpha):
	return alpha * rhs_i

@vectorize('float64(float64, float64)', target="cuda")
def add(rhs1_i, rhs2_i):
	return rhs1_i + rhs2_i

@vectorize('float64(float64, float64, float64)', target="cuda")
def triad(rhs1_i, rhs2_i, alpha):
	return rhs1_i + alpha * rhs2_i

def bench(arrayLength, dtype, innerRepeat, outerRepeat):
	import numpy as np
	setup = """
		a = np.full(arrayLength, 1, dtype=real_t);
		b = np.full(arrayLength, 2, dtype=real_t);
		c = np.full(arrayLength, 3, dtype=real_t);
		alpha = real_t(42)
	"""
	times = {
			"copy":  'copy(a)',
			"scale": 'scale(a, alpha)',
			"add": 'add(a, b)',
			"triad": 'triad(a, b, alpha)'
		}
	namespace = {
		"copy": copy,
		"scale": scale,
		"add": add,
		"triad": triad,
		"np": np,
		"arrayLength": arrayLength,
		"real_t": dtype
	}
	state = state_o(namespace, innerRepeat, outerRepeat)
	results = benchmark(times, inspect.cleandoc(setup), state)
	return [
		("COPY", results["copy"]),
		("SCALE", results["scale"]),
		("ADD", results["add"]),
		("TRIAD", results["triad"])
	]
