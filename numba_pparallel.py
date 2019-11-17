# from infra import real_t
import inspect
import logging

logger = logging.getLogger(__name__)

from infra import state_o, benchmark

from numba import jit, prange, npyufunc

@jit('(float64[:], float64[:])', nopython=True, parallel=True)
def copy(lhs, rhs):
	for i in prange(len(lhs)):
		lhs[i] = rhs[i]

@jit('(float64[:], float64[:], float64)', nopython=True, parallel=True)
def scale(lhs, rhs, alpha):
	for i in prange(len(lhs)):
		lhs[i] = alpha * rhs[i]

@jit('(float64[:], float64[:], float64[:])', nopython=True, parallel=True)
def add(lhs, rhs1, rhs2):
	for i in prange(len(lhs)):
		lhs[i] = rhs1[i] + rhs2[i]

@jit('(float64[:], float64[:], float64[:], float64)', nopython=True, parallel=True)
def triad(lhs, rhs1, rhs2, alpha):
	for i in prange(len(lhs)):
		lhs[i] = rhs1[i] + alpha * rhs2[i]

def bench(arrayLength, dtype, innerRepeat, outerRepeat):
	import numpy as np
	setup = """
		a = np.full(arrayLength, 1, dtype=real_t);
		b = np.full(arrayLength, 2, dtype=real_t);
		c = np.full(arrayLength, 3, dtype=real_t);
		alpha = real_t(42)
	"""
	times = {
		"copy":  'copy(a, b)',
		"scale": 'scale(a, b, alpha)', 
		"add":   'add(a, b, c)',
		"triad": 'triad(a, b, c, alpha)'
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

	logger.debug("NUMBA Threading Layer: {}".format(npyufunc.parallel.threading_layer()))
	logger.debug("NUMBA N Threads: {}".format(npyufunc.parallel.get_thread_count()))
	
	return [
		("COPY", results["copy"]), 
		("SCALE", results["scale"]), 
		("ADD", results["add"]), 
		("TRIAD", results["triad"])
	]
