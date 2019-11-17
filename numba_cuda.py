# from infra import real_t
from infra import state_o, benchmark
import inspect

from numba import cuda

@cuda.jit('void(float64[:], float64[:])')
def copy(lhs, rhs):
	i = cuda.grid(1)
	if i < lhs.shape[0]:
		lhs[i] = rhs[i]

@cuda.jit('void(float64[:], float64[:], float64)')
def scale(lhs, rhs, alpha):
	i = cuda.grid(1)
	if i < lhs.shape[0]:
		lhs[i] = alpha * rhs[i]

@cuda.jit('void(float64[:], float64[:], float64[:])')
def add(lhs, rhs1, rhs2):
	i = cuda.grid(1)
	if i < lhs.shape[0]:
		lhs[i] = rhs1[i] + rhs2[i]

@cuda.jit('void(float64[:], float64[:], float64[:], float64)')
def triad(lhs, rhs1, rhs2, alpha):
	i = cuda.grid(1)
	if i < lhs.shape[0]:
		lhs[i] = rhs1[i] + alpha * rhs2[i]

def bench(arrayLength, dtype, innerRepeat, outerRepeat):
	import numpy as np
	setup = """
		a = np.full(arrayLength, 1, dtype=real_t);
		b = np.full(arrayLength, 2, dtype=real_t);
		c = np.full(arrayLength, 3, dtype=real_t);
		alpha = real_t(42)

		stream_a = cuda.stream()
		stream_b = cuda.stream()
		stream_c = cuda.stream()
		d_a = cuda.to_device(a, stream_a)
		d_b = cuda.to_device(b, stream_b)
		d_c = cuda.to_device(c, stream_c)

		n_threads = 256
		n_blocks = np.ceil(arrayLength / n_threads).astype("int32")
	"""
	times = {
		"copy":  'copy[n_blocks, n_threads](d_a, d_b)',
		"scale": 'scale(a, b, alpha)', 
		"add":   'add(a, b, c)',
		"triad": 'triad(a, b, c, alpha)'
	}
	namespace = {
		"np": np,
		"cuda": cuda,
		"copy": copy,
		"scale": scale,
		"add": add,
		"triad": triad,
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
