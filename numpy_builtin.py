# from infra import real_t
from infra import state_o, benchmark
import inspect

from numpy import copy as np_copy, kron as np_kron, add as np_add

def copy(lhs, rhs):
	lhs = np_copy(rhs)

def scale(lhs, rhs, alpha):
	lhs = np_kron(alpha, rhs)

def add(lhs, rhs1, rhs2):
	lhs = np_add(rhs1, rhs2)

def triad(lhs, rhs1, rhs2, alpha):
	lhs = np_add(rhs1, np_kron(alpha, rhs2))

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
	return [
		("COPY", results["copy"]), 
		("SCALE", results["scale"]), 
		("ADD", results["add"]), 
		("TRIAD", results["triad"])
	]
