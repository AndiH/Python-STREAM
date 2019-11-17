# from infra import real_t
import inspect

from infra import state_o, benchmark

import classic

from numba import guvectorize
copy  = guvectorize('(float64[:], float64[:])', '(n),(n)')(classic.copy)
scale = guvectorize('(float64[:], float64[:], float64)', '(n),(n),()')(classic.scale)
add   = guvectorize('(float64[:], float64[:], float64[:])', '(n),(n),(n)')(classic.add)
triad = guvectorize('(float64[:], float64[:], float64[:], float64)', '(n),(n),(n),()')(classic.triad)

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
			"add": 'add(a, b, c)',
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
