# from infra import real_t
from infra import state_o, benchmark
import inspect

def init(value, length):
	return [value for i in range(length)]

def copy(lhs, rhs):
	for i in range(len(lhs)):
		lhs[i] = rhs[i]

def scale(lhs, rhs, alpha):
	for i in range(len(lhs)):
		lhs[i] = alpha * rhs[i]

def add(lhs, rhs1, rhs2):
	for i in range(len(lhs)):
		lhs[i] = rhs1[i] + rhs2[i]

def triad(lhs, rhs1, rhs2, alpha):
	for i in range(len(lhs)):
		lhs[i] = rhs1[i] + alpha * rhs2[i]

def bench(arrayLength, dtype, innerRepeat, outerRepeat):
	setup = """
		a = init(real_t(1), arrayLength); 
		b = init(real_t(2), arrayLength); 
		c = init(real_t(3), arrayLength); 
		alpha = real_t(42);
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
		"init": init,
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
