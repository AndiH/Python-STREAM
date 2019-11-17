import timeit

from numpy import float64 as real_t

class state_o(object):
	def __init__(self, namespace_, inner_, outer_):
		self.namespace = namespace_
		self.inner = inner_
		self.outer = outer_
		
def timeWrapper(func, setup_="pass", state=None):
	if state == None:
		state = state_o(None, 1, 1)
	return timeit.repeat(func, setup=setup_, globals=state.namespace, number=state.inner, repeat=state.outer)

def runBenchmark(times, init, state):
	for key, value in times.items():
		times[key] = timeWrapper(value, init, state)

def benchmark(funcs, init, state):
	results = {}
	for key, value in funcs.items():
		results[key] = timeWrapper(value, init, state)
	return results

def print_stat(data):
	print("{}".format(my_stat_array(data)))

class my_stat_array(object):
	def __init__(self, array):
		self.array = array
	def __format__(self, format_spec=None):
		from numpy import mean, min, max, std, median
		ansi_colors = {}
		ansi_colors["red"] = "\033[91m"
		ansi_colors["green"] = "\033[92m"
		ansi_colors["cyan"] = "\033[96m"
		ansi_colors["purple"] = "\033[95m"
		ansi_colors["none"] = "\033[00m"

		if not format_spec:
			_precision = 6
		else:
			split_format = format_spec.split(".")
			__decimals = split_format[0]  # not yet supported
			__precision = split_format[1]
			if not __precision:
				_precision = 6
			else:
				_precision = __precision

		return "({red}mean{none} ± {purple}σ{none}): {red}{:>.{prec}f}{none} ± {purple}{:>.{prec}f}{none}; {cyan}min{none}: {cyan}{:.{prec}}{none}, med: {:.{prec}}, max {:.{prec}}".format(
			mean(self.array), 
			std(self.array), 
			min(self.array),
			median(self.array),
			max(self.array),
			prec=_precision,
			**ansi_colors
		)
