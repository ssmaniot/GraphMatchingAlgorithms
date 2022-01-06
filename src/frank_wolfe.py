from typing import Callable, Tuple
import numpy as np


def frank_wolfe(
	x: np.ndarray, 
	gradient: Callable[[np.ndarray], np.ndarray],
	duality_gap: Callable[[np.ndarray, np.ndarray, np.ndarray], float], 
	gradient_minimizer: Callable[[np.ndarray], np.ndarray], 
	step_calculator: Callable[[np.ndarray, np.ndarray, int], float], 
	tol: float = 1e-4, 
	max_iters: int = 1000
) -> Tuple[np.ndarray, int]:
	"""
	Generalized structure for Frank-Wolfe conditional gradient algorithm.
	The `gradient_minimizer` returns the solution to $\\argmin_{s \in C} <\\nabla f(x), s>$.
	"""
	# print(x)
	grad = gradient(x)
	s = gradient_minimizer(grad)
	gap = duality_gap(grad, x, s)
	counter = 1
	while gap >= tol and counter < max_iters:
		step_size = step_calculator(x, s, counter)
		x = x + step_size * (s - x)
		grad = gradient(x)
		s = gradient_minimizer(grad)
		gap = duality_gap(grad, x, s)
		counter += 1
	return x, counter 