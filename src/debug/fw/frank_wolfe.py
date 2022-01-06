from typing import Callable, Tuple
import numpy as np
from scipy.optimize import linprog 


def FF(A, mu):
	def f(x):
		return x @ (A + (mu - 1) * np.eye(A.shape[0])) @ x - mu * np.sum(x)
	
	def grad(x):
		return 2. * (A + (mu - 1) * np.eye(A.shape[0])) @ x - mu
	
	def grad_min(x):
		return linprog(grad(x), bounds = [(0., 1.) for _ in range(x.shape[0])]).x
	
	def step_calc(x, s, k):
		return 2 / (2 + k)
	
	def duality_gap(x, s):
		return grad(x) @ (x - s)
	
	return f, grad, grad_min, step_calc, duality_gap


def frank_wolfe(
	x: np.ndarray, 
	duality_gap: Callable[[np.ndarray, np.ndarray], float], 
	gradient_minimizer: Callable[[np.ndarray], np.ndarray], 
	step_calculator: Callable[[np.ndarray, np.ndarray, int], float], 
	tol: float = 1e-4, 
	max_iters: int = 1000
) -> Tuple[np.ndarray, int]:
	"""
	Generalized structure for Frank-Wolfe conditional gradient algorithm.
	The `gradient_minimizer` returns the solution to $\\argmin_{s \in C} <\\nabla f(x), s>$.
	"""
	s = gradient_minimizer(x)
	gap = duality_gap(x, s)
	counter = 1
	while gap >= tol and counter < max_iters:
		step_size = step_calculator(x_old, s, counter)
		x = x + step_size * (s - x)
		s = gradient_minimizer(x)
		gap = duality_gap(x, s)
		counter += 1
	return x, counter 