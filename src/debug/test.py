import sys
import os
import numpy as np
import scipy.sparse as sp 
from scipy.sparse.linalg import eigsh
from scipy.optimize import linprog, line_search
import matplotlib.pyplot as plt 
import itertools 

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from frank_wolfe import frank_wolfe
from experiments import generate_random_graph as grg
from typing import Tuple, Callable


def sigma_minmax(A: np.ndarray) -> Tuple[float, float]:
	"""Computes the minimum and maximum eigenvalues of matrix A and 
	returns their absolute values.
	
	Parameters
	----------
	A : numpy.ndarray 
		A real, symmetric matrix.
	
	Returns
	-------
	lmin : float 
		The absolute values of the minimum eigenvalue of A.
	
	lmax : float 
		The absolute values of the maximum eigenvalue of A.
		
	"""
	w, _ = eigsh(A)
	lmin = np.abs(np.min(w))
	lmax = np.abs(np.max(w))
	return lmin, lmax 
	

def FUN_GRAD(M: np.ndarray, mu: float, copy: bool = False) -> Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
	"""Generates the objective function of the model and its gradient given
	the affinity matrix M and the free parameter mu.
	
	Parameters
	----------
	M : numpy.ndarray
		The affinity matrix of the complement of the association graph.
	
	mu : float 
		The free parameter of the model. It allows to control the definiteness of the 
		quadratic form, which governs the convexity/concavity of the objective function.
	
	copy : bool, optional
		If True, make a local copy of the affinity matrix M to make the returned functions 
		immutable. Otherwise, use a reference to the original matrix. 
	
	Returns 
	-------
	f : function 
		The objective function of the model.
	
	grad : function 
		The gradient of the objective function f. 
		
	"""
	if copy:
		M = M.copy()
	M_bar = M + (mu - 1.) * sp.eye(M.shape[0])
		
	def f(x: np.ndarray) -> float:
		return x @ M_bar @ x - mu * np.sum(x)
		
	def grad(x: np.ndarray) -> np.ndarray:
		return 2. * (M @ x + (mu - 1.) * x) - mu 
		
	return f, grad 
	
	
def GRAD_MIN(grad: Callable[[np.ndarray], np.ndarray], n: int) -> Callable[[np.ndarray], np.ndarray]:
	BDs = [(0., 1.) for _ in range(n)]
	def grad_min(x: np.ndarray) -> np.ndarray:
		res = linprog(grad(x), bounds = BDs)
		return res.x 
	
	return grad_min 


def STEP_CALC(f: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], float]) -> Callable[[np.ndarray, np.ndarray, int], float]:
	def step_calc(x: np.ndarray, s: np.ndarray, k: int) -> float:
		return 2 / (2 + k)
	
	return step_calc 


def vertices(n: int) -> np.ndarray:
	return np.array([list(i) for i in itertools.product([0, 1], repeat = n)])


def duality_gap(x: np.ndarray, s: np.ndarray) -> float:
	return grad(x) @ (x - s)
	

def main() -> None:

	n = 10
	p = 0.9
	steps = 10
	G = grg(n, p)
	print('Adjacency Matrix of G\n')
	print(G, end='\n\n')
	
	M = sp.csr_matrix(1 - G - np.eye(n))
	
	I = sp.eye(n)
	
	_, mu = sigma_minmax(M - sp.eye(n))
	mu = -(mu+0.1)
	w, _ = sp.linalg.eigsh(M + (mu - 1) * I)
	print(w)
	
	def f(x):
		return x @ (M + (mu - 1) * I) @ x - mu * np.sum(x)
	
	def grad(x):
		return 2. * (M + (mu - 1) * I) @ x - mu 
	
	tests = 1
	max_iters = 10000
	tol = 1.e-15
	res = np.empty(tests * n).reshape((tests, n))
	iters = np.empty(tests * steps).reshape((tests, steps)).astype('int')
	grad_min = GRAD_MIN(grad, n)
	step_calc = STEP_CALC(f, grad)
	
	def duality_gap(x, s):
		return grad(x) @ (x - s)
	
	for i in range(tests):
		x = np.ones(n) #np.random.rand(n)
		print(f'\rtest {i+1}/{tests}', end='')
		x, _ = frank_wolfe(x, duality_gap, grad_min, step_calc, tol=tol, max_iters=max_iters)
		res[i,:] = x
	print('')
	print(np.round(res, 3))
	
	idx = np.nonzero(res[0,:] == 1.)[0]
	print('\nSUBSET:\n')
	print(G[np.ix_(idx, idx)])

if __name__ == '__main__':
	main()