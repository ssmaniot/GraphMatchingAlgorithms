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
	w, _ = eigsh(A, k = A.shape[0] - 1)
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
	
	
def GRAD_MIN(n: int) -> Callable[[np.ndarray], np.ndarray]:
	BDs = [(0., 1.) for _ in range(n)]
	def grad_min(grad: np.ndarray) -> np.ndarray:
		res = linprog(grad, bounds = BDs)
		return res.x 
	
	return grad_min 


def STEP_CALC() -> Callable[[np.ndarray, np.ndarray, int], float]:
	def step_calc(x: np.ndarray, s: np.ndarray, k: int) -> float:
		return 2 / (2 + k)
	
	return step_calc 


def vertices(n: int) -> np.ndarray:
	return np.array([list(i) for i in itertools.product([0, 1], repeat = n)])


def DUALITY_GAP() -> Callable[[np.ndarray, np.ndarray], float]:
	def duality_gap(grad: np.ndarray, x: np.ndarray, s: np.ndarray) -> float:
		return grad @ (x - s)
	
	return duality_gap
	

def main() -> None:

	N = 10
	p = 0.5
	steps = 10
	if True:
		G1 = grg(N, p)
		G2 = grg(N, p)
		G = 1 - (np.kron(G1, G2) + np.kron(1 - G1 - np.eye(N), 1 - G2 - np.eye(N))) - np.eye(N*N)
	else:
		G = grg(N, p)
		G = np.array([
			[0, 1, 1, 0, 1],
			[1, 0, 1, 0, 1],
			[1, 1, 0, 0, 1],
			[0, 0, 0, 0, 0],
			[1, 1, 1, 0, 0]
		])
	n = G.shape[0]
	print(f'avg. deg: {np.mean(np.sum(G, axis = 0))}\n')
	A = sp.csr_matrix(1 - G - sp.eye(n))
	
	lmin, lmax = sigma_minmax(A - sp.eye(n))
	
	MU = np.linspace(lmin + 0.1, -(lmax + 0.1), num = steps)
	
	tests = 10
	max_iters = 2000
	tol = 1.e-10
	res = np.empty(tests * n).reshape((tests, n))
	iters = np.empty(tests * steps).reshape((tests, steps)).astype('int')
	for i in range(tests):
		x = np.random.rand(n)
		print(f'\rtest {i+1}/{tests}', end='')
		for j, mu in enumerate(MU):
			f, grad = FUN_GRAD(A, mu)
			grad_min = GRAD_MIN(A.shape[0])
			step_calc = STEP_CALC()
			duality_gap = DUALITY_GAP()
			
			x, iters[i,j] = frank_wolfe(x, grad, duality_gap, grad_min, step_calc, tol=tol, max_iters=max_iters)
		res[i,:] = x
	print('')
	print(np.round(res, 3))
	
	idx = np.nonzero(res[0,:]>1.e-8)[0]
	print('\nSUBSET:\n')
	print(G[np.ix_(idx, idx)])
	print(f'\nMATCH SIZE: {idx.shape[0]}\n')
	
	plt.plot(np.arange(1, steps + 1), np.mean(iters, axis = 0), '-o')
	plt.xlabel('step')
	plt.ylabel('avg iters')
	plt.show()

if __name__ == '__main__':
	main()