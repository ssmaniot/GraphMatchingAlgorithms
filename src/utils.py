import numpy as np
import scipy.sparse as sp 
from scipy.sparse.linalg import eigsh
from scipy.optimize import linprog, line_search
from typing import Tuple, Callable
from time import process_time

def timeit(f, *args):
	start = process_time()
	ret = f(*args)
	end = process_time()
	print(f'time elapsed: {end - start}')
	return ret 


def generate_random_graph(n, p):
	A = np.zeros(n ** 2, dtype = np.int).reshape((n, n))
	A[np.triu_indices(n, 1)] = np.random.rand(n * (n - 1) // 2) <= p
	return (A + A.T)


def perturbe_edges(A, p):
	n = A.shape[0]
	B = A.copy()
	mask = np.triu(np.random.rand(n ** 2).reshape((n, n)) < p)
	B[mask] = 1 - B[mask]
	B = np.triu(B, 1) + np.triu(B, 1).T 
	return B.astype(np.int)


def random_permutation(n1, n2):
	n = min(n1, n2)
	s = np.random.choice(np.arange(n1), size = n, replace = False)
	t = np.random.choice(np.arange(n2), size = n, replace = False)
	P = np.zeros(n1 * n2).reshape((n1, n2))
	for i in range(n):
		P[s[i], t[i]] = 1
	# P[np.ix_(s, t)] = 1
	return P.astype(np.int)

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
	