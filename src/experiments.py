import numpy as np
from permutations import *
from frank_wolfe import *
import scipy.sparse as sp 
from scipy.sparse.linalg import eigsh
from scipy.optimize import linprog, line_search
import matplotlib.pyplot as plt 
import itertools 

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

if __name__ == '__main__':
	np.random.seed(1)
	n = 30
	p = 0.7
	steps = int(3 * n)
	for i in range(10):
		G1 = generate_random_graph(n, p)
		P = random_permutation(n, n)
		G2 = P.T @ G1 @ P 
		# print(G1)
		# print(G2)
		G = np.kron(G1, G2) + np.kron(1 - G1 - np.identity(n), 1 - G2 - np.identity(n))
		w, _ = np.linalg.eig(G)
		N = G.shape[0]
		GC = 1 - G - np.identity(N)
		# print(GC - np.identity(N))
		# exit()
		lmin, lmax = kron_eig_bound(G1, G2)
		w, _ = np.linalg.eig(GC)
		print(f'min eval: {np.min(w)}')
		print(f'max eval: {np.max(w)}')
		# exit()
		# print(f'lmin = {abs_min(w)}, est = {lmin}')
		# print(f'lmax = {abs_max(w)}, est = {lmax}')
		# W, _ = np.linalg.eig(GC - np.identity(N))
		# print(f'lminc = {abs_min(W)}, lmaxc = {abs_max(W)}')
		k = 0
		# x = np.ones(N) / 2 + 1.e-3 * (np.random.rand(N) - 0.5)
		# x /= np.sum(x)
		x = np.random.rand(N) 
		print('Start!')
		# if True:
		w, _ = np.linalg.eig(GC - np.identity(N))
		lmin = np.abs(np.min(w))
		lmax = np.abs(np.max(w))
		# for mu in np.linspace(lmin, lmax, num = steps):
		for mu in np.linspace(lmin, -lmax, num = steps):
			# mu = 0.
			
			if (k % 1 == 0):
				# print('')
				# print('----------------------')
				print('\rStep: {}/{}'.format(k, steps), end = '')
				# print('----------------------')
				# print('')
			def f(x):
				return x @ (GC + (mu - 1) * np.identity(N)) @ x - mu * np.sum(x)
			
			def df(x):
				return (GC + (mu - 1) * np.identity(N)) @ x - mu * np.ones(N)
			
			def proj(y):
				return np.clip(y, 0., 1.)
				
			def grad_min_(x: np.ndarray, square_lim=5):
				# print('x:', x)
				grad = -df(x)
				# print('Grad:', grad)
				grad[grad<0] = 0.
				grad[grad>1] = 1.
				# print('s:', grad)
				return grad 
			
			def grad_min(x: np.ndarray, square_lim=5):
				# print('x:', x)
				s = None
				with gp.Env(empty=True) as env:
					env.setParam("OutputFlag", 0)
					env.setParam("LogToConsole", 0)
					env.start()
					grad = df(x)
					# print('Grad:', grad)
					with gp.Model(env=env) as m:
						s = m.addMVar(shape=x.shape, name="s")
						m.addConstr(s >= 0)
						m.addConstr(s <= 1)
						m.setObjective(grad @ s, GRB.MINIMIZE)
						m.optimize()
						X = s.X
						# print('s:', X)
						return s.X
			
			def step_calc_(x, s, k):
				return 2 / (2 + k)
			
			def step_calc(x: np.ndarray, s: np.ndarray, k):
				res = line_search(f, df, x, s - x, amax = 1.)
				alpha = res[0]
				if alpha is None:
					alpha = 1.
				return alpha
			
			def dist(x1, x2):
				return np.linalg.norm(x1 - x2)
			
			x = frank_wolfe(x, dist, grad_min, step_calc, dist_tolerance=1e-20, tolerance = 1.e-15)
			# x = cs.ProjectedGD(f, df, proj, ss.ScaledInvIterStepSize()).solve(x0 = x, max_iter = 10000, tol = 1.e-15, disp = 1)
			# x = cs.ProjectedGD(f, df, proj, ss.ScaledConstantStepSize(1 / np.log(x.shape[0]))).solve(x0 = x, max_iter = 1000, tol = 1.e-15, disp = 1)
			# print(x)
			# exit()
			k += 1
		# print(x)
		# idx = np.argwhere(x > 0).flatten()
		# print(idx)
		print('\rStep: {}/{}'.format(k, steps))
		# print(G[np.ix_(idx,idx)])
		idx = np.argwhere(x > 0).flatten()
		print(f'Number of matched vertices: {len(idx)}/{n}')
		# print(x)
		print(idx)
		# print(G[np.ix_(idx,idx)])
		exit()