import numpy as np
import scipy.sparse as sp 
from scipy.sparse.linalg import eigsh
from scipy.optimize import linprog, line_search
import matplotlib.pyplot as plt 
import itertools 
from time import process_time
from permutations import *
from frank_wolfe import frank_wolfe
from typing import Tuple, Callable
from utils import *


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


def solve(GC, x0, steps, stol = 1.e-15):
	k = 0
	x = x0
	N = GC.shape[0]
	print('Start!')
	lmin, lmax = sigma_minmax(GC - sp.eye(N))
	print(f'min eval: {lmin}')
	print(f'max eval: {lmax}')
	MU = np.linspace(lmin + .1, -lmax - .1, num = steps)
	if (k % 1 == 0):
		print('\rStep: {}/{}'.format(k, steps), end = '')
	mu = MU[-1]
	
	def dist(x1, x2):
		return np.linalg.norm(x1 - x2)
		
	def df(x):
		return GC @ x + (mu - 1.) * x - mu 
	
	def grad_min(x: np.ndarray, square_lim = 5):
		grad = df(x)
		res = linprog(grad, bounds = [(0., 1.) for _ in range(x.shape[0])])
		return res.x 
		
	def step_calc(x, s, k):
		return 2 / (2 + k)
	
	def f(x):
			return x @ (GC + (mu - 1) * eye(N)) @ x - mu * np.sum(x)
	
	x0 = x 
	x = frank_wolfe(x, dist, grad_min, step_calc, dist_tolerance=1e-20, tolerance = 1.e-15)
	print(f'x: {x}')
	print(f'f(x) = {f(x)}')
	k += 1
	
	if False:
		for mu in MU[1:]:
			if (k % 1 == 0):
				print('\rStep: {}/{}'.format(k, steps), end = '')
				
			def f(x):
				return x @ (GC + (mu - 1) * eye(N)) @ x - mu * np.sum(x)
			
			if np.abs(f(x0) - f(x)) < stol:
				k += 1
				continue 
			
			def df(x):
				# return (GC + (mu - 1) * np.identity(N)) @ x - mu # * np.ones(N)
				return GC @ x + (mu - 1.) * x - mu 
			
			def grad_min(x: np.ndarray, square_lim = 5):
				grad = df(x)
				res = linprog(grad, bounds = [(0., 1.) for _ in range(x.shape[0])])
				return res.x 
			
			def grad_min__(x: np.ndarray, square_lim=5):
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
			
			def step_calc(x, s, k):
				return 2 / (2 + k)
			
			def step_calc_(x: np.ndarray, s: np.ndarray):
				res = line_search(f, df, x, s, maxiter=100)
				alpha, fc, gc, new_fval, old_fval, new_slope = res
				if alpha is None:
					return 0.
				if alpha > 1:
					alpha = 1.
				# print('alpha:', alpha)
				return alpha
			
			def dist(x1, x2):
				return np.linalg.norm(x1 - x2)
			
			x0 = x 
			x = frank_wolfe(x, dist, grad_min, step_calc, dist_tolerance=1e-20, tolerance = 1.e-15)
			k += 1
		print('\rStep: {}/{}'.format(k, steps))
	idx = np.argwhere(x > 0).flatten()
	print(f'Number of matched vertices: {len(idx)}/{x.shape[0]}')
	print(idx)
	return idx 

def generate_instance(n, p):
		G1 = generate_random_graph(n, p)
		P = random_permutation(n, n)
		# G1 = np.zeros(n * n).reshape((n, n))
		# rng = np.arange(n-1)
		# G1[rng,rng+1] = 1
		# G1[rng+1,rng] = 1
		G2 = P.T @ G1 @ P 
		return G1, G2 

def run_experiment(n, p, num_exp, stol = 0.01):
	print('-----')
	print(f'EXP n: {n}, p: {p}')
	print('-----')
	match = np.empty(num_exp)
	for i in range(num_exp):
		print(f'Exp. {i + 1}/{num_exp}')
		G1, G2 = generate_instance(n, p)
		G = np.kron(G1, G2) + np.kron(1 - G1 - np.identity(n), 1 - G2 - np.identity(n))
		w, _ = np.linalg.eig(G)
		N = G.shape[0]
		GC = csr_matrix(1 - G - np.identity(N)) 
		# x = solve(GC, np.random.rand(N), int((1. + np.log10(n)) * n))
		# x = solve(GC, np.ones(N) / 2, int((1. + np.log10(n)) * 100), stol = tol)
		# x = timeit(solve, GC, np.ones(N) / 2, int((1. + np.log10(n)) * 100), tol)
		x = timeit(solve, GC, np.ones(N) / 2, 10, stol)
		match[i] = len(x) / n 
	print(f'match: {match}')
	# exit()
	return match 


if __name__ == '__main__':
	np.random.seed(1)
	size = np.array([10]) # , 20, 30, 40, 50, 60])
	probs = np.array([0.9]) # np.array([0.5, 0.6, 0.7, 0.8, 0.9])
	num_exp = 5
	matches = np.empty(size.shape[0] * probs.shape[0] * num_exp).reshape((size.shape[0], probs.shape[0], num_exp)).astype(np.float)
	for i, n in enumerate(size):
		for j, p in enumerate(probs):
			matches[i,j,:] = run_experiment(n, p, num_exp)
	print('----')
	print('match %:')
	print(matches)
	# print(match)
	print(f'sizes: {size}')
	for j, p in enumerate(probs):
		print(f'avg. p: {p} {np.mean(matches[:,j,:], axis = 1)}')
	np.save('results10.npy', matches)