import numpy as np 
from scipy.sparse import csr_matrix 
from frank_wolfe import frank_wolfe 
from utils import *

def solve(GC, x0, steps, tol = 1.e-15, max_iters = 2000):
	lmin, lmax = sigma_minmax(GC - sp.eye(GC.shape[0]))
	MU = np.linspace(lmin + 0.1, -(lmax + 0.1), num = steps)
	x = x0 
	step_calc = STEP_CALC()
	duality_gap = DUALITY_GAP()
	for i, mu in enumerate(MU, 1):
		print(f'\rStep {i}/{MU.shape[0]}', end='')
		_, grad = FUN_GRAD(GC, mu)
		grad_min = GRAD_MIN(GC.shape[0])
		x, _ = frank_wolfe(x, grad, duality_gap, grad_min, step_calc, tol=tol, max_iters=max_iters)
	print(f'\r                         ', end='\r')
	idx = np.argwhere(x > 1. - 1.e-5)
	return x 

def generate_instance(n, p):
		G1 = generate_random_graph(n, p)
		P = random_permutation(n, n)
		G2 = P.T @ G1 @ P 
		return G1, G2 

def run_experiment(n, p, num_exp):
	print('-----')
	print(f'EXP n: {n}, p: {p}')
	print('-----')
	match = np.empty(num_exp)
	for i in range(num_exp):
		print(f'Exp. {i + 1}/{num_exp}')
		G1, G2 = generate_instance(n, p)
		G = np.kron(G1, G2) + np.kron(1 - G1 - np.identity(n), 1 - G2 - np.identity(n))
		GC = csr_matrix(1 - G - np.identity(G.shape[0])) 
		x = timeit(solve, GC, np.random.rand(GC.shape[0]), 10)
		match[i] = np.argwhere(x > 1. - 1.e-5).shape[0] / n
	print(f'match: {match}')
	return match 

def main() -> None:
	np.random.seed(1)
	size = np.array([20]) # , 20, 30, 40, 50, 60])
	probs = np.array([0.7]) # np.array([0.5, 0.6, 0.7, 0.8, 0.9])
	num_exp = 5
	matches = np.empty(size.shape[0] * probs.shape[0] * num_exp).reshape((size.shape[0], probs.shape[0], num_exp)).astype('float')
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

if __name__ == '__main__':
	main()