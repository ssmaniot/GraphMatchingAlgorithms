from permutations import *
from time import time
from random import sample
import os 
import numpy as np

def generate_random_graph(n, p):
	A = np.random.rand(n ** 2).reshape((n, n))
	A[A > p] = 0.
	A = np.triu(A, 1) + np.triu(A, 1).T 
	A[A>0] = 1
	return A.astype(np.int)

def perturbe_edges(A, p):
	n = A.shape[0]
	B = A.copy()
	mask = np.triu(np.random.rand(n ** 2).reshape((n, n)) < p)
	B[mask] = 1 - B[mask]
	B = np.triu(B, 1) + np.triu(B, 1).T 
	return B.astype(np.int)

if __name__ == '__main__':
	reps = 100
	p = 0.8
	pp = 0.4
	vmin, vmax = 4, 6
	
	nvert = np.arange(vmin, vmax + 1).astype(np.int)
	perm_error = np.empty((vmax - vmin + 1) * reps).reshape((vmax - vmin + 1, reps))
	
	def f(A, B, P):
		return -np.trace(A @ P @ B.T @ P.T) 
	
	for n in range(vmin, vmax + 1):
		size = n * (n - 1) / 2
		vertices = np.arange(n)
		for k in range(reps):
			if k % 10:
				print('Batch {} - {:.3f}%'.format(n, k / reps * 100.), end = '\r')
			A = generate_random_graph(n, p)
			B = perturbe_edges(A, pp)
			
			bestp, _ = find_global_best(f, A, B, mode = 'full')
			max_error = np.Inf
			for perm in bestp:
				mismatches = 0
				for i in range(n):
					for j in range(i+1, n):
						if A[i,j] != B[perm[i],perm[j]]:
							mismatches += 1
				error = mismatches / size
				if error < max_error:
					max_error = error
			perm_error[n-vmin,k] = max_error
		print('Batch {} - done.  '.format(n))
	np.savez('results_bruteforce_perm.npz', nvert = nvert, perm_error = perm_error)
	
	if False:
		for n in range(6, 9):
			folder = os.path.join('data', 'graph{}c'.format(n))
			files = [fname for fname in os.listdir(folder)]
			print(len(files))
			chosen_files = sample(files, size)
			for file in chosen_files:
				A = np.load(file)
				print(A)
				exit()