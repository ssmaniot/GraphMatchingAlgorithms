from permutations import *
from time import time
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
	A = np.array([
		[0, 1, 0, 0],
		[1, 0, 1, 1],
		[0, 1, 0, 1],
		[0, 1, 1, 0]
	])
	
	B = A.copy()
	# B[0, 1] = 0
	# B[1, 0] = 0
	G = np.kron(A, B)
	G = G - np.identity(G.shape[0])
	
	def f(A, B, P):
		return -np.trace(A @ P @ B.T @ P.T) 
	
	def ff(A, B, P):
		X = np.ndarray.flatten(P)
		return X.T @ G @ X
		
	start = time()
	bestp, iters = find_global_best(f, A, B, mode = 'binary')
	print('Elapsed time:', time() - start)
	print(len(bestp), np.math.factorial(16))
	exit()
	for i, perm in enumerate(bestp):
		# print(perm)
		if True:
			print(perm)
			a, b = perm[0], perm[1]
			assert len(a) == len(b)
			print('Permutation {}:'.format(i + 1))
			for j in range(len(a)):
				print(a[j] + 1, '->', b[j] + 1)