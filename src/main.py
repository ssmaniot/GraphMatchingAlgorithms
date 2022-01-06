from replicator_dynamics import *
from permutations import *
import numpy as np

def generate_random_graph(n, p):
	A = np.random.rand(n ** 2).reshape((n, n))
	A[A > p] = 0.
	A = np.triu(A, 1) + np.triu(A, 1).T 
	A[A>0] = 1
	return A.astype(np.int)

if False and __name__ == '__main__':
	n, p = 64, 1/8
	A = generate_random_graph(n, p)
	print(A)
	C = simple_max_clique(A)
	print(A[np.ix_(C, C)])
	print(C)

if False and __name__ == '__main__':
	A1 = np.array([
		[0, 1, 1, 0, 0, 0],
		[1, 0, 1, 0, 0, 0],
		[1, 1, 0, 1, 1, 0],
		[0, 0, 1, 0, 0, 1],
		[0, 0, 1, 0, 0, 1],
		[0, 0, 0, 1, 0, 1]
	])
	C = simple_max_clique(A1)
	print(C)

if False and __name__ == '__main__':
	A1 = np.array([
		[0, 1, 1, 0, 0, 0],
		[1, 0, 1, 0, 0, 0],
		[1, 1, 0, 1, 1, 0],
		[0, 0, 1, 0, 0, 1],
		[0, 0, 1, 0, 0, 1],
		[0, 0, 0, 1, 0, 1]
	])
	
	A2 = np.array([
		[0, 0, 1, 0, 0, 0],
		[0, 0, 0, 0, 1, 0],
		[1, 0, 0, 1, 1, 0],
		[0, 0, 1, 0, 0, 1],
		[0, 1, 1, 0, 0, 1],
		[0, 0, 0, 1, 1, 0]
	])
	
	def f(A, B, P):
		return -np.trace(A @ P @ B.T @ P.T)
	
	perm, F0P = find_global_best(f, A1, A2)
	print(perm + 1)

if True and __name__ == '__main__':
	A1 = np.array([
		[0, 1, 0, 0],
		[1, 0, 1, 1],
		[0, 1, 0, 1],
		[0, 1, 1, 0]
	])
	
	A2 = A1.copy()
	A2[0,1] = 0
	A2[1,0] = 0
	
	n = A1.shape[0]
	A = np.kron(A1, A2)
	AI = A + 0.5 * np.identity(A.shape[0])
	n = AI.shape[0]
	x = np.ones(n) / n
	ess = replicator_dynamics(AI, x)
	print(ess)
	print(get_vertices(ess))
	n1, n2 = A1.shape[0], A2.shape[0]
	
	match = extract_match(ess, n1, n2)
	print(match)