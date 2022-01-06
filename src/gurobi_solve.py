import numpy as np
import gurobipy as gp
from gurobipy import GRB
from permutations import *

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
	n = 10
	p = 0.7
	# if True:
		# A = np.array([
			# [0, 1, 1, 0, 0, 0],
			# [1, 0, 1, 0, 0, 0],
			# [1, 1, 0, 1, 1, 0],
			# [0, 0, 1, 0, 0, 1],
			# [0, 0, 1, 0, 0, 1],
			# [0, 0, 0, 1, 1, 0]
		# ])
		
		# B = np.array([
			# [0, 1, 0, 0, 0, 0],
			# [1, 0, 1, 1, 0, 0],
			# [0, 1, 0, 0, 0, 1],
			# [0, 1, 0, 0, 1, 1],
			# [0, 0, 0, 1, 0, 0],
			# [0, 0, 1, 1, 0, 0]
		# ])
	# else:
		# A = np.array([
			# [0, 1, 0, 0],
			# [1, 0, 1, 1],
			# [0, 1, 0, 1],
			# [0, 1, 1, 0]
		# ])
		
		# B = A.copy()
		# B[0, 1] = 0
		# B[1, 0] = 0
	
	# if True:
		# n = 15
		# p = 0.7
		# A = generate_random_graph(n, p)
		# pp = 0.3
		# B = perturbe_edges(A, pp)
		# P = random_permutation(n, n)
		# B = P.T @ A @ P 
	
	# P = random_permutation(A.shape[0], A.shape[0])
	# B = P.T @ A @ P 
	# print(A, '\n\n')
	# print(B)
	A = generate_random_graph(n, p)
	P = random_permutation(n, n)
	B = P.T @ A @ P 
	# print(A)
	# print(B)
	
	AG = np.kron(A, B) + np.kron(1 - A - np.identity(A.shape[0]), 1 - B - np.identity(B.shape[0]))
	n = AG.shape[0]
	AGC = 1 - AG - np.identity(n)
	
	m = gp.Model()
	# x = m.addMVar(n, vtype = GRB.CONTINUOUS)
	x = m.addMVar(n, vtype = GRB.BINARY)
	# lo = m.addConstr(x >= 0, 'lo')
	# hi = m.addConstr(x <= 1, 'hi')
	m.setObjective(x @ (AGC - np.identity(n)) @ x, GRB.MINIMIZE)
	m.params.NonConvex = 2
	m.optimize()
	
	X = np.array(m.getAttr('x'))
	# print(X)
	# exit()
	# X[X<0.5] = 0
	# X[X>=0.5] = 1
	# X = X.astype(np.int)
	n1, n2 = A.shape[0], B.shape[0]
	print('MATCH:')
	for i, xx in enumerate(X):
		if xx == 1:
			print(' ', 1 + i // n2, '->', 1 + i % n2, '\t', i)