import numpy as np
import scipy.sparse as sps
from itertools import permutations
import matplotlib.pyplot as plt

def generate_permutation_matrix_(perm):
	n = len(perm)
	return sps.coo_matrix((np.ones(n), (np.arange(n), perm)), shape = (n, n))

def generate_permutation_matrix(perm):
	n = len(perm)
	P = np.zeros(n ** 2, dtype = np.float).reshape((n, n))
	P[np.arange(n),perm] = 1.
	return P 

def find_global_best(f, A, B, mode):
	n = A.shape[0]
	F0P = np.empty(np.math.factorial(n))
	best = np.Inf if mode == 'min' else -np.Inf
	iter = 0
	niter = np.math.factorial(n)
	besti = []
	bestp = []
	
	for perm in permutations(np.arange(n)):
		P = generate_permutation_matrix(perm)
		f0p = f(A, B, P)
		F0P[iter] = f0p
		is_best = f0p < best if mode == 'min' else f0p > best
		if is_best:
			best = f0p 
			best_perm = perm 
			besti.append(iter)
			bestp.append(perm)
		elif f0p == best:
			besti.append(iter)
			bestp.append(perm)
		iter += 1
		if iter % 1000 == 0:
			print('\r{:.3f}%'.format(iter/niter*100.), end = '')
	print('\r{:.3f}%'.format(iter/niter*100.), end = '')
	print('')
	nbest = 0
	for f0p in F0P:
		if f0p == best:
			nbest += 1
	print('How many best?', nbest)
	i = len(besti)
	while i > 0:
		if F0P[besti[i-1]] == best:
			i -= 1
		else:
			break 
	bestp = bestp[i:]
	print(len(bestp))
	for bperm in bestp:
		print(np.array(bperm) + 1)
	return np.array(best_perm), F0P 

def generate_random_matrix(n, p = 0.7, permute = True):
	A1 = (np.random.rand(n ** 2).reshape((n, n)) > p).astype(np.float)
	A1 = A1 + A1.T 
	A1[A1>0] = 1.
	for i in range(n):
		A1[i,i] = 0.
	if permute:
		perm = np.random.permutation(n)
		P = generate_permutation_matrix(perm)
		A2 = P @ A1 @ P.T
	else:
		A2 = A1 
	return A1, A2 

def check_clique(A, C):
	for id, i in enumerate(C):
		for j in C[id+1:]:
			if A[i,j] != 1.:
				return False 
	return True 

def extract_clique(x):
	# return np.where(np.logical_not(np.isclose(x, 0.)))[0]
	return np.where(x == np.max(x))[0]
	
def replicator_dynamics(A, x, tol = 1.e-15, max_iter = 1000):
	dist = np.Inf
	iter = 0
	while dist > tol and iter < max_iter:
		old_x = x 
		Ax = A.dot(x)
		x = (x * Ax) / (x @ Ax)
		dist = np.linalg.norm(x - old_x)
		# if dist <= tol and not check_clique(A, extract_clique(x)):
			# x = x + tol * np.random.rand(x.shape[0])
			# x[x < 0] = 0.
			# x /= np.sum(x)
		iter += 1
	print('Is clique?', check_clique(A, extract_clique(x)))
	print(x)
	return x 

def extract_match(x, n):
	clique = extract_clique(x)
	match = []
	for m in clique:
		u1, u2 = m // n, m % n
		match.append((u1, u2))
	return match

# A1 = np.array([
	# [0, 1, 0, 0],
	# [1, 0, 1, 1],
	# [0, 1, 0, 1],
	# [0, 1, 1, 0]
# ])

A1 = np.array([
	[0, 1, 1, 0, 0, 0],
	[1, 0, 1, 0, 0, 0],
	[1, 1, 0, 1, 1, 0],
	[0, 0, 1, 0, 0, 1],
	[0, 0, 1, 0, 0, 1],
	[0, 0, 0, 1, 1, 0]
])
# A2 = np.array([
	# [0, 1, 0, 0, 0, 0],
	# [1, 0, 1, 1, 0, 0],
	# [0, 1, 0, 0, 0, 1],
	# [0, 1, 0, 0, 1, 1],
	# [0, 0, 0, 1, 0, 0],
	# [0, 0, 1, 1, 0, 0]
# ])
A2 = A1
# A1 = np.array([
	# [0, 1, 1, 0, 0, 0],
	# [1, 0, 1, 0, 0, 0],
	# [1, 1, 0, 1, 0, 0],
	# [0, 0, 1, 0, 1, 1],
	# [0, 0, 0, 1, 0, 0],
	# [0, 0, 0, 1, 0, 0]
# ])

# A2 = np.array([
	# [0, 1, 1, 0, 0, 0],
	# [1, 0, 1, 0, 0, 0],
	# [1, 1, 0, 0, 0, 0],
	# [0, 0, 0, 0, 1, 1],
	# [0, 0, 0, 1, 0, 0],
	# [0, 0, 0, 1, 0, 0]
# ])

# A1 = np.array([
	# [0, 1, 0, 1],
	# [1, 0, 1, 0],
	# [0, 1, 0, 1],
	# [1, 0, 1, 0]
# ])

# A1 = np.array([
	# [0, 1, 1],
	# [1, 0, 1],
	# [1, 1, 0]
# ])

# A2 = A1

def FAQ(A, B, P):
	return -np.trace(A @ P @ B.T @ P.T)

def f_(A, B, P):
	M = A.dot(P) - P.dot(B)
	if sps.issparse(M):
		return sps.linalg.norm(M) ** 2
	else:
		return np.linalg.norm(M) ** 2

def f(A, B, P):
	return np.linalg.norm(A @ P - P @ B)

if __name__ == '__main__':
	np.random.seed(1)
	# A1, A2 = generate_random_matrix(10, 0.5)
	# A1 = np.array([
		# [0, 1, 1, 0, 0],
		# [1, 0, 1, 0, 0],
		# [1, 1, 0, 0, 0],
		# [0, 0, 0, 0, 0],
		# [0, 0, 0, 0, 0]
	# ])
	# A2 = A1 
	# A2[0,3] = 1
	# A2[1,4] = 1
	A1 = np.array([
		0, 1, 1, 0, 0, 0, 0, 0,
		1, 0, 1, 0, 0, 0, 0, 0,
		1, 1, 0, 1, 1, 0, 0, 0,
		0, 0, 1, 0, 0, 1, 0, 0,
		0, 0, 1, 0, 0, 1, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0
	]).reshape((8,8))
	A2 = np.array([
		0, 1, 1, 1, 0, 0, 1, 1,
		1, 0, 1, 0, 0, 0, 1, 1,
		1, 1, 0, 1, 1, 0, 0, 0,
		1, 0, 1, 0, 0, 1, 0, 0,
		0, 0, 1, 0, 0, 1, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		1, 1, 0, 0, 0, 0, 0, 0,
		1, 1, 0, 0, 0, 0, 0, 0
	]).reshape((8,8))
	
	A1 = np.array([
		0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0,
		1, 0, 0, 1, 1, 0, 1, 0,
		0, 1, 1, 0, 0, 1, 0, 1,
		0, 0, 1, 0, 0, 0, 1, 0, 
		0, 0, 0, 1, 0, 0, 0, 1,
		0, 0, 1, 0, 1, 0, 0, 1,
		0, 0, 0, 1, 0, 1, 1, 1,
	]).reshape((8,8))
	
	A1 = np.array([
		[0, 1, 0, 0],
		[1, 0, 1, 1],
		[0, 1, 0, 1],
		[0, 1, 1, 0]
	])
	print(A1)
	n = A1.shape[0]
	P = generate_permutation_matrix(np.random.permutation(n))
	A2 = P @ A1 @ P.T
	print(A2)
	
	n = A1.shape[0]
	Ac = 1. - A1 - np.identity(n)
	Bc = 1. - A2 - np.identity(n)
	
	def ft(A, B, P):
		# d1 = np.sum(np.triu(np.kron(A, B))) 
		# d2 = np.sum(np.triu(np.kron(Ac, Bc))) 
		# return np.trace((d1/(d1+d2)) * A @ P @ B @ P.T + (d2/(d1+d2)) * Ac @ P @ Bc @ P.T)
		return -np.trace(A @ P @ B @ P.T + Ac @ P @ Bc @ P.T)
		
	perm, F0P = find_global_best(ft, A1, A2, 'min')
	A = np.kron(A1, A2)
	A = A + np.identity(A.shape[0]) / 2
	ssd = replicator_dynamics(A, np.ones(A.shape[0]) / A.shape[0])
	match = extract_match(ssd, A2.shape[0])
	print(ssd[match])
	for m in match:
		print(m[0] + 1, '-->', m[1] + 1)
	# print(match)
	#print(perm + 1)
	plt.plot(np.arange(len(F0P)), F0P, 'o')
	plt.show()