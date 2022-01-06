import numpy as np
import matplotlib.pyplot as plt 

def extract_clique(x):
	return np.where(np.logical_not(np.isclose(x, 0.)))[0]

def check_clique(A, C):
	for id, i in enumerate(C):
		for j in C[id+1:]:
			if A[i,j] != 1.:
				return False 
	return True 
	
def replicator_dynamics(A, x, tol = 1.e-15, max_iter = 1000):
	dist = np.Inf
	iter = 0
	prev_match = None
	while dist > tol and iter < max_iter:
		old_x = x 
		Ax = A.dot(x)
		x = (x * Ax) / (x @ Ax)
		dist = np.linalg.norm(x - old_x)
		if dist <= tol:
			match = extract_clique(x)
			if not check_clique(A, match):
				x = x + 1.e-9 * np.random.normal(0., 1., x.shape[0])
				x[x<0] = 1.e-9
				x /= np.sum(x)
		iter += 1
	print(x)
	return x 

def print_matrix(A):
	print('   ', end = '')
	for i in range(A.shape[0]):
		print('{:2d} '.format(i), end = '')
	print('')
	for i, l in enumerate(A.astype(np.int).tolist()):
		print('{:2d} {}'.format(i, l))

# Isomorphism 1
A1 = np.array([
	[0., 1., 1., 0., 0., 0.], 
	[1., 0., 1., 0., 0., 0.], 
	[1., 1., 0., 1., 1., 0.], 
	[0., 0., 1., 0., 0., 1.], 
	[0., 0., 1., 0., 0., 1.], 
	[0., 0., 0., 1., 1., 0.]
])
A2 = np.array([
	[0, 1, 1, 0, 0, 0],
	[1, 0, 1, 0, 0, 0],
	[1, 1, 0, 0, 0, 0],
	[0, 0, 0, 0, 1, 1],
	[0, 0, 0, 1, 0, 1],
	[0, 0, 0, 1, 1, 0]
])

# Isomorphism 2
A1 = np.array([
	[0, 1, 1, 0, 0, 0],
	[1, 0, 1, 0, 0, 0],
	[1, 1, 0, 1, 0, 0],
	[0, 0, 1, 0, 1, 1],
	[0, 0, 0, 1, 0, 1],
	[0, 0, 0, 1, 1, 0]
])
A2 = np.array([
	[0, 1, 1, 0, 0, 0, 1],
	[1, 0, 1, 0, 0, 0, 0],
	[1, 1, 0, 1, 0, 0, 0],
	[0, 0, 1, 0, 1, 1, 1],
	[0, 0, 0, 1, 0, 1, 0],
	[0, 0, 0, 1, 1, 0, 0],
	[1, 0, 0, 1, 0, 0, 0]
])

# Isomorphism 3
A1 = np.array([
	[0, 1, 0, 0],
	[1, 0, 1, 1],
	[0, 1, 0, 1],
	[0, 1, 1, 0]
])
A2 = np.array([
	[0, 0, 0, 0],
	[0, 0, 1, 1],
	[0, 1, 0, 1],
	[0, 1, 1, 0]
])
# B = np.array([
	# [0, 1, 1, 0, 0, 0],
	# [1, 0, 1, 0, 0, 0],
	# [1, 1, 0, 0, 0, 0],
	# [0, 0, 0, 0, 1, 1],
	# [0, 0, 0, 1, 0, 1],
	# [0, 0, 0, 1, 1, 0]
# ])

# A1 = A2
# n = A1.shape[0]
# print('[', end='')
# for r in A1.tolist():
	# print(r)
# print('];')
# exit()

if False:
	p = 0.7
	n = 100
	A1 = (np.random.rand(n ** 2).reshape((n, n)) > p).astype(np.float)
	A1 = A1 + A1.T 
	A1[A1>0] = 1.
	for i in range(n):
		A1[i,i] = 0.
	if True:
		perm = np.random.permutation(n)
		P = np.zeros(n ** 2).reshape((n, n))
		P[np.arange(n),perm] = 1.
		A2 = P @ A1 @ P.T

# A1 = np.array([
	# [0., 1., 0., 0.],
	# [1., 0., 1., 1.],
	# [0., 1., 0., 1.],
	# [0., 1., 1., 0.]
# ])
# A2 = np.array([
	# [0., 0., 0., 0.],
	# [0., 0., 1., 1.],
	# [0., 1., 0., 1.],
	# [0., 1., 1., 0.]
# ])

# A1 = np.array([
	# 0, 0, 1, 0, 0, 0, 0, 0,
	# 0, 0, 0, 1, 0, 0, 0, 0,
	# 1, 0, 0, 1, 1, 0, 1, 0,
	# 0, 1, 1, 0, 0, 1, 0, 1,
	# 0, 0, 1, 0, 0, 0, 1, 0, 
	# 0, 0, 0, 1, 0, 0, 0, 1,
	# 0, 0, 1, 0, 1, 0, 0, 1,
	# 0, 0, 0, 1, 0, 1, 1, 1,
# ]).reshape((8,8))
# n = A1.shape[0]
# perm = np.random.permutation(n)
# P = np.zeros(n ** 2).reshape((n, n))
# P[np.arange(n),perm] = 1.
# A2 = P @ A1 @ P.T
# A2 = A1

n = A1.shape[0]
A = np.kron(A1, A2) + np.kron(1 - A1 - np.identity(A1.shape[0]), 1 - A2 - np.identity(A2.shape[0]))
x = np.ones(A.shape[0]) # np.random.rand(A.shape[0])
x /= np.sum(x)
ssd = replicator_dynamics(A + np.identity(A.shape[0]) / 2, x)
print('1/(1-x\'Ax) = {:.3}'.format(1/(1-(ssd.T@A@ssd))))
match = extract_clique(ssd)
print('Is clique?', check_clique(A, match))
print('Vertex matched: {}/{}'.format(len(match), n))
print(match)
# print_matrix(A[np.ix_(match, match)])
X = np.zeros(n ** 2).reshape((n, n))
for m in match:
	u1, u2 = m // A2.shape[0], m % A2.shape[0]
	X[u1,u2] = 1
	print(u1+1, '->', u2+1)
# print(X)
# print('err:', np.linalg.norm(A1 - X @ A2 @ X.T))
# print(A1)
# print(X @ A2 @ X.T)