import numpy as np
import matplotlib.pyplot as plt 

def replicator_dynamics(A, x, tol = 1.e-15, max_iter = 1000):
	dist = np.Inf
	iter = 0
	while dist > tol and iter < max_iter:
		old_x = x 
		Ax = A.dot(x)
		x = (x * Ax) / (x @ Ax)
		dist = np.linalg.norm(x - old_x)
		iter += 1
	return x 

def check_clique(A, C):
	for id, i in enumerate(C):
		for j in C[id+1:]:
			if A[i,j] != 1.:
				return False 
	return True 

if __name__ == '__main__':
	
	n = 10
	csize = np.zeros(n+1).astype(np.int)
	for _ in range(1000):
		
		if False:
			A = np.random.rand(n ** 2).reshape((n, n))
			A[A > 0.7] = 1.
			A[A <= 0.7] = 0.
			A = A + A.T 
			A[A>0] = 1.
			for i in range(n):
				A[i,i] = 0.
			perm = np.random.permutation(n)
			P = np.zeros(n ** 2).reshape((n, n))
			for i, p in enumerate(perm):
				P[i, p] = 1.
			B = P.T @ A @ P 
			A = np.kron(A, B)
		
		else:
			A1 = np.array([
				[0., 1., 1., 0., 0., 0.], 
				[1., 0., 1., 0., 0., 0.], 
				[1., 1., 0., 1., 1., 0.], 
				[0., 0., 1., 0., 0., 1.], 
				[0., 0., 1., 0., 0., 1.], 
				[0., 0., 0., 1., 1., 0.]
			])
			A2 = np.array([
				[0., 1., 0., 0., 0., 0.], 
				[1., 0., 1., 1., 0., 0.], 
				[0., 1., 0., 0., 0., 1.], 
				[0., 1., 0., 0., 1., 1.], 
				[0., 0., 0., 1., 0., 0.], 
				[0., 0., 1., 1., 0., 0.]
			])
			if False:
				A1 = np.array([
					[0., 1., 0., 0.],
					[1., 0., 1., 1.],
					[0., 1., 0., 1.],
					[0., 1., 1., 0.]
				])
				A2 = np.array([
					[0., 0., 0., 0.],
					[0., 0., 1., 1.],
					[0., 1., 0., 1.],
					[0., 1., 1., 0.]
				])
			
			A = np.kron(A1, A2) + np.identity(36) / 2
		
		# x = np.zeros(16)
		# x[1*4 + 1] = 1/3
		# x[2*4 + 2] = 1/3
		# x[3*4 + 3] = 1/3
		x = np.zeros(36)
		x[2*6+3] = 1/4 # 13
		x[4*6+5] = 1/4 # 20
		x[5*6+2] = 1/4 # 27
		x[2*6+1] = 1/4 # 35
		match = np.where(np.logical_not(np.isclose(x, 0.)))[0]
		if not check_clique(A, match):
			print('not a clique!')
		print(A[np.ix_(match, match)])
		print(match)
		exit()
		# x += 1.e-12
		x /= np.sum(x)
		# x = np.random.rand(A.shape[0])
		# x /= np.sum(x)
		ssd = replicator_dynamics(A + np.identity(A.shape[0]) / 2, x)
		match = np.where(np.logical_not(np.isclose(ssd, 0.)))[0]
		if not check_clique(A, match):
			print('not a clique!')
		print(A[np.ix_(match, match)])
		exit()
		csize[len(match)] += 1
	
	plt.bar(np.arange(n + 1), csize)
	plt.show()

	if False:
		n = 20
		A = np.random.rand(n ** 2).reshape((n, n))
		A[A > 0.7] = 1.
		A[A <= 0.7] = 0.
		A = A + A.T 
		A[A>0] = 1.
		for i in range(n):
			A[i,i] = 0.
		print(A)
		x = np.random.rand(n)
		x /= np.sum(x)
		ssd = replicator_dynamics(A + np.identity(n) / 2, x)
		print(ssd)
		C = np.where(np.logical_not(np.isclose(ssd, 0.)))[0]
		print(C)
		print('Clique?', check_clique(A, C))
		print(A[np.ix_(C, C)])