import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment, minimize_scalar

def replicator_dynamics(A, x, tol = 1.e-15):
	dist = np.Inf
	while dist > tol:
		old_x = x 
		Ax = A.dot(x)
		x = (x * Ax) / (x @ Ax)
		dist = np.linalg.norm(x - old_x)
	return x 

def f(P, A, B):
	return np.trace(A @ P @ B.T @ P.T)

def FAQ(A, B, max_iter = 100, tol = 1.e-12):
	n = A.shape[0]
	P = np.ones(n ** 2).reshape(n, n) / n**2
	dist = np.Inf
	
	iter = 0
	while dist > tol and iter < max_iter:
		P_old = P
		print(P)
		grad = (A @ P) @ (B.T + B)
		gradTfP = grad.T @ P
		row_ind, col_ind = linear_sum_assignment(lambda P: np.trace(((A @ P) @ (B.T + B)).T @ P))
		Q = np.zeros(n ** 2).reshape(n, n)
		Q[row_ind, col_ind] = 1
		alpha = minimize_scalar(lambda alpha: f(P + alpha * Q, A, B), bounds = (0., 1.), method = 'bounded').x
		P = P + alpha * Q
		dist = np.linalg.norm(P - P_old)
		iter += 1
		print('{:2d}. dist = {:.3e}'.format(iter, dist))
	row_ind, col_ind = linear_sum_assignment(P)
	P = np.zeros(n ** 2).reshape(n, n)
	P[row_ind, col_ind] = 1.
	return P

if __name__ == '__main__':
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
	
	# P = FAQ(A1, A2)
	# print(P)
	# print('err: ', np.linalg.norm(A1 - P.T @ A2 @ P))
	
	# exit()
	
	A = np.kron(A1, A2)
	# x /= np.sum(x)
	
	found = np.zeros(37).astype(np.int)
	
	for _ in range(1000):
		x = np.random.rand(A.shape[0])
		x /= np.sum(x)
		ssd = replicator_dynamics(A, x)
		match = np.where(np.logical_not(ssd < np.isclose(ssd, 0.)))[0]
		found[len(match)] += 1
	
	_ = plt.bar(np.arange(37), found)
	plt.show()
	
	if False:
		x = np.ones(A.shape[0]) / A.shape[0] #np.random.rand(A.shape[0])
		ssd = replicator_dynamics(A + np.identity(36) / 2, x)
		match = np.where(np.logical_not(np.isclose(ssd, 0.)))[0]
		print(match)
		
		idx = np.argsort(ssd)
		# for i in idx[::-1]:
			# print(i // A2.shape[0], '->', i % A2.shape[0], 'p = {:.3e}'.format(ssd[i]))
			
		V1 = match // A2.shape[0]
		V2 = match % A2.shape[0]
		
		v1 = []
		v2 = []
		for i in range(len(match)):
			u1, u2 = V1[i]+1, V2[i]+1
			if u1 in v1 or u2 in v2:
				continue 
			v1.append(u1)
			v2.append(u2)
			print(u1, '->', u2)
		
		print(A[np.ix_(match,match)])
		
		# for i in match:
			# for j in match[i+1:]:
				# if not A[i,j]:
					# print('porco dio')
					# exit()
		
		_ = plt.hist(ssd)
		plt.show()