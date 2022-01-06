import numpy as np

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
	while dist > tol and iter < max_iter:
		old_x = x 
		Ax = A.dot(x)
		x = x * Ax / (x @ Ax)
		dist = np.linalg.norm(x - old_x)
		if dist <= tol and not check_clique(A, extract_clique(x)):
			x = x + tol * np.random.normal(0, 1, x.shape[0]) 
			x[x<0] = 0
			x /= np.sum(x)
		iter += 1
	return x 

if __name__ == '__main__':
	A1 = np.array([
		[0, 1, 0, 0],
		[1, 0, 1, 1],
		[0, 1, 0, 1],
		[0, 1, 1, 0]
	])
	
	A2 = A1.copy()
	# A2[0,1] = 0
	# A2[1,0] = 0
	
	A = np.kron(A1, A2)
	n = A.shape[0]
	AI = A + np.identity(n) / 2
	x = np.ones(n) 
	x /= np.sum(x)
	ssd = replicator_dynamics(AI, x)
	print(ssd)
	print('Is clique?', check_clique(AI, extract_clique(ssd)))
	print(ssd[np.where(np.logical_not(np.isclose(ssd, 0.)))[0]])