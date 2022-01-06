import numpy as np

# Replicator Dynamics

def check_clique(A, C):
	for i, u in enumerate(C):
		for v in C[i+1:]:
			if A[u,v] == 0 or A[v,u] == 0:
				return False
	return True

def get_vertices(x):
	return np.where(np.logical_not(np.isclose(x, 0.)))[0]

def extract_match(x, n1, n2, off = 1):
	match = []
	for v in get_vertices(x):
		u1, u2 = v // n2, v % n2 
		match.append((u1 + off, u2 + off))
	return match

def is_indicator(x):
	i = 0
	while i < x.shape[0]:
		if x[i] != 0.:
			val = x[i]
			break 
		i += 1
	while i < x.shape[0]:
		if x[i] != 0. and x[i] != val:
			return False
	return True 
	
def replicator_dynamics(A, x, tol = 1.e-10, max_iter = 10000):
	dist = np.Inf
	iter = 0
	while dist > tol and iter < max_iter:
		old_x = x 
		Ax = A.dot(x)
		x = x * Ax / (x @ Ax)
		dist = np.linalg.norm(x - old_x)
		loop = False
		if dist <= tol and not is_indicator(x): #not check_clique(A, get_vertices(x)): 
			if dist <= tol:
				loop = True
				x = x + np.abs(np.random.normal(0, tol, x.shape[0])) 
				x /= np.sum(x)
				dist = np.linalg.norm(x - old_x)
		iter += 1
	return x

def simple_max_clique(A):
	max_clique = []
	clique = []
	
	def aux(n):
		for i in range(n, A.shape[0]):
			is_clique = True 
			for j in clique:
				if A[i,j] == 0 or A[j,i] == 0:
					is_clique = False
					break
			if is_clique:
				clique.append(i)
				if len(clique) > len(max_clique):
					max_clique.clear()
					for v in clique:
						max_clique.append(v)
				aux(i+1)
				clique.pop()
			aux(i+1)
	
	aux(0)
	return max_clique 