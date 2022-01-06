from permutations import *
from time import time

if __name__ == '__main__':
	A = np.array([
		[0, 1, 1, 1, 1, 1],
		[1, 0, 1, 1, 1, 1],
		[1, 1, 0, 1, 1, 1],
		[1, 1, 1, 0, 1, 1],
		[1, 1, 1, 1, 0, 1],
		[1, 1, 1, 1, 1, 0]
	])
	
	B = A.copy()
	# B[0, 1] = 0
	# B[1, 0] = 0
	
	A = np.array([
		[0, 1, 1, 0, 0, 0, 0],
		[1, 0, 1, 0, 0, 0, 1],
		[1, 1, 0, 1, 1, 0, 0],
		[0, 0, 1, 0, 0, 1, 0],
		[0, 0, 1, 0, 0, 1, 1],
		[0, 0, 0, 1, 1, 0, 0],
		[0, 1, 0, 0, 1, 0, 0]
	])
	
	B = np.array([
		[0, 1, 0, 0, 0, 0, 1],
		[1, 0, 1, 1, 0, 0, 0],
		[0, 1, 0, 0, 0, 1, 0],
		[0, 1, 0, 0, 1, 1, 0],
		[0, 0, 0, 1, 0, 0, 1],
		[0, 0, 1, 1, 0, 0, 0],
		[1, 0, 0, 0, 1, 0, 0]
	])
	
	A = np.array([
		[0, 1, 0, 0],
		[1, 0, 1, 1],
		[0, 1, 0, 1],
		[0, 1, 1, 0]
	])
	
	B = A.copy()
	B[0, 1] = 0
	B[1, 0] = 0
	
	def f(A, B, P):
		return -np.trace(A @ P @ B.T @ P.T) + np.trace(P @ B.T @ P @ P.T @ B @ P)
	
	n = A.shape[0]
	alpha = 1.
	Ac = alpha * (1 - A - np.identity(n))
	Bc = alpha * (1 - B - np.identity(n))
	
	def ff(A, B, P):
		return -np.trace(A @ P @ B @ P.T + Ac @ P @ Bc @ P.T)
	
	start = time()
	bestp, iters = find_global_best(ff, A, B, mode = 'partial')
	print('Elapsed time:', time() - start)
	for i, perm in enumerate(bestp):
		# print(perm)
		if True:
			print(perm)
			a, b = perm[0], perm[1]
			assert len(a) == len(b)
			print('Permutation {}:'.format(i + 1))
			for j in range(len(a)):
				print(a[j] + 1, '->', b[j] + 1)