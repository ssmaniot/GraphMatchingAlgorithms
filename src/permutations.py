import numpy as np
import scipy.sparse as sps
from itertools import permutations, product

def generate_permutation_matrix_(perm):
	n = len(perm)
	return sps.coo_matrix((np.ones(n), (np.arange(n), perm)), shape = (n, n))

def generate_permutation_matrix(perm):
	n = len(perm)
	P = np.zeros(n ** 2, dtype = np.float).reshape((n, n))
	P[np.arange(n),perm] = 1.
	return P 

def sorted_perms(iterable, r=None):
	pool = tuple(sorted(iterable))
	n = len(pool)
	r = n if r is None else r
	for indices in product(range(n), repeat=r):
		if len(set(indices)) == r and tuple_is_sorted(indices):
			yield [pool[i] for i in indices]

memo = {}  # simple memoization for efficiency.
def tuple_is_sorted(t):
	return memo.setdefault(t, bool(sorted(t) == list(t)))

def binary_matrices(n):
	shift = np.arange(n ** 2).reshape(n, n)
	for j in range(2 ** (n ** 2)):
		yield j >> shift & 1

# Always accept problem as min f(A, B)
def find_global_best(f, A, B, mode = 'full'):
	n = A.shape[0]
	F0P = np.empty(np.math.factorial(n))
	iter = 0
	vertices = np.arange(n)
	P = np.zeros(n ** 2).reshape((n, n))
	best = np.Inf 
	bestp = []
	best_size = 1
	
	if mode == 'partial':
		for perm in permutations(vertices):
			found = False
			for window_size in range(best_size, n + 1):
			# for window_size in range(1, n + 1):
				for subperm in sorted_perms(vertices, r = window_size):
					sliced_perm = [perm[i] for i in subperm]
					window = np.ix_(subperm, sliced_perm)
					P[window] = 1.
					f0p = f(A, B, P)
					if f0p < best:
						found = True
						bestp.clear()
						bestp.append([np.array(subperm), np.array(sliced_perm)])
						best = f0p 
						best_size = window_size
					elif f0p == best:
						found = True
						bestp.append([np.array(subperm), np.array(sliced_perm)])
					P[window] = 0.
					iter += 1
				if not found:
					break
	elif mode == 'full':
		for perm in permutations(np.arange(n)):
			window = np.ix_(vertices, perm)
			P[window] = 1.
			f0p = f(A, B, P)
			if f0p < best:
				bestp.clear()
				bestp.append(perm)
				best = f0p 
			elif f0p == best:
				bestp.append(perm)
			P[window] = 0.
			iter += 1
	
	elif mode == 'binary':
		for bmat in binary_matrices(n):
			f0p = f(A, B, P)
			if f0p < best:
				bestp.clear()
				bestp.append(bmat)
				best = f0p
			elif f0p == best:
				bestp.append(bmat)
			iter += 1
	
	return bestp, iter

# Always accept problem as min f(A, B)
# def find_global_best(f, A, B, partial = False):
	# n = A.shape[0]
	# F0P = np.empty(np.math.factorial(n))
	# best = np.Inf 
	# iter = 0
	# niter = np.math.factorial(n)
	# bestp = []
	
	# if partial:
		# for perm in permutations(np.arange(n)):
			# P = generate_permutation_matrix(perm)
				# for window_size in range(1, n):
					# for window in sorted_perms(np.arange(n), r = window_size):
						# pass
						# f0p = f(A, B, P[np.ix_(window, window)])
		
	# else:
		# f0p = f(A, B, P)
		# F0P[iter] = f0p
		# is_best = f0p < best
		# if is_best:
			# best = f0p 
			# besti.append(iter)
			# bestp.append(perm)
		# elif f0p == best:
			# besti.append(iter)
			# bestp.append(perm)
		# iter += 1
		# if iter % 1000 == 0:
			# print('\r{:.3f}%'.format(iter/niter*100.), end = '')
	# print('\r{:.3f}%'.format(iter/niter*100.), end = '')
	# print('')
	# nbest = 0
	# for f0p in F0P:
		# if f0p == best:
			# nbest += 1
	# print('How many best?', nbest)
	# i = len(besti)
	# while i > 0:
		# if F0P[besti[i-1]] == best:
			# i -= 1
		# else:
			# break 
	# bestp = bestp[i:]
	# return np.array(bestp).reshape(len(bestp), len(bestp[0])), F0P 