import numpy as np

def minmax_prod(w1, w2):
	m1, m2 = np.min(w1), np.min(w2)
	M1, M2 = np.max(w1), np.max(w2)
	vals = [m1 * m2, m1 * M2, M1 * m2, M1 * M2]
	return vals 
	
def lower_bound(w1, w2):
	return min(minmax_prod(w1, w2))

def upper_bound(w1, w2):
	return max(minmax_prod(w1, w2))

def kron_eig_bound(G1, G2):
	"""
	If \lambda \in \sigma(G1) and \mu \in \sigma(G2) 
	=> \lambda\sigma \in \sigma(G1 \kron G2),
	Where \sigma(A) is the spectrum of A 
	https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/kronthesisschaecke04.pdf
	"""
	w1 , _ = np.linalg.eig(G1)
	w2 , _ = np.linalg.eig(G2)
	w1c, _ = np.linalg.eig(1 - G1 - np.identity(G1.shape[0]))
	w2c, _ = np.linalg.eig(1 - G2 - np.identity(G2.shape[0]))
	lmin = lower_bound(w1, w2) + lower_bound(w1c, w2c)
	lmax = upper_bound(w1, w2) + upper_bound(w1c, w2c)
	return lmin, lmax 

def main():
	pass 

if __name__ == '__main__':
	main()