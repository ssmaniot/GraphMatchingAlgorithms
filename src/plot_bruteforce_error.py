import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	data = np.load('results_bruteforce_perm.npz')
	nvert = data['nvert']
	perm_error = data['perm_error']
	plt.boxplot(perm_error.T, labels = nvert)
	plt.show()