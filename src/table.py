import numpy as np

def latex_mean(data):
	mean = np.round(np.mean(data, axis = 2), 2)
	std = np.round(np.std(data, axis = 2), 2)
	print('\\hline')
	for i in range(data.shape[0]):
		print(f'{(i+1)*10} & ', end = '')
		for j in range(data.shape[1]):
			print(f'{mean[i,j]} $\\pm$ {std[i,j]} ', end = '')
			if j < data.shape[1] - 1:
				print('& ', end = '')
		print('\\\\')
		print('\\hline')

def latex_exact(data):
	exact = np.sum(data == 1, axis = 2)
	print('\\hline')
	for i in range(data.shape[0]):
		print(f'{(i+1)*10} & ', end = '')
		for j in range(data.shape[1]):
			print(f'{exact[i,j]} ', end = '')
			if j < data.shape[1] - 1:
				print('& ', end = '')
		print('\\\\')
		print('\\hline')

data = np.load('results.npy')
latex_mean(data)
print('\n----\n')
latex_exact(data)