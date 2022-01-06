from script import *
import matplotlib.pyplot as plt

"""

min s @ grad(x) => s = 0
min f(x + alpha * (s - x)) => alpha = 1
=> f(s) 

"""
def main():
	iters = 200
	A = np.array([[0., 1., 0., 1., 1., 1., 1., 1., 1., 1.],
			   [1., 0., 1., 0., 1., 1., 1., 1., 1., 0.],
			   [0., 1., 0., 0., 1., 1., 1., 1., 1., 0.],
			   [1., 0., 0., 0., 1., 1., 0., 1., 1., 0.],
			   [1., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
			   [1., 1., 1., 1., 1., 0., 1., 1., 0., 0.],
			   [1., 1., 1., 0., 1., 1., 0., 0., 1., 1.],
			   [1., 1., 1., 1., 1., 1., 0., 0., 1., 1.],
			   [1., 1., 1., 1., 1., 0., 1., 1., 0., 1.],
			   [1., 0., 0., 0., 1., 0., 1., 1., 1., 0.]])
	n = A.shape[0]
	B = 1 - A - np.eye(n)
	m, M = sigma_minmax(B - np.eye(n))
	MU = np.linspace(m+0.1, -(M+0.1), iters)
	BDs = [(0., 1.) for _ in range(n)]
	x = np.ones(n) / 2
	max_iter = 1000
	func = []
	adaptive = False 
	plot = False   
	Alpha = np.linspace(0., 1., 100)
	np.seterr(all = 'warn')
	MU = np.array([-(M+0.1)])
	for mu in MU:
		f, grad = FUN_GRAD(B, mu)
		# Frank Wolfe algorithm
		k = 0
		dist = np.Inf
		ff = []
		while dist > 1.e-15 and k < max_iter:
			print(f'\rIter {k}/{max_iter}', end = '')
			ff.append(f(x))
			x_old = x 
			# min s @ grad(x)
			s = linprog(grad(x), bounds = BDs).x
			# min f(x + alpha * (s - x)), 0 <= alpha <= 1
			if adaptive:
				# print("I'm adaptive!")
				if plot:
					y = np.array([f(x + a * (s - x)) for a in Alpha])
					plt.plot(Alpha, y, '-')
					plt.xlabel('alpha')
					plt.ylabel('y')
					plt.title(f'Linesearch at iteration {k}')
					plt.show()
					plt.clf()
				# print('why am I not plotting??')
				alpha = line_search(f, grad, x, s - x, amax = 1.)[0]
				if alpha is None:
					alpha = 1.
			else:
				alpha = 2 / (2 + k)
			x = x + alpha * (s - x)
			# print(x)
			dist = np.linalg.norm(x - x_old)
			k += 1
			# if k == 10:
				# exit()
		print(f'\rIter {k}/{max_iter}, DONE')
		ff.append(f(x))
		if plot:
			plt.plot(np.arange(k + 1), ff, 'o')
			plt.xlabel('k')
			plt.ylabel('f(x)')
			plt.title('Frank-Wolfe Algorithm')
			plt.show()
			plt.clf()
		func.append(f(x))
	plt.plot(MU, func, 'o')
	plt.xlabel('mu')
	plt.ylabel('f(x,mu)')
	plt.title('f(x) as mu increase')
	plt.show()
	plt.clf()
	x[x<.5] = 0.
	x[x>0.] = 1.
	i = np.nonzero(x)[0]
	print(B[np.ix_(i,i)])

if __name__ == '__main__':
    main()