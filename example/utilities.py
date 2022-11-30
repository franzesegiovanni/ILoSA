import numpy as np

def plot_model(data):

	import matplotlib.pyplot as plt

	pos_data = [d.pos for d in data.demos]
	speed_data = [np.sqrt(np.sum(d.vel**2,0)) for d in data.demos]

	# plt.plot(speed_data[0])
	# plt.show()
	# plt.figure()
	fig = plt.figure()

	fig.suptitle(data.name, fontsize=16)

	f1_ax1 = plt.subplot(2,1,1)
	f1_ax1.set_title("Position Trajectory")

	f1_ax2 = plt.subplot(2,1,2)
	f1_ax2.set_title("Speed Profile")

	for i, vals in enumerate(pos_data):
		lines = f1_ax1.plot(vals[0,:], vals[1,:],'b')
		start = f1_ax1.scatter(vals[0,0], vals[1,0], c = 'k', marker = 'x')
		target = f1_ax1.scatter(vals[0,-1], vals[1,-1], c = 'k', marker = '*')
		f1_ax2.plot(speed_data[i],'b')

	start.set_label("Starting Points")
	target.set_label("Target")

	f1_ax1.set_xlabel("x (mm)")	
	f1_ax1.set_ylabel("y (mm)")	

	f1_ax2.set_xlabel("time (s)")	
	f1_ax2.set_ylabel("speed (mm/s)")	
	f1_ax1.legend()
	plt.axis('tight')
	plt.show()



if __name__ == '__main__':
	import pyLasaDataset as lasa

	plot_model(lasa.Angle)


