import matplotlib.pyplot as plt
import numpy as np

from calc_e_consumption import run_calculations
from sailing_speed import get_vessel_speed


def plot_results_x_velocity(plots, figs):
	"""
	Select the data which needs to be plotted in which figure.
	plots has options for values 1-xx, each returning a different term.
	For figure, you can chose which figure number you want which plot.
	Velocity on the x-axis.
	"""
	# Creates the values for the sailing speed.
	# V_pr = get_vessel_speed(1)
	V_pr = np.linspace(3.25, 5.0, 1000)
	# Create h profile
	h_pr = list(range(4, 10))

	for i in h_pr:
		P_bt_i = []
		S_rt_i = []

		for j in V_pr:
			P_b, S_r = run_calculations(11.40, 110, 3.50, 50, i,
									efficiency='n',
									energy_carrier='diesel',
									x_results=2, speed=j, sailing=0)
			P_bt_i.append(round(P_b / 1000, 2))
			S_rt_i.append(round(S_r / 1000, 2))
	
		if plots == 1:
			plt.figure(figs)
			plt.plot(
				V_pr * 3.6,
				P_bt_i,
				label=f"h = {i} m",
				)
			plt.xlabel("V (km/h)", fontsize=12)
			plt.ylabel("P (MW)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Brake Power, M8", fontsize=16)
		if plots == 2:
			plt.figure(figs)
			plt.plot(
				V_pr * 3.6,
				S_rt_i,
				label=f"h = {i} m",
				)
			plt.xlabel("V (km/h)", fontsize=12)
			plt.ylabel("S_r (km)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Sailing Range, M8, Diesel", fontsize=16)

	# Plot and adjust the graph
	plt.rcParams["legend.loc"] = 'best'
	plt.rcParams["legend.shadow"] = True
	plt.grid(which='both', axis='both')
	plt.legend()


# plot_results_x_velocity(1, 1)
plot_results_x_velocity(2, 2)

plt.show()