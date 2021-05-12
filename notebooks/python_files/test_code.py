import numpy as np
from scipy.optimize import fsolve
from ship import ships

# from waterway_profile import WaterwayProfile
# from ship import Ship

# bmvb = [5.05, 6.60, 8.20, 4.70, 7.50, 8.20, 9.5, 11.40, 15.00]
# lmvb = [38.50, 55, 80, 41, 57, 70, 85, 110, 140]
# tmvb = [2.20, 2.50, 2.50, 1.40, 1.60, 2.0, 2.50, 2.80, 3.90]
# bpc = [8.2, 9.5, 11.40, 11.40, 22.80, 22.80, 22.80, 33.00, 33.00]
# lpc = [132, 85, 110, 185, 110, 195, 280, 200, 285]
# tpc = [2.0, 2.8, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]

# amvb = [i/j for i,j in zip(bmvb, lmvb)]
# Ldc15mvb = [(i**2) / (0.85 * j * k) for i,j,k in zip(lmvb, bmvb, tmvb)]
# apc = [i/j for i,j in zip(bpc,lpc)]
# Ldc15pc = [(i**2) / (0.85 * j * k) for i,j,k in zip(lpc, bpc, tpc)]

import matplotlib.pyplot as plt

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
	V_pr = [2, 3, 4, 5]

	# Create h profile
	h_pr = list(range(4, 10))
	i = 6

	P_bt_i, S_rt_i = [], []

# 	for j in V_pr:
# 		P_b, S_r = run_calculations(11.40, 110, 3.50, 50, i, 
# 								x_results=2, speed=j, sailing=0)
# 		P_bt_i.append(round(P_b / 1000000, 2))
# 		S_rt_i.append(round(S_r, 2))


# 	if plots == 1:
# 		plt.figure(figs)
# 		plt.plot(
# 			V_pr,
# 			P_bt_i,
# 			label=f"h = {i} m",
# 			)
# 		plt.xlabel("V (m/s)", fontsize=12)
# 		plt.ylabel("P (MW)", fontsize=12)
# 		plt.tick_params(axis='both', labelsize=12)
# 		plt.title("Brake Power, M8", fontsize=16)
# 	if plots == 2:
# 		plt.figure(figs)
# 		plt.plot(
# 			V_pr,
# 			S_rt_i,
# 			label=f"h = {i} m",
# 			)
# 		plt.xlabel("V (m/s)", fontsize=12)
# 		plt.ylabel("S_r (km)", fontsize=12)
# 		plt.tick_params(axis='both', labelsize=12)
# 		plt.title("Sailing Range, M8, Diesel", fontsize=16)

# 	# Plot and adjust the graph
# 	plt.rcParams["legend.loc"] = 'best'
# 	plt.rcParams["legend.shadow"] = True
# 	plt.grid(which='both', axis='both')
# 	plt.legend()


# plot_results_x_velocity(1, 1)
# plot_results_x_velocity(2, 2)

# plt.show()

average_efficiency = 0.38 * 1.00 * 0.99 * 0.88

V = np.linspace(0.01, 6, 1000)
n = []
for i in V:
	if i < 0.3:
		n_i = 1.8 * (0.3 * i)**2
		if n_i < 0:
			n.append(0)
		else:
			n.append(n_i)
	elif i < 1.0:
		n_i = 2.5 * (0.3 * i)**2
		if n_i < 0:
			n.append(0)
		else:
			n.append(n_i)
	elif i < 1.4:
		n_i = 0.505 * (3.5 * i)**2 - 2 * i -4
		if n_i < 0:
			n.append(0)
		else:
			n.append(n_i)
	else:
		n_i = average_efficiency * 150 - 2.5 * i - 2.1 * ((i - 5.671))**2 \
			+ 1 * (i)**3 - 0.18 * (i + 0.25)**4 -4
		if n_i < 0:
			n.append(0)
		else:
			n.append(n_i)


plt.plot(V * 3.6, n)
# plt.axes([0, 10, 0, 100])
plt.xlabel("V (km/h)", fontsize=12)
plt.ylabel("n (%)", fontsize=12)
plt.tick_params(axis='both', labelsize=12)
plt.title("Engine Efficiency", fontsize=16)
plt.grid(which='both', axis='both')

plt.show()