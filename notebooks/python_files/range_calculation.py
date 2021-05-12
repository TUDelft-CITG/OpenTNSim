import numpy as np
import matplotlib.pyplot as plt
import mplcursors

from fuel_storage import Bunker
from calc_e_consumption import run_calculations

# Sailing speed
V_s = 3.5 # 3.5 * 3.6 = 12.6 km/h

# Water depth
h = 5

# Determine the brake power, M8 
P_b = run_calculations(11.40, 110, 3.50,
						50, h, 
						x_results=2, speed=V_s, sailing=0)
# Create Bunker instance:
energy_carriers = ['diesel', 'linmcbat', 'hydrogenl', 'hydrogenc']

for ec in energy_carriers:
	bd = Bunker(ec, V_s)
	# energy density e_d (kWh/L), engine efficiency n_e (-),
	#  total stored energy E_tot (kWh)
	e_d, n_e = bd.energy_carrier_select()

	# Optional efficiency profile
	# if efficiency == 'profile'
	# 	n = bd.efficiency_curve(n_e)
	# else:
	# 	n = n_e

	# Initiate a time profile in hrs.
	t = np.linspace(0, 480, 10000)
	x = []

	E_demand = []
	E_consumption = []
	E_bunker = []

	# Initial bunker level in kWh
	E_0 = bd.energy_amount(e_d)

	for t_i in t:
		# Distance in km
		x_i = t_i * V_s * 3.6
		x.append(x_i)
		# Energy demand
		E_d = P_b * t_i * 3600
		E_demand.append(E_d)
		# Calculate the energy consumption
		E_c = E_d / n_e
		E_consumption.append(E_c / 3600000)
		# Bunker Level
		E_b = E_0 - E_c / 3600000
		E_bunker.append(E_b)

	# E_bunker_n = [x for x in E_bunker if x >= 0]
	# t_n = np.linspace(0, 480, len(E_bunker_n))

	# plt.figure(1)
	# plt.plot(
	# 	t,
	# 	E_consumption,
	# 	label=f"h = {h} m, V = {V_s} m/s, Energy Carrier = {ec}",
	# 	)
	# plt.xlabel("t (hr)", fontsize=12)
	# plt.ylabel("E_c (kWh)", fontsize=12)
	# plt.ticklabel_format(style='plain')
	# plt.tick_params(axis='both', labelsize=12)
	# plt.title("Fuel Consumption, M8", fontsize=16)
	# plt.rcParams["legend.loc"] = 'best'
	# plt.rcParams["legend.shadow"] = True
	# plt.legend()
	# plt.ylim(ymin=0)

	plt.figure(1)
	plt.plot(
		x,
		E_bunker,
		label=f"h = {h} m, V = {V_s * 3.6} km/h, Energy Carrier = {ec}",
		)
	plt.xlabel("x (km)", fontsize=12)
	plt.ylabel("E_bunker (kWh)", fontsize=12)
	plt.tick_params(axis='both', labelsize=12)
	plt.title("Bunker Content, M8, Bunker Volume 58 m3", fontsize=16)
	plt.rcParams["legend.loc"] = 'best'
	plt.rcParams["legend.shadow"] = True
	plt.legend()
	plt.xlim(xmax=3000)
	plt.ylim(ymin=0)

	plt.figure(2)
	plt.plot(
		t / 24,
		E_bunker,
		label=f"h = {h} m, V = {V_s * 3.6} km/h, Energy Carrier = {ec}",
		)
	plt.xlabel("t (days)", fontsize=12)
	plt.ylabel("E_bunker (kWh)", fontsize=12)
	plt.tick_params(axis='both', labelsize=12)
	plt.title("Bunker Content, M8, Bunker Volume 58 m3", fontsize=16)
	plt.rcParams["legend.loc"] = 'best'
	plt.rcParams["legend.shadow"] = True
	plt.legend()
	plt.xlim(xmax=10)
	plt.ylim(ymin=0)

plt.grid(which='both', axis='both')

mplcursors.cursor()

plt.show()


# Calculate the distance sailed on 1 tank in km
# E_tot (kWh), V_s (m/s), P_b (W), S_r (m)

# S_r = 3.6 * E_tot * n * V_s / (P_b / 1000000)

# print(S_r)