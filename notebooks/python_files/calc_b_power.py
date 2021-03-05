import numpy as np


def get_effective_power(R_t, V_s):
	"""
	Determines the required effective power, 
	using the velocity and the total resistance.
	"""
	P_e = R_t * (V_s)
	return P_e

def get_brake_power(P_e, n_h, n_o, n_r, n_s, n_gb):
	"""
	Determines the required brake power,
	using the effective power and the differen drive train efficiencies.
	"""
	P_b = P_e / (n_h * n_o * n_r * n_s * n_gb)
	return P_b

def get_gearbox_efficiency(): # TNS
	"""Defines the gearbox efficiency and returns it."""
	n_gb = 0.95
	return(n_gb)

def get_shaft_efficiency(): # TNS
	"""Defines the shaft efficiency and returns it."""
	n_s = 0.99
	return(n_s)

def get_relative_rotative_efficiency(): # TNS
	"""Defines the relative rotative efficiency and returns it."""
	n_r = 1.00
	return(n_r)

def get_open_water_efficiency(): # TNS
	"""Defines the open water efficiency and returns it."""
	n_o = 0.65
	return(n_o)

def get_hull_efficiency(t_s, l_s, V_s, cb, disp):
	"""Defines the hull efficiency and returns it."""
	Fr = V_s / np.sqrt(9.81 * l_s)
	# X is the number of propellers a ship has. Most inland ships have 1 or 2.
	x = 2 # TNS
	D = 0.70 * t_s # TNS
	
	if Fr < 0.2:
		dw = 0
	else:
		dw = 0.1 * (Fr - 0.2)

	w = 0.11 * 0.16 / x * cb * np.sqrt((disp**(1/3)) / D) - dw

	if x == 1:
		t = 0.6 * w * (1 + 0.67 * w)
	elif x == 2:
		t = 0.8 * w * (1 + 0.25 * w)
	else:
		print(f"x = {x}, t is not defined for this amount.")

	n_h = (1 - t) / (1 - w)

	return n_h
