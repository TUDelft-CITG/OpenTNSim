import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def get_schijf_limit_speed(A_s, A_c, h_s):
	"""Determine the Schijf limit speed and returns it."""
	A_s = A_s
	A_c = A_c
	h_s = h_s
	def schijf_limit_speed(V_lim_1):
		"""Define the function for the Schijf limit speed V_lim as x."""
		#global A_s, A_c, h_s
		return 1 - A_s / A_c + 0.5 \
			* (V_lim_1 / np.sqrt(9.81 * h_s))**2 \
			- 3/2 * (V_lim_1 / np.sqrt(9.81 * h_s))**(2/3)

	V_lim_1 = fsolve(schijf_limit_speed, 3)[0]
	return V_lim_1

def get_length_limit_speed(l_s):
	"""Determines the length limited ship velocity."""
	V_lim_2 = np.sqrt(9.81 * l_s * 0.5 / np.pi)
	return V_lim_2

def get_vessel_limit_speed(V_lim_1, V_lim_2):
	"""Returns the vessel length."""
	V_s = 0.90 * min(V_lim_1, V_lim_2)
	return V_s

def get_ship_bottom_velocity(V_s, h_s, T):
	"""Calculates and returns the velocity under the ship's bottom"""
	V_b = 0.4277 * V_s * np.exp((h_s / T)**-0.07634)
	return V_b

def get_reynolds_number(v_s, l_s, v=0.000001):
	"""Calculates the Reynolds number with kinematic viscosity 1E-6 m2/s."""
	R_e = v_s * l_s / v
	return R_e

def get_vessel_speed(n):
	"""Returns the sailing speed array"""
	if n == 0:
		V_pr = list(range(1, 8))
	elif n == 1:	
		V_pr = np.linspace(2.77, 5, 100)
	return V_pr
