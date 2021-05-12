import numpy as np
import matplotlib.pyplot as plt

from ship import Ship
from waterway_profile import WaterwayProfile
from resistances import Resistances

from sailing_speed import get_schijf_limit_speed, get_length_limit_speed
from sailing_speed import get_vessel_limit_speed, get_reynolds_number
from sailing_speed import get_ship_bottom_velocity

from depth_influence import get_froude_depth_number
from depth_influence import karpov_factor, karpov_speed

from calc_b_power import get_effective_power, get_brake_power
from calc_b_power import get_gearbox_efficiency, get_shaft_efficiency
from calc_b_power import get_relative_rotative_efficiency
from calc_b_power import get_open_water_efficiency, get_hull_efficiency


def run_calculations(
	beam,
	length,
	draught,
	width,
	water_depth,
	energy_carrier='diesel',
	x_results=1,
	sailing=0,
	speed=100,
	current=0,
	talud_angle=0,
	channel_length=10000,
	efficiency='profile',
	):
	"""Run the calculations by initiating the different functions."""
	
	# Make an instance for a ship and bunker.
	ship = Ship(beam, length, draught,)

	# Make an instance for a waterway section.
	waterway = WaterwayProfile(
		width,
		water_depth,
		talud_angle,
		channel_length,
		current,
		)
	
	# Calculate and call the ship attributes.
	l_s = ship.get_vessel_length()
	b_s = ship.get_vessel_beam()
	t_s = ship.get_vessel_draught()
	A_s = ship.get_vessel_underwater_cross_section()
	S_t = ship.get_wetted_surface_area()
	S_b = ship.get_wetted_flatbottom_surface()

	# Calculate and call the waterway attributes.
	A_c = waterway.get_channel_cross_section()
	h_c = waterway.get_channel_water_depth()
	rho_w = waterway.get_water_density()
	rho_a = waterway.get_air_density()
	L_ch = waterway.get_channel_length()
	U = waterway.get_channel_current()

	# Calculate both limit speeds and return the sailing speed.
	#  Calculate the velocity under the ship's bottom.
	V_lim_1 = get_schijf_limit_speed(A_s, A_c, h_c)
	V_lim_2 = get_length_limit_speed(l_s)
	
	# Set the speed as variable or constant
	if sailing == 0:
		V_s_1 = min(get_vessel_limit_speed(V_lim_1, V_lim_2), speed)
	elif sailing == 1:
		V_s_1 = speed
	V_s = V_s_1 + U

	V_b = get_ship_bottom_velocity(V_s, h_c, t_s)

	# Calculate the different ship coefficients.
	cb = ship.get_block_coefficient_full(V_s)
	cm = ship.get_midship_coefficient(cb)
	cw = ship.get_waterplane_coefficient(cb)
	cp = ship.get_prismatic_coefficient(cb, cm)

	# Calculate the Reynolds number and displacement.
	R_e = get_reynolds_number(V_s, l_s)
	disp = ship.get_displacement(cb)

	# Calculate the friction coefficients.
	c_f = ship.get_friction_coefficient(R_e)
	c_f_prop = ship.get_rbsw_friction_coefficient(R_e, h_c, t_s)
	c_f_katsui = ship.get_katsui_friction_coefficient(R_e)
	c_f_shallow = ship.get_shallow_water_friction_coefficient(
		c_f,
		c_f_prop,
		c_f_katsui,
		S_t,
		S_b,
		V_s,
		V_b
		)

	# Calculate the influence of the water depth, Karpov
	#  Depth related Froude number
	Fr_h = get_froude_depth_number(h_c, V_s)
	a_xx = karpov_factor(h_c, t_s, V_s, Fr_h)
	V_k = karpov_speed(V_s, a_xx)

	# Calculate the resistance terms
	resistance = Resistances(rho_w, rho_a, V_s, V_k, S_t, c_f_shallow)
	R_f = resistance.get_resistance_skin_friction()
	R_app = resistance.get_resistance_appendages_friction()
	k_1 = resistance.get_viscous_form_resistance(b_s, l_s, t_s, disp, cp)
	R_b = resistance.get_bulbous_bow_resistance()
	R_tr = resistance.get_transom_immersion_resistance(A_s, b_s, cw)
	R_a = resistance.get_model_ship_correlation_resistance(t_s, l_s, cb)
	R_w = resistance.get_wave_making_resistance(
		l_s,
		t_s,
		b_s,
		A_s,
		disp,
		cp,
		cw,
		cm,
		)
	R_t = resistance.get_total_resistance(R_f, k_1, R_app, R_w, R_b, R_tr, R_a)

	# Calculate the required brake power for propulsion at speed V_s.
	P_e = get_effective_power(R_t, V_s)
	n_gb = get_gearbox_efficiency()
	n_s = get_shaft_efficiency()
	n_r = get_relative_rotative_efficiency()
	n_o = get_open_water_efficiency()
	n_h = get_hull_efficiency(t_s, l_s, V_s, cb, disp)
	P_b = get_brake_power(P_e, n_h, n_o, n_r, n_s, n_gb)

	if x_results == 1:
		return R_f, R_app, k_1, R_b, R_tr, R_a, R_w, R_t, P_b, V_lim_1, \
			V_lim_2, cb, cm, cw, cp
	elif x_results == 2:
		return P_b
