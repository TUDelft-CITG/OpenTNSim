import numpy as np
from scipy.optimize import fsolve

from waterway_profile import WaterwayProfile as WP
from sailing_speed import get_schijf_limit_speed
from sailing_speed import get_length_limit_speed
from sailing_speed import get_vessel_limit_speed


class Ship():
	"""
	Defines the characteristics of the ship.
	Dimensions in m, m/s
	"""

	def __init__( # TNS
		self,
		beam,
		length,
		draught,
		# draught_empty,
		# draught_full,
		# tank_capacity,
		):
		"""Initialize the ship's attribute values."""
		self.vessel_width = beam
		self.vessel_length = length
		self.vessel_draught = draught
		# self.draught_empty = draught_empty
		# self.draught_full = draught_full
		# self.tank_capacity = tank_capacity

	def get_vessel_length(self): # TNS
		"""Returns the vessel length."""
		l_s = self.vessel_length
		return l_s

	def get_vessel_draught(self): # TNS
		"""Returns the vessel draught."""
		t_s = self.vessel_draught
		return t_s

	def get_vessel_beam(self): # TNS
		"""Returns the vessel beam."""
		b_s = self.vessel_width
		return b_s

	def get_wetted_surface_area(self):
		"""Calculates the wetted surface area and returns it."""
		S_t = self.vessel_length * (2 * self.vessel_draught \
			+ self.vessel_width)
		return S_t

	def get_wetted_flatbottom_surface(self):
		"""Returns the wetted surface of the flat bottom of the ship."""
		l_s = self.vessel_length
		b_s = self.vessel_width
		S_b = 0.63 * l_s * b_s
		return S_b

	def get_vessel_underwater_cross_section(self):
		"""Returns the underwater cross-section of the vessel."""
		A_s = self.vessel_width * self.vessel_draught
		return A_s

	def get_block_coefficient_full(self, v_s):
		"""
		Determines and returns the block coefficient
		of a fully laden/empty ship.
		"""
		# Rewrites average vessel speed, and vessel length
		#  to knots and feet respectively.
		v_a = v_s / 0.5144
		l_f = self.vessel_length / 0.3048

		k_1 = 1.33 - 0.54 * v_a / np.sqrt(l_f) + 0.24 * (v_a / np.sqrt(l_f))**2
		c_b = min(k_1 - 0.5 * v_a / np.sqrt(l_f), 0.90)
		return c_b
	
	def get_midship_coefficient(self, c_b): # TNS
		"""Returns the midship section coefficient."""
		c_m = 1 / (1 + (1 - c_b)**3.5)
		return c_m

	def get_waterplane_coefficient(self, c_b): # TNS
		"""Returns the midship section coefficient."""
		c_wl = (1 + 2 * c_b) / 3
		return c_wl

	def get_prismatic_coefficient(self, c_b, c_m): # TNS
		"""Returns the prismatic coefficient."""
		c_p = c_b / c_m
		return c_p

	def get_displacement(self, c_b): # TNS
		"""Calculate and returns the displacement."""
		disp = c_b * self.vessel_width * self.vessel_length \
			* self.vessel_draught
		return disp

	def get_friction_coefficient(self, Re):
		"""Calculates and returns the vessel speed."""
		c_f = 0.075 / ((np.log10(Re)-2)**2)
		return c_f

	def get_rbsw_friction_coefficient(self, Re, h, T):
		"""
		Calculates and returns the regression based shallow water coefficient.
		"""
		d = h - T
		L_B = 0.70 * self.vessel_length
		A = 0.08169 / ((np.log10(Re)-1.717)**2)
		B = 0.003998 / (np.log10(Re) - 4.393)
		C = (d / L_B)**(-1.083)
		c_f_prop = A * (1 + B * C)
		return c_f_prop

	def get_katsui_friction_coefficient(self, Re):
		"""Calculates and returns the Katsui friction coefficient."""
		a = 0.042612 * np.log10(Re) + 0.56725
		c_f_katsui = 0.0066577 / ((np.log10(Re) - 4.362)**a)
		return c_f_katsui

	def get_shallow_water_friction_coefficient(
		self,
		c_f,
		c_f_prop,
		c_f_katsui,
		S_t,
		S_b,
		V_s,
		V_b,
		):
		"""Calculates the shallow water friction coefficient and returns it."""
		S = S_b / S_t
		V = V_b / V_s
		c_f_shallow = c_f + (c_f_prop - c_f_katsui) * (S) * (V)**2
		return c_f_shallow

# RWS Classes
# Motor Vessels 
# M1, ship_1 = Ship(5.05, 38.50, 2.50, )
# M2, ship_2 = Ship(6.60, 55, 2.60, )
# M3, ship_3 = Ship(7.20, 70, 2.60, )
# M4, ship_4 = Ship(8.20, 73, 2.70, )
# M5, ship_5 = Ship(8.20, 85, 2.70, )
# M6, ship_6 = Ship(9.5, 85, 2.90, )
# M7, ship_7 = Ship(9.5, 105, 3.00, )
# M8, ship_8 = Ship(11.40, 110, 3.50, )
# M9, ship_9 = Ship(11.40, 135, 3.50, )
# M10, ship_10 = Ship(13.50, 110, 4.00, )
# M11, ship_11 = Ship(14.20, 135, 4.00, )
# M12, ship_12 = Ship(17.00, 135, 4.00, )

# Barges
# BO1, ship_10 = Ship(5.2, 55, 1.90, )
# BO2, ship_11 = Ship(6.6, 70, 2.60, )
# BO3, ship_12 = Ship(7.5, 80, 2.60)
# BO4, ship_13 = Ship(8.2, 85, 2.70)
# BI, ship_14 = Ship(9.5, 105, 3.00)
# BII-1, ship_15 = Ship(11.40, 110, 3.50)
# BIIa-1, ship_16 = Ship(11.40, 110, 4.00)
# BIIL-1, ship_17 = Ship(11.40, 135, 4.00)
# BII-2I, ship_18 = Ship(11.40, 190, 4.00)
# BII-2b, ship_19 = Ship(22.80, 145, 4.00)
# BII-4, ship_20 = Ship(22.80, 195, 4.00)
# BII-6I, ship_21 = Ship(22.80, 270, 4.00)
# BII-6b, ship_22 = Ship(34.20, 195, 4.00)

# Convoys
# C1l, ship_23 = Ship(5.05, 80, 2.50)
# C1b, ship_24 = Ship(10.10, 38.5, 2.50)
# C2l, ship_25 = Ship(9.50, 185, 3.00)
# C3l, ship_26 = Ship(11.40, 190, 4.00)
# C2b, ship_27 = Ship(19.00, 105, 3.00)
# C3b, ship_28 = Ship(22.80, 110, 4.00)
# C4, ship_29 = Ship(22.80, 185, 4.00)

# ships = [ship_1, ship_2, ship_3, ship_4, ship_5,
# 		ship_6, ship_7, ship_8, ship_9]
# ships.append(ship_10)
# ships.append(ship_11)
# ships.append(ship_12)
# ships.append(ship_13)
# ships.append(ship_14)
# ships.append(ship_15)
# ships.append(ship_16)
# ships.append(ship_17)
# ships.append(ship_18)

# ships = {}
# ships["M1"] = (5.05, 38.50, 2.50)
# ships["M2"] = (6.60, 55, 2.60)
# ships["M3"] = (7.20, 70, 2.60)
# ships["M4"] = (8.20, 73, 2.70)
# ships["M5"] = (8.20, 85, 2.70)
# ships["M6"] = (9.5, 85, 2.90)
# ships["M7"] = (9.5, 105, 3.00)
# ships["M8"] = (11.40, 110, 3.50)
# ships["M9"] = (11.40, 135, 3.50)
# ships["M10"] = (13.50, 110, 4.00)
# ships["M11"] = (14.20, 135, 4.00)
# ships["M12"] = (17.00, 135, 4.00)

ships = {'M1': (5.05, 38.5, 2.5),
		 'M2': (6.6, 55, 2.6),
		 'M3': (7.2, 70, 2.6),
		 'M4': (8.2, 73, 2.7),
		 'M5': (8.2, 85, 2.7),
		 'M6': (9.5, 85, 2.9),
		 'M7': (9.5, 105, 3.0),
		 'M8': (11.4, 110, 3.5),
		 'M9': (11.4, 135, 3.5),
		 #'M10': (13.5, 110, 4.0),
		 #'M11': (14.2, 135, 4.0),
		 #'M12': (17.0, 135, 4.0)
		 }

# for i, j in ships.items():
# 	print(i)
# 	print(j)
