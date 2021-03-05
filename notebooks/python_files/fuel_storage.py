import numpy as np
import matplotlib.pyplot as plt

class Bunker():
	"""
	Defines the characteristics of the tank of a ship.
	Include the volume, the type of energy carriers, the limits.
	Diesel Direct Drive on M8 ship has a bunker volume of 46.5 m^3,
	Additional Engine Room space is 11.5 m^3, so total volume is 58 m^3.
	1 m^3 = 1000 L
	"""

	def __init__(
		self,
		energy_carrier_type,
		speed,
		starting_volume = 58000,
		empty_margin = 10,
		):
		"""
		Initialize the attributes of the ship's tank.
		Empty margin is the amount of fuel left as insurance for an empty tank.
		The value is set to 10 (percent).
		The energy carrier type allows for selection of energy carriers.
		Options: 'diesel', 'CNG', 'LNG', 'methanol', 'methylether', 'ammonia',
				 'hydrogenc', 'hydrogenl','LOHC', 'NaBH', 'linmcbat'
		"""
		self.volume = starting_volume
		self.ec = energy_carrier_type
		self.em = empty_margin
		self.speed = speed

	def energy_carrier_select(self):
		"""Defines the different energy carriers and uses their attributes."""
		if self.ec == 'diesel':
			# Energy carrier is diesel
			# Gross energy density from MJ/L to kWh/L
			# Efficiency is product of serie efficiencies in the fuel process,
			#  energy conversion, pre-treatment, after_treatment, power system
			energy_density = 31 / 3.6
			average_efficiency = 0.38 * 1.00 * 0.99 * 0.88
		elif self.ec == 'CNG':
			# Energy carrier is compressed natural gas
			energy_density = 8.5 / 3.6
		elif self.ec == 'LNG':
			# Energy carrier is LNG
			energy_density = 15 / 3.6
		elif self.ec == 'methanol':
			# Energy carrier is methanol
			energy_density = 14 / 3.6
		elif self.ec == 'methylether':
			# Energy carrier is methylether
			energy_density = 13.5 / 3.6
		elif self.ec == 'ammonia':
			# Energy carrier is ammonia
			energy_density = 9.6 / 3.6
		elif self.ec == 'hydrogenc':
			# Energy carrier is compressed hydrogen
			energy_density = 3.8 / 3.6
			average_efficiency = 0.46 * 0.98 * 1.00 * 0.97
		elif self.ec == 'hydrogenl':
			# Energy carrier is liquid hydrogen
			energy_density = 5 / 3.6
			average_efficiency = 0.46 * 0.92 * 1.00 * 0.97
		elif self.ec == 'LOHC':
			# Energy carrier is liquid organic hydrogen carrier
			energy_density = 5.5 / 3.6
		elif self.ec == 'NaBH':
			# Energy carrier is natriumborohydride
			energy_density = 15.3 / 3.6
		elif self.ec == 'linmcbat':
			# Energy carrier is li... battery
			energy_density = 0.4 / 3.6
			average_efficiency = 0.94 * 1.00 * 1.00 * 0.97
		return energy_density, average_efficiency

	def efficiency_curve(self, average_efficiency):
		"""Formulates the efficiency dependency on the sailing speed."""
		if self.speed < 0.3:
			n_i = 1.8 * (0.3 * self.speed)**2
			if n_i < 0:
				n = 0
			else:
				n = n_i
		elif self.speed < 1.0:
			n_i = 2.5 * (0.3 * self.speed)**2
			if n_i < 0:
				n = 0
			else:
				n = n_i
		elif self.speed < 1.4:
			n_i = 0.505 * (3.5 * self.speed)**2 - 2 * self.speed - 4
			if n_i < 0:
				n = 0
			else:
				n = n_i
		else:
			n_i = average_efficiency * 150 - 2.5 * self.speed - 2.1 \
				* ((self.speed - 5.671))**2 + 1 * (self.speed)**3 \
				- 0.18 * (self.speed + 0.25)**4 - 4
			if n_i < 0:
				n = 0
			else:
				n = n_i
		return n / 100

	def energy_amount(self, energy_density):
		"""Returns the stored amount of energy in kWh."""
		energy_amount = self.volume * energy_density
		return energy_amount
