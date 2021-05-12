import numpy as np

class WaterwayProfile():
	"""
	Stores the data and creates a waterway profile.
	"""
	def __init__(
		self,
		width,
		water_depth,
		talud_angle=0,
		channel_length=10000,
		current=0
		):
		"""Initialize the attributes of the waterway profile."""
		# The value for width should be the width of the surface level.
		self.width = width
		self.water_depth = water_depth
		# Provide angle in degrees and convert to radians.
		self.angle = talud_angle
		self.water_density = 1000
		self.air_density = 1.2
		self.channel_length = channel_length
		self.current = current

	def get_channel_cross_section(self): # TNS
		"""Calculate the channel cross-section."""
		# If angle is provided, calculate the cross-section with angle in rad.
		if self.angle:
			a_c = self.water_depth * (self.width \
				- self.water_depth / np.tan(self.angle * np.pi / 180))
		else:
			a_c = self.water_depth * self.width
		return a_c

	def get_channel_water_depth(self): # TNS
		"""Returns the channel water depth."""
		h_s = self.water_depth
		return h_s

	def get_water_density(self): # TNS
		"""Returns the water density."""
		rho_w = self.water_density
		return rho_w

	def get_air_density(self):
		"""Returns the water density."""
		rho_a = self.air_density
		return rho_a

	def get_channel_length(self): # TNS
		"""Returns the channel length."""
		L_ch = self.channel_length
		return L_ch

	def get_channel_current(self):
		"""
		Returns the current in the channel, standard is 0.
		Positive current means ship is sailing upstream.
		Negative current means ship is sailing downstream.
		"""
		U = self.current
		return U

# wp1 = WaterwayProfile(30, 10, 30)
# a_c = wp1.get_channel_cross_section()
# h_s = wp1.get_channel_water_depth()
# print(wp1.current)
# print(wp1.channel_length)