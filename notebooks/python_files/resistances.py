import numpy as np

from ship import Ship

class Resistances():
	"""A class to manage the resistances encountered by the ship."""

	def __init__(
			self,
			rho_w,
			rho_a,
			V_s,
			V_k,
			S_t,
			c_f_shallow,
			C_stern=10,
			):
		"""Initialize the variables used in the resistance equations"""
		self.water_density = rho_w
		self.air_density = rho_a
		self.velocity = V_s
		self.karpov_velocity = V_k
		self.wetted_area = S_t
		self.shallowcoefficient = c_f_shallow
		self.C_stern = C_stern

	def get_resistance_skin_friction(self):
		"""Defines the equation for the skin friction resistance."""
		R_f = 0.5 * self.water_density * self.velocity**2 \
			* self.wetted_area * self.shallowcoefficient
		return R_f

	def get_resistance_appendages_friction(self):
		"""Defines the equation for the appendages skin friction resistance."""
		S_app = 0.064 * self.wetted_area
		k_2 = 3.0
		R_app = 0.5 * self.water_density * self.velocity**2 \
			* S_app * self.shallowcoefficient * (k_2)
		return R_app

	def get_viscous_form_resistance(self, b_s, l_s, t_s, disp, cp):
		"""Calculates the viscous form resistance and returns it."""
		l_cb = -0.135 + 0.194 * cp
		l_r = l_s * (1 - cp + 0.06 * cp * l_cb / (4 * cp -1))
		k_1 = 0.93 + 0.487118 * (1 + 0.011 * self.C_stern) \
			* (b_s / l_s)**1.06806 * (t_s / l_s)**0.46106 \
			* (l_s / l_r)**0.121563 * ((l_s**3) / disp)**0.36486 \
			* (1 - cp)**(-0.604247)
		return k_1

	def get_wave_making_resistance(
			self,
			l_s,
			t_s,
			b_s,
			A_s,
			disp,
			cp,
			cw,
			cm,
			k=0.2,
			):
		"""Calculates the wave making resistance and returns it."""
		A_tr = k * A_s
		Fr_wl = self.karpov_velocity / np.sqrt(9.81 * l_s)
		l_cb = -0.135 + 0.194 * cp
		l_r = l_s * (1 - cp + 0.06 * cp * l_cb / (4 * cp -1))
		x_s = b_s / l_s
		y_s = l_s / b_s
		
		if x_s < 0.11:
			c7 = 0.229577 * (x_s)**(1/3)
		elif 0.11 < x_s < 0.25:
			c7 = x_s
		else:
			c7 = 0.5 - 0.0625 * y_s
		
		ie = 1 + 89 * np.exp(-1 * x_s**0.80856 * (1 - cw)**0.30484 \
			* (1 - cp - 0.0225 * l_cb)**0.6367 * (l_r / b_s)**0.34574 \
			* (100 * disp / l_s**3)**0.16302)
		c1 = 2223105 * c7**3.78613 * (t_s / b_s)**1.07961 \
			* (90 - ie)**(-1.37565)
		c2 = 1.0
		c5 = 1 - 0.8 * A_tr / (b_s * t_s * cm)
		
		if (l_s**3) / disp < 512:
			c15 = -1.69385 
		elif 512 < (l_s**3) / disp < 1726.91:
			c15 = -1.69385 + (l_s / (disp**(1/3)) - 8) / 2.36
		else:
			c15 = 0

		if cp < 0.8:
			c16 = 8.07981 * cp - 13.8673 * cp**2 + 6.984388 * cp**3
		else:
			c16 = 1.73014 - 0.7067 * cp

		m_1 = 0.0140407 * l_s / t_s - 1.75254 * (disp**(1/3)) / l_s \
			- 4.79323 * x_s - c16
		m_4 = c15 * 0.4 * np.exp(-0.034 * Fr_wl**(-3.29))
		
		if y_s < 12:
			lba = 1.446 * cp - 0.03 * y_s 
		else:
			lba = 1.446 * cp - 0.36

		R_w = c1 * c2 * c5 * disp * self.water_density * 9.81 \
			* np.exp(m_1 * Fr_wl**(-0.9) + m_4 * np.cos(lba * Fr_wl**(-2.0)))

		return R_w

	def get_bulbous_bow_resistance(self):
		"""
		Calculates the bulbous bow resistance and returns it.
		No bulbous bow means no resistance contribution.
		Other bows might contribute. For now R_b = 0.
		"""
		R_b = 0
		return R_b

	def get_transom_immersion_resistance(self, A_s, b_s, cw, k=0.2):
		"""Calculates the transom immersion resistance and returns it."""
		A_tr = k * A_s
		Fr_T = self.karpov_velocity / np.sqrt(2 * 9.81 * A_tr / (b_s * (1 + cw)))
		# if Fr_T < 5:
		# 	C_tr = 0.2 * (1 - Fr_T / 5)
		# else:
		# 	C_tr = 0
		C_tr = 0.2 * (1 - Fr_T / 5)
		R_tr = 0.5 * self.water_density * self.velocity**2 * A_tr * C_tr
		return R_tr

	def get_model_ship_correlation_resistance(self, t_s, l_s, cb):
		"""Calculates the model-ship correlation resistance and returns it."""
		t_f = 0.85 * t_s
		c2 = 1.0
		if t_f / l_s <= 0.04:
			c4 = t_f / l_s
		else:
			c4 = 0.04
		C_air = 0.006 * (l_s + 100)**(-0.16) - 0.00205 \
			+ 0.003 * np.sqrt(l_s / 7.5) * cb**4 * c2 * (0.04 - c4)
		R_a = 0.5 * self.air_density * self.karpov_velocity**2 \
			* self.wetted_area * C_air
		return R_a

	def get_total_resistance(self, R_f, k_1, R_app, R_w, R_b, R_tr, R_a):
		"""Combines the various resistances in one resistance term."""
		R_t = R_f * k_1 + R_app + R_w + R_b + R_tr + R_a
		return R_t
		