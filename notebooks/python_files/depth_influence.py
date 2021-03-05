import numpy as np


# The velocity needs to be corrected for limited water depths,
#  when using the Holtrop & Mennen method.
def get_froude_depth_number(h, V_s):
	"""Determines the depth related Froude number and returns it."""
	Fr_h = V_s / np.sqrt(9.81 * h)
	return Fr_h

def karpov_factor(h_c, t_s, V_s, Fr_h):
	"""
	The Karpov method accounts for limited depth
	 when calculating resistance terms according to Holtrop & Mennen.
	The Froude number is the depth related froude number.
	Karpov changes the velocity input for the resistance terms,
	 excluding the skin friction resistance and appendage resistance.
	"""
	if Fr_h <= 0.4:
		if 0 <= h_c / t_s < 1.75:
			a_xx = (-4*10**(-12)) * Fr_h**3 - 0.2143 \
			* Fr_h**2 -0.0643 * Fr_h + 0.9997
		elif 1.75 <= h_c / t_s < 2.25:
			a_xx = -0.8333 * Fr_h**3 + 0.25 * Fr_h**2 \
			- 0.0167 * Fr_h + 1
		elif 2.25 <= h_c / t_s < 2.75:
			a_xx = -1.25 * Fr_h**4 + 0.5833 * Fr_h**3 \
			- 0.0375 * Fr_h**2 - 0.0108 * Fr_h + 1
		elif h_c / t_s >= 2.75:
			a_xx = 1

	elif Fr_h > 0.4:
		if 0 <= h_c / t_s < 1.75:
			a_xx = -0.9274 * Fr_h**6 + 9.5953 * Fr_h**5 - 37.197 * Fr_h**4 \
				+ 69.666 * Fr_h**3 - 65.391 * Fr_h**2 + 28.025 * Fr_h - 3.4143
		elif 1.75 <= h_c / t_s < 2.25:
			a_xx = 2.2152 * Fr_h**6 - 11.852 * Fr_h**5 + 21.499 * Fr_h**4 \
				- 12.174 * Fr_h**3 - 4.7873 * Fr_h**2 + 5.8662 * Fr_h - 0.2652
		elif 2.25 <= h_c / t_s < 2.75:
			a_xx = 1.2205 * Fr_h**6 - 5.4999 * Fr_h**5 + 5.7966 * Fr_h**4 \
				+ 6.6491 * Fr_h**3 - 16.123 * Fr_h**2 + 9.2016 * Fr_h - 0.6342
		elif 2.75 <= h_c / t_s < 3.25:
			a_xx = -0.4085 * Fr_h**6 + 4.534 * Fr_h**5 - 18.443 * Fr_h**4 \
				+ 35.744 * Fr_h**3 - 34.381 * Fr_h**2 + 15.042 * Fr_h - 1.3807
		elif 3.25 <= h_c / t_s < 3.75:
			a_xx = 0.4078 * Fr_h **6 - 0.919 * Fr_h**5 - 3.8292 * Fr_h**4 \
				+ 15.738 * Fr_h**3 - 19.766 * Fr_h**2 + 9.7466 * Fr_h - 0.6409
		elif 3.75 <= h_c / t_s < 4.5:
			a_xx = 0.3067 * Fr_h**6 - 0.3404 * Fr_h**5 - 5.0511 * Fr_h**4 \
				+ 16.892 * Fr_h**3 - 20.265 * Fr_h**2 + 9.9002 * Fr_h - 0.6712
		elif 4.5 <= h_c / t_s < 5.5:
			a_xx = 0.3212 * Fr_h**6 - 0.3559 * Fr_h**5 - 5.1056 * Fr_h**4 \
				+ 16.926 * Fr_h**3 - 20.253 * Fr_h**2 + 10.013 * Fr_h - 0.7196
		elif 5.5 <= h_c / t_s < 6.5:
			a_xx = 0.9252 * Fr_h**6 - 4.2574 * Fr_h**5 + 5.0363 * Fr_h**4 \
				+ 3.3282 * Fr_h**3 - 10.367 * Fr_h**2 + 6.3993 * Fr_h - 0.2074
		elif 6.5 <= h_c / t_s < 7.5:
			a_xx = 0.8442 * Fr_h**6 - 4.0261 * Fr_h**5 + 5.313 * Fr_h**4 \
				+ 1.6442 * Fr_h**3 - 8.1848 * Fr_h**2 + 5.3209 * Fr_h - 0.0267
		elif 7.5 <= h_c / t_s < 8.5:
			a_xx = 0.1211 * Fr_h**6 + 0.628 * Fr_h**5 - 6.5106 * Fr_h**4 \
				+ 16.7 * Fr_h**3 - 18.267 * Fr_h**2 + 8.7077 * Fr_h - 0.4745
		elif 8.5 <= h_c / t_s < 9.5:
			if Fr_h < 0.6:
				a_xx = 1
			elif Fr_h >= 0.6:
				a_xx = -6.4069 * Fr_h**6 + 47.308 * Fr_h**5 - 141.93 \
					* Fr_h**4 + 220.23 * Fr_h**3 - 185.05 * Fr_h**2 \
					+ 79.25 * Fr_h - 12.484
		elif h_c / t_s >= 9.5: 
			if Fr_h < 0.6:
				a_xx = 1
			elif Fr_h >= 0.6:
				a_xx = -6.0727 * Fr_h**6 + 44.97 * Fr_h**5 - 135.21 \
					* Fr_h**4 + 210.13 * Fr_h**3 - 176.72 \
					* Fr_h**2 + 75.728 * Fr_h - 11.893

	return a_xx

def karpov_speed(V_s, a_xx):
	"""
	Determines the karpov velocity using the original velocity
	 and the karpov factor.
	"""
	V_k = V_s / a_xx
	return V_k
