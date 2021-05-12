import numpy as np
import matplotlib.pyplot as plt
import mplcursors

c_b = np.linspace(0.70, 0.90, 100)

c_m, c_m_2 = [], []

for i in c_b:
	c_m_i = 1 / (1 + (1 - i)**3.5)
	c_m_2_i = 1.006 - 0.0056 * i ** (-3.56)
	c_m.append(c_m_i)
	c_m_2.append(c_m_2_i)

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(c_b, c_m)
ax1.set_title('c_m_nils')
ax2.plot(c_b, c_m_2)
ax2.set_title('c_m_loes')

mplcursors.cursor()

plt.show()