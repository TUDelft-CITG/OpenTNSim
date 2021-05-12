import matplotlib.pyplot as plt
import numpy as np

from ship import ships
from calc_e_consumption import run_calculations
from sailing_speed import get_vessel_speed


def plot_results_x_ship_size(plots, figs):
	"""
	Select the data which needs to be plotted in which figure.
	Velocity on the x-axis.
	"""
	h_pr = np.linspace(4, 10, 100)
	V_pr = get_vessel_speed(1)
	
	# Unpack the required values from the ship dictionary

	for k, i in ships.items():
		R_ft_i, R_appt_i, R_bt_i, R_trt_i, R_at_i = [], [], [], [], []
		k_1t_i, R_wt_i, R_tt_i, P_bt_i = [], [], [], []
		cb_i, cm_i, cw_i, cp_i = [], [], [], []
		for j in V_pr:
			R_f, R_app, k_1, R_b, R_tr, R_a, R_w, R_t, P_b, \
					V_lim_1, V_lim_2, cb, cm, cw, cp = \
					run_calculations(i[0], i[1], i[2],
										50, 8,
										x_results=1, speed=j, sailing=0)
			R_ft_i.append(round(R_f * k_1 / 1000, 2))
			R_appt_i.append(round(R_app / 1000, 2))
			k_1t_i.append(round(k_1, 2))
			R_bt_i.append(round(R_b / 1000, 2))
			R_trt_i.append(round(R_tr / 1000, 2))
			R_at_i.append(round(R_a / 1000, 2))
			R_wt_i.append(round(R_w / 1000, 2))
			R_tt_i.append(round(R_t / 1000, 2))
			P_bt_i.append(round(P_b / 1000, 2))
			cb_i.append(round(cb, 2))
			cm_i.append(round(cm, 2))
			cw_i.append(round(cw, 2))
			cp_i.append(round(cp, 2))

		if plots == 1:
			plt.figure(figs)
			plt.plot(
				V_pr,
				R_ft_i,
				label=f"{k}",
				)
			plt.xlabel("V (m/s)", fontsize=12)
			plt.ylabel("R_f(1 + k1) (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Frictional Resistance", fontsize=16)
		elif plots == 2:
			plt.figure(figs)
			plt.plot(
				V_pr,
				R_appt_i,
				label=f"{k}",
				)
			plt.xlabel("V (m/s)", fontsize=12)
			plt.ylabel("R_app (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Appendage Resistance", fontsize=16)
		elif plots == 3:
			plt.figure(figs)
			plt.plot(
				V_pr,
				k_1t_i,
				label=f"{k}",
				)
			plt.xlabel("V (m/s)", fontsize=12)
			plt.ylabel("1 + k1", fontsize=12)
			plt.axis([3.5, 10.5, 1, 1.5])
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Viscous Form Factor", fontsize=16)
		elif plots == 4:
			plt.figure(figs)
			plt.plot(
				V_pr,
				R_bt_i,
				label=f"{k}",
				)
			plt.xlabel("V (m/s)", fontsize=12)
			plt.ylabel("R_b (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Bulbous Bow Resistance", fontsize=16)
		elif plots == 5:
			plt.figure(figs)
			plt.plot(
				V_pr,
				R_trt_i,
				label=f"{k}",
				)
			plt.xlabel("V (m/s)", fontsize=12)
			plt.ylabel("R_tr (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Transom Immersion Resistance", fontsize=16)
		elif plots == 6:
			plt.figure(figs)
			plt.plot(
				V_pr,
				R_at_i,
				label=f"{k}",
				)
			plt.xlabel("V (m/s)", fontsize=12)
			plt.ylabel("R_a (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Model Correlation Resistance", fontsize=16)
		elif plots == 7:
			plt.figure(figs)
			plt.plot(
				V_pr,
				R_wt_i,
				label=f"{k}",
				)
			plt.xlabel("V (m/s)", fontsize=12)
			plt.ylabel("R_f (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Wave Making Resistance", fontsize=16)
		elif plots == 8:
			plt.figure(figs)
			plt.plot(
				V_pr,
				R_tt_i,
				label=f"{k}",
				)
			plt.xlabel("V (m/s)", fontsize=12)
			plt.ylabel("R_t (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Total Resistance", fontsize=16)
		elif plots == 9:
			plt.figure(figs)
			plt.plot(
				V_pr,
				P_bt_i,
				label=f"{k}",
				)
			plt.xlabel("V (m/s)", fontsize=12)
			plt.ylabel("P (kW)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Brake Power", fontsize=16)

		elif plots == 10:
			plt.figure(figs)
			plt.plot(
				V_pr,
				cb_i,
				label=f"{k}",
				)
			plt.xlabel("V (m/s)", fontsize=12)
			plt.ylabel("Cb", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Block Coefficient", fontsize=16)

		elif plots == 11:
			plt.figure(figs)
			plt.plot(
				V_pr,
				cm_i,
				label=f"{k}",
				)
			plt.xlabel("V (m/s)", fontsize=12)
			plt.ylabel("Cm", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Midship Coefficient", fontsize=16)

		elif plots == 12:
			plt.figure(figs)
			plt.plot(
				V_pr,
				cw_i,
				label=f"{k}",
				)
			plt.xlabel("V (m/s)", fontsize=12)
			plt.ylabel("Cw", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Waterplane Coefficient", fontsize=16)

		elif plots == 13:
			plt.figure(figs)
			plt.plot(
				V_pr,
				cp_i,
				label=f"{k}",
				)
			plt.xlabel("V (m/s)", fontsize=12)
			plt.ylabel("Cp", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Prismatic Coefficient", fontsize=16)


	# Plot and adjust the graph
	plt.rcParams["legend.loc"] = 'best'
	plt.rcParams["legend.shadow"] = True
	plt.grid(which='both', axis='both')
	plt.legend()

# plot_results_x_ship_size(1, 1)
# plot_results_x_ship_size(2, 2)
# plot_results_x_ship_size(3, 3)
# plot_results_x_ship_size(4, 4)
# plot_results_x_ship_size(5, 5)
# plot_results_x_ship_size(6, 6)
# plot_results_x_ship_size(7, 7)
# plot_results_x_ship_size(8, 8)
# plot_results_x_ship_size(9, 9)
plot_results_x_ship_size(10, 10)
# plot_results_x_ship_size(11, 11)
# plot_results_x_ship_size(12, 12)
# plot_results_x_ship_size(13, 13)

plt.show()
