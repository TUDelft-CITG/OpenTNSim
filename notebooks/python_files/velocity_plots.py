import matplotlib.pyplot as plt
import numpy as np

from calc_e_consumption import run_calculations
from sailing_speed import get_vessel_speed


def plot_results_x_velocity(plots, figs):
	"""
	Select the data which needs to be plotted in which figure.
	plots has options for values 1-xx, each returning a different term.
	For figure, you can chose which figure number you want which plot.
	Velocity on the x-axis.
	"""
	# Creates the values for the sailing speed.
	V_pr = get_vessel_speed(1)

	# Create h profile
	h_pr = [10]

	for i in h_pr:
		R_ft_i = []
		R_appt_i = []
		k_1t_i = []
		R_bt_i = []
		R_trt_i = []
		R_at_i = []
		R_wt_i = []
		R_tt_i = []
		P_bt_i = []
		cb_i = []
		cm_i = []
		cw_i = []
		cp_i = []
		for j in V_pr:
			R_f, R_app, k_1, R_b, R_tr, R_a, R_w, R_t, P_b,\
				V_lim_1, V_lim_2, cb, cm, cw, cp = \
				run_calculations(11.75, 135, 2.75,
									50, i, 
									x_results=1, speed=j, sailing=1)
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

			
			V_lim = round(min(V_lim_1, V_lim_2), 2)
			if V_lim == round(V_lim_1, 2):
				lim_name = 'V_Schijf'
			elif V_lim == round(V_lim_2, 2):
				lim_name = "V_Length"
			Fr_lim = round(V_lim / (np.sqrt(9.81) * i), 2)
		
		if plots == 1:
			plt.figure(figs)
			plt.plot(
				V_pr * 3.6,
				R_ft_i,
				label=f"h = {i} m",
				)
			plt.xlabel("V (km/h)", fontsize=12)
			plt.ylabel("R_f(1 + k1) (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Frictional Resistance, M8", fontsize=16)
		elif plots == 2:
			plt.figure(figs)
			plt.plot(
				V_pr * 3.6,
				R_appt_i,
				label=f"h = {i} m",
				)
			plt.xlabel("V (km/h)", fontsize=12)
			plt.ylabel("R_app (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Appendage Resistance, M8", fontsize=16)
		elif plots == 3:
			plt.figure(figs)
			plt.plot(
				V_pr * 3.6,
				k_1t_i,
				label=f"h = {i} m",
				)
			plt.xlabel("V (km/h)", fontsize=12)
			plt.ylabel("1 + k1", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Viscous Form Factor, M8", fontsize=16)
		elif plots == 4:
			plt.figure(figs)
			plt.plot(
				V_pr * 3.6,
				R_bt_i,
				label=f"h = {i} m",
				)
			plt.xlabel("V (km/h)", fontsize=12)
			plt.ylabel("R_b (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Bulbous Bow Resistance, M8", fontsize=16)
		elif plots == 5:
			plt.figure(figs)
			plt.plot(
				V_pr * 3.6,
				R_trt_i,
				label=f"h = {i} m",
				)
			plt.xlabel("V (km/h)", fontsize=12)
			plt.ylabel("R_tr (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Transom Immersion Resistance, M8", fontsize=16)
		elif plots == 6:
			plt.figure(figs)
			plt.plot(
				V_pr * 3.6,
				R_at_i,
				label=f"h = {i} m",
				)
			plt.xlabel("V (km/h)", fontsize=12)
			plt.ylabel("R_a (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Model Correlation Resistance, M8", fontsize=16)
		elif plots == 7:
			plt.figure(figs)
			plt.plot(
				V_pr * 3.6,
				R_wt_i,
				label=f"h = {i} m",
				)
			plt.xlabel("V (km/h)", fontsize=12)
			plt.ylabel("R_f (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Wave Making Resistance, M8", fontsize=16)
		elif plots == 8:
			plt.figure(figs)
			plt.plot(
				V_pr * 3.6,
				R_tt_i,
				label=f"h = {i} m",
				)
			plt.xlabel("V (km/h)", fontsize=12)
			plt.ylabel("R_t (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Total Resistance, M8", fontsize=16)
		elif plots == 9:
			plt.figure(figs)
			plt.plot(
				V_pr * 3.6,
				P_bt_i,
				label=f"h = {i} m, {lim_name} = {V_lim} m/s, Fr_h = {Fr_lim}",
				)
			plt.xlabel("V (km/h)", fontsize=12)
			plt.ylabel("P (kW)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Brake Power, M8", fontsize=16)

		elif plots == 10:
			plt.figure(figs)
			plt.plot(
				V_pr * 3.6,
				cb_i,
				label=f"h = {i} m",
				)
			plt.xlabel("V (km/h)", fontsize=12)
			plt.ylabel("Cb", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Block Coefficient, M8", fontsize=16)

		elif plots == 11:
			plt.figure(figs)
			plt.plot(
				V_pr * 3.6,
				cm_i,
				label=f"h = {i} m",
				)
			plt.xlabel("V (km/h)", fontsize=12)
			plt.ylabel("Cm", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Midship Coefficient, M8", fontsize=16)

		elif plots == 12:
			plt.figure(figs)
			plt.plot(
				V_pr * 3.6,
				cw_i,
				label=f"h = {i} m",
				)
			plt.xlabel("V (km/h)", fontsize=12)
			plt.ylabel("Cw", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Waterplane Coefficien, M8", fontsize=16)

		elif plots == 13:
			plt.figure(figs)
			plt.plot(
				V_pr * 3.6,
				cp_i,
				label=f"h = {i} m",
				)
			plt.xlabel("V (km/h)", fontsize=12)
			plt.ylabel("Cp", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Prismatic Coefficient, M8", fontsize=16)


	# Plot and adjust the graph
	plt.rcParams["legend.loc"] = 'best'
	plt.rcParams["legend.shadow"] = True
	plt.grid(which='both', axis='both')
	plt.legend()


plot_results_x_velocity(1, 1)
plot_results_x_velocity(2, 2)
plot_results_x_velocity(3, 3)
plot_results_x_velocity(4, 4)
plot_results_x_velocity(5, 5)
plot_results_x_velocity(6, 6)
plot_results_x_velocity(7, 7)
plot_results_x_velocity(8, 8)
plot_results_x_velocity(9, 9)
# #plot_results_x_velocity(10, 10)
# plot_results_x_velocity(11, 11)
# plot_results_x_velocity(12, 12)
# plot_results_x_velocity(13, 13)


plt.show()

# fig, ax = plt.subplots()
# # fig, axs = plt.subplots(3, 3, sharex=True)
# fig.tight_layout()

# axs[0, 0].plot(V_pr, R_ft, linewidth=3)
# axs[0, 0].set_title("Skin Friction Resistance", fontsize=16)
# axs[0, 0].set_xlabel("V in m/s", fontsize=12)
# axs[0, 0].set_ylabel("R_f in N", fontsize=12)
# axs[0, 0].tick_params(axis='both', labelsize=12)
# axs[0, 0].axvline(x = min(V_lim_1, V_lim_2), color='red')

# axs[1, 0].plot(V_pr, R_appt, linewidth=3)
# axs[1, 0].set_title("Appendage Resistance", fontsize=16)
# axs[1, 0].set_xlabel("V in m/s", fontsize=12)
# axs[1, 0].set_ylabel("R_app in N", fontsize=12)
# axs[1, 0].tick_params(axis='both', labelsize=12)
# axs[1, 0].axvline(x = min(V_lim_1, V_lim_2), color='red')

# axs[2, 0].plot(V_pr, k_1t, linewidth=3)
# axs[2, 0].set_title("Viscous Form Resistance Factor", fontsize=16)
# axs[2, 0].set_xlabel("V in m/s", fontsize=12)
# axs[2, 0].set_ylabel("1 + k1", fontsize=12)
# axs[2, 0].tick_params(axis='both', labelsize=12)
# axs[2, 0].axvline(x = min(V_lim_1, V_lim_2), color='red')

# axs[0, 1].plot(V_pr, R_bt, linewidth=3)
# axs[0, 1].set_title("Bulbous Bow Wave Making Resistance", fontsize=16)
# axs[0, 1].set_xlabel("V in m/s", fontsize=12)
# axs[0, 1].set_ylabel("R_b in N", fontsize=12)
# axs[0, 1].tick_params(axis='both', labelsize=12)
# axs[0, 1].axvline(x = min(V_lim_1, V_lim_2), color='red')

# axs[1, 1].plot(V_pr, R_trt, linewidth=3)
# axs[1, 1].set_title("Transom Immersion Resistance", fontsize=16)
# axs[1, 1].set_xlabel("V in m/s", fontsize=12)
# axs[1, 1].set_ylabel("R_tr in N", fontsize=12)
# axs[1, 1].tick_params(axis='both', labelsize=12)
# axs[1, 1].axvline(x = min(V_lim_1, V_lim_2), color='red')

# axs[2, 1].plot(V_pr, R_at, linewidth=3)
# axs[2, 1].set_title("Model Ship Correlation Resistance", fontsize=16)
# axs[2, 1].set_xlabel("V in m/s", fontsize=12)
# axs[2, 1].set_ylabel("R_a in N", fontsize=12)
# axs[2, 1].tick_params(axis='both', labelsize=12)
# axs[2, 1].axvline(x = min(V_lim_1, V_lim_2), color='red')

# axs[0, 2].plot(V_pr, R_wt, linewidth=3)
# axs[0, 2].set_title("Wave Making Resistance", fontsize=16)
# axs[0, 2].set_xlabel("V in m/s", fontsize=12)
# axs[0, 2].set_ylabel("R_w in N", fontsize=12)
# axs[0, 2].tick_params(axis='both', labelsize=12)
# axs[0, 2].axvline(x = min(V_lim_1, V_lim_2), color='red')

# axs[1, 2].plot(V_pr, R_tt, linewidth=3)
# axs[1, 2].set_title("Total Resistance", fontsize=16)
# axs[1, 2].set_xlabel("V in m/s", fontsize=12)
# axs[1, 2].set_ylabel("R_t in N", fontsize=12)
# axs[1, 2].tick_params(axis='both', labelsize=12)
# axs[1, 2].axvline(x = min(V_lim_1, V_lim_2), color='red')

# ax.plot(V_pr, P_bt, linewidth=3)
# ax.set_title("Brake Power", fontsize=16)
# ax.set_xlabel("V in m/s", fontsize=12)
# ax.set_ylabel("P_b in W", fontsize=12)
# ax.tick_params(axis='both', labelsize=12)
# ax.axvline(x = min(V_lim_1, V_lim_2), color='red')
