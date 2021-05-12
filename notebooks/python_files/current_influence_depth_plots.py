import matplotlib.pyplot as plt
import numpy as np

from calc_e_consumption import run_calculations


def plot_results_x_return_current(plots, figs):
	"""
	Select the data which needs to be plotted in which figure.
	plots has options for values 1-xx, each returning a different term.
	For figure, you can chose which figure number you want which plot.
	Velocity on the x-axis.
	"""
	# Creates the values for the current.
	U_pr = np.linspace(-2, 3, 100)

	# Create h profile
	h_pr = list(range(4, 10)) 

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
		for j in U_pr:
			R_f, R_app, k_1, R_b, R_tr, R_a, R_w, R_t, P_b, \
				V_lim_1, V_lim_2, cb, cm, cw, cp = \
				run_calculations(11.40, 110, 3.50,
									50, i, 
									x_results=1, speed=4, sailing=0, current=j)
			R_ft_i.append(round(R_f * k_1 / 1000, 2))
			R_appt_i.append(round(R_app / 1000, 2))
			k_1t_i.append(round(k_1, 2))
			R_bt_i.append(round(R_b / 1000, 2))
			R_trt_i.append(round(R_tr / 1000, 2))
			R_at_i.append(round(R_a / 1000, 2))
			R_wt_i.append(round(R_w / 1000, 2))
			R_tt_i.append(round(R_t / 1000000, 2))
			P_bt_i.append(round(P_b / 1000000, 2))
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
				U_pr,
				R_ft_i,
				label=f"h = {i} m, V = 4 m/s",
				)
			plt.xlabel("U (m/s)", fontsize=12)
			plt.ylabel("R_f(1 + k1) (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Frictional Resistance vs. Current, M8", fontsize=16)
		elif plots == 2:
			plt.figure(figs)
			plt.plot(
				U_pr,
				R_appt_i,
				label=f"h = {i} m, V = 4 m/s",
				)
			plt.xlabel("U (m/s)", fontsize=12)
			plt.ylabel("R_app (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Appendage Resistance vs. Current, M8", fontsize=16)
		elif plots == 3:
			plt.figure(figs)
			plt.plot(
				U_pr,
				k_1t_i,
				label=f"h = {i} m, V = 4 m/s",
				)
			plt.xlabel("U (m/s)", fontsize=12)
			plt.ylabel("1 + k1", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Viscous Form Factor vs. Current, M8", fontsize=16)
		elif plots == 4:
			plt.figure(figs)
			plt.plot(
				U_pr,
				R_bt_i,
				label=f"h = {i} m, V = 4 m/s",
				)
			plt.xlabel("U (m/s)", fontsize=12)
			plt.ylabel("R_b (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Bulbous Bow Resistance vs. Current, M8", fontsize=16)
		elif plots == 5:
			plt.figure(figs)
			plt.plot(
				U_pr,
				R_trt_i,
				label=f"h = {i} m, V = 4 m/s",
				)
			plt.xlabel("U (m/s)", fontsize=12)
			plt.ylabel("R_tr (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Transom Immersion Resistance vs. Current, M8", fontsize=16)
		elif plots == 6:
			plt.figure(figs)
			plt.plot(
				U_pr,
				R_at_i,
				label=f"h = {i} m, V = 4 m/s",
				)
			plt.xlabel("U (m/s)", fontsize=12)
			plt.ylabel("R_a (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Model Correlation Resistance vs. Current, M8", fontsize=16)
		elif plots == 7:
			plt.figure(figs)
			plt.plot(
				U_pr,
				R_wt_i,
				label=f"h = {i} m, V = 4 m/s",
				)
			plt.xlabel("U (m/s)", fontsize=12)
			plt.ylabel("R_f (kN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Wave Making Resistance vs. Current, M8", fontsize=16)
		elif plots == 8:
			plt.figure(figs)
			plt.plot(
				U_pr,
				R_tt_i,
				label=f"h = {i} m, V = 4 m/s",
				)
			plt.xlabel("U (m/s)", fontsize=12)
			plt.ylabel("R_t (MN)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Total Resistance vs. Current, M8", fontsize=16)
		elif plots == 9:
			plt.figure(figs)
			plt.plot(
				U_pr,
				P_bt_i,
				label=f"h = {i} m, {lim_name} = {V_lim} m/s, Fr_h = {Fr_lim}",
				)
			plt.xlabel("U (m/s)", fontsize=12)
			plt.ylabel("P (MW)", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Brake Power vs. Current, M8", fontsize=16)

		elif plots == 10:
			plt.figure(figs)
			plt.plot(
				U_pr,
				cb_i,
				label=f"h = {i} m, V = 4 m/s",
				)
			plt.xlabel("U (m/s)", fontsize=12)
			plt.ylabel("Cb", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Block Coefficient vs. Current, M8", fontsize=16)

		elif plots == 11:
			plt.figure(figs)
			plt.plot(
				U_pr,
				cm_i,
				label=f"h = {i} m, V = 4 m/s",
				)
			plt.xlabel("U (m/s)", fontsize=12)
			plt.ylabel("Cm", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Midship Coefficient vs. Current, M8", fontsize=16)

		elif plots == 12:
			plt.figure(figs)
			plt.plot(
				U_pr,
				cw_i,
				label=f"h = {i} m, V = 4 m/s",
				)
			plt.xlabel("U (m/s)", fontsize=12)
			plt.ylabel("Cw", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Waterplane Coefficien vs. Current, M8", fontsize=16)

		elif plots == 13:
			plt.figure(figs)
			plt.plot(
				U_pr,
				cp_i,
				label=f"h = {i} m, V = 4 m/s",
				)
			plt.xlabel("U (m/s)", fontsize=12)
			plt.ylabel("Cp", fontsize=12)
			plt.tick_params(axis='both', labelsize=12)
			plt.title("Prismatic Coefficient vs. Current, M8", fontsize=16)


	# Plot and adjust the graph
	plt.rcParams["legend.loc"] = 'best'
	plt.rcParams["legend.shadow"] = True
	plt.grid(which='both', axis='both')
	plt.legend()


# plot_results_x_return_current(1, 1)
# plot_results_x_return_current(2, 2)
# plot_results_x_return_current(3, 3)
# plot_results_x_return_current(4, 4)
# plot_results_x_return_current(5, 5)
# plot_results_x_return_current(6, 6)
# plot_results_x_return_current(7, 7)
# plot_results_x_return_current(8, 8)
# plot_results_x_return_current(9, 9)
plot_results_x_return_current(10, 10)
# plot_results_x_return_current(11, 11)
# plot_results_x_return_current(12, 12)
# plot_results_x_return_current(13, 13)

plt.show()
