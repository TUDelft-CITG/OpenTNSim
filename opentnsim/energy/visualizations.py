import matplotlib.pyplot as plt
import numpy as np

def create_energy_use_plot(vessel, testing=False):
    """Create a plot of the energy use of a vessel

    Parameters
    ----------
    vessel : object
        A vessel object to plot. vessel need to have a logbook (mixin class Log)
    testing : bool
        If True, the plot will not be shown. This is useful for testing purposes.
    """
    energy_use_loading = 0  # concumption between loading start and loading stop
    energy_use_sailing_full = 0  # concumption between sailing full start and sailing full stop
    energy_use_unloading = 0  # concumption between unloading  start and unloading  stop
    energy_use_sailing_empty = 0  # concumption between sailing empty start and sailing empty stop
    energy_use_waiting = 0  # concumption between waiting start and waiting stop

    for i in range(len(vessel.log["Message"])):
        if vessel.log["Message"][i] == "Energy use loading":
            energy_use_loading += vessel.log["Value"][i]

        elif vessel.log["Message"][i] == "Energy use sailing full":
            energy_use_sailing_full += vessel.log["Value"][i]

        elif vessel.log["Message"][i] == "Energy use unloading":
            energy_use_unloading += vessel.log["Value"][i]

        elif vessel.log["Message"][i] == "Energy use sailing empty":
            energy_use_sailing_empty += vessel.log["Value"][i]

        elif vessel.log["Message"][i] == "Energy use waiting":
            energy_use_waiting += vessel.log["Value"][i]

    # For the total plot
    fig, ax1 = plt.subplots(figsize=[15, 10])

    # For the barchart
    height = [
        energy_use_loading,
        energy_use_unloading,
        energy_use_sailing_full,
        energy_use_sailing_empty,
        energy_use_waiting,
    ]
    labels = ["Loading", "Unloading", "Sailing full", "Sailing empty", "Waiting"]
    colors = [
        (55 / 255, 126 / 255, 184 / 255),
        (98 / 255, 192 / 255, 122 / 255),
        (255 / 255, 150 / 255, 0 / 255),
        (98 / 255, 141 / 255, 122 / 255),
        (124 / 255, 10 / 255, 2 / 255),
    ]

    positions = np.arange(len(labels))
    ax1.bar(positions, height, color=colors)

    # For the cumulative percentages
    total_use = sum(
        [
            energy_use_loading,
            energy_use_unloading,
            energy_use_sailing_full,
            energy_use_sailing_empty,
            energy_use_waiting,
        ]
    )

    energy_use_unloading += energy_use_loading
    energy_use_sailing_full += energy_use_unloading
    energy_use_sailing_empty += energy_use_sailing_full
    energy_use_waiting += energy_use_sailing_empty
    y = [
        energy_use_loading,
        energy_use_unloading,
        energy_use_sailing_full,
        energy_use_sailing_empty,
        energy_use_waiting,
    ]
    n = [
        energy_use_loading / total_use,
        energy_use_unloading / total_use,
        energy_use_sailing_full / total_use,
        energy_use_sailing_empty / total_use,
        energy_use_waiting / total_use,
    ]

    ax1.plot(positions, y, "ko", markersize=10)
    ax1.plot(positions, y, "k")

    for i, txt in enumerate(n):
        x_txt = positions[i] + 0.1
        y_txt = y[i] * 0.95
        ax1.annotate("{:02.1f}%".format(txt * 100), (x_txt, y_txt), size=12)

    # Further markup
    plt.ylabel("Energy useage in KWH", size=12)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, size=12)
    plt.title("Energy use - {}".format(vessel.name), size=15)

    if testing is False:
        plt.show()