"""
Logging utilities for energy-related calculations.
"""

# %% IMPORT DEPENDENCIES
# generic


# internal
import opentnsim.graph.mixins as graph_module


# %% ADD ENERGY ATTRIBUTES INTO EVENT TABLE
def add_energy_attributes_to_eventtable(df, objs):
    """
    Add energy-related attributes to a 'minimum event table'.

    When a vessel object is created with the ConsumesEnergy and VesselProperties
    mixins, and the path along the graph contains Info - Generaldepth
    information, the energy module can calculate:
     - resistance components for passing an edge at a given speed,
     - power needed to overcome that resistance,
     - energy consumed to deliver that power.

    This method takes a 'minimum event table' and a list of objs as input, and
    returns an event table with resistance, power, and energy attributes.

    References
    ----------
    Van der Werff, S.E., F. Baart and M. van Koningsveld (2025). “Merging
    Multiple System Perspectives: The Key to Effective Inland Shipping
    Emission-Reduction Policy Design.” Journal of Marine Science and Engineering
    13(4), 716. https://doi.org/10.3390/jmse13040716

    Van der Werff, S.E., S. Eppenga, A. van der Hout, F. Baart and M. van
    Koningsveld (2025). “Multi-perspective nautical safety risk assessment of
    allisions with offshore wind parks.” Applied Ocean Research 158(2025),104564.
    https://doi.org/10.1016/j.apor.2025.104564

    For waterborne traffic over a network, a unique event is defined by:
     - a unique vessel,
     - a specific section of the waterway,
     - a specific time.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame with elements of a 'minimum event table'.
    objs: list
        List of OpenTNSim simulation objects with log information.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame with energy-related attributes added.
    """

    for index, row in df.iterrows():

        # the generator option (next) is used to make sure we get the full copy of
        # the object found in the list
        obj = next((x for x in objs if x.id == row["object id"]), None)

        # get the depth from the edge sailed in the event (and check if squat effects
        # need to be considered
        h_0 = graph_module.calculate_depth(row["start location"], row["stop location"], obj.env.graph)
        h_0 = obj.calculate_h_squat(
            v=obj.v, h_0=h_0
        )  # TODO: actually takes width as arg

        obj.calculate_total_resistance(v=obj.v, h_0=h_0)
        obj.calculate_total_power_required(v=obj.v, h_0=h_0)
        obj.calculate_emission_factors_total(v=obj.v, h_0=h_0)
        obj.calculate_SFC_final(v=obj.v, h_0=h_0)

        df.at[index, "waterdepth (m)"] = h_0
        df.at[index, "waterway width (m)"] = None
        df.at[index, "current (m/s)"] = 0  # TODO: get current from graph
        df.at[index, "engine age (year)"] = obj.C_year

        df.at[index, "P_tot (kW)"] = obj.P_tot
        df.at[index, "P_given (kW)"] = obj.P_given
        df.at[index, "P_installed (kW)"] = obj.P_installed

        energy_delta = obj.P_tot * row["duration (s)"] / 3600  # kJ/3600 = kWh
        df.at[index, "total_energy (kWh)"] = energy_delta

    return df


# %% ADD FUEL ATTRIBUTES INTO EVENT TABLE
def add_fuel_attributes_to_event_table(df, objs):
    """
    Method to add fuel-related attributes to a 'minimum event table' plus
    energy attributes.

    When we have an event table that includes energy attributes, we can add
    fuel-related attributes. Depending on the energy carrier and the energy
    converter, we can estimate:
     - fuel consumption
     - emissions

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame that contains the elements of a 'minimum event table' plus
        energy-related attributes.
    objs: list
        List of OpenTNSim simulation objects that have log information.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame that contains a 'minimum event table' plus energy-related and
        fuel-related attributes.
    """
    # check if total_energy (kWh) is in the DataFrame
    if "total_energy (kWh)" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'total_energy (kWh)' column to add fuel "
            "attributes. Make sure to call add_energy_attributes_to_eventtable first."
        )

    for index, row in df.iterrows():

        # the generator option (next) is used to make sure we get the full copy of the object found in the list
        obj = next((x for x in objs if x.id == row["object id"]), None)

        diesel_C_year = row["total_energy (kWh)"] * obj.final_SFC_diesel_C_year_ICE_mass

        df.at[index, "diesel_consumption (g)"] = diesel_C_year
        df.at[index, "diesel_consumption_m (g/m)"] = diesel_C_year / row["distance (m)"]
        df.at[index, "diesel_consumption_s (g/s)"] = diesel_C_year / row["duration (s)"]

        CO2_emission_total = (
            row["total_energy (kWh)"] * obj.total_factor_CO2
        )  # in g (total_factor is the Emission Factor)                                                                                              #stationary phase # in g
        PM10_emission_total = row["total_energy (kWh)"] * obj.total_factor_PM10  # in g
        NOX_emission_total = row["total_energy (kWh)"] * obj.total_factor_NOX  # in g

        df.at[index, "CO2_emission_total (g)"] = CO2_emission_total
        df.at[index, "PM10_emission_total (g)"] = PM10_emission_total
        df.at[index, "NOX_emission_total (g)"] = NOX_emission_total

        # TODO: see if these entries are useful. They can also be easily calculated
        # from other entries
        df.at[index, "CO2_emission_per_m (g/m)"] = (
            CO2_emission_total / row["distance (m)"]
        )
        df.at[index, "PM10_emission_per_m (g/m)"] = (
            PM10_emission_total / row["distance (m)"]
        )
        df.at[index, "NOX_emission_per_m (g/m)"] = (
            NOX_emission_total / row["distance (m)"]
        )

        # TODO: see if these entries are useful. They can also be easily calculated
        # from other entries
        df.at[index, "CO2_emission_per_s (g/s)"] = (
            CO2_emission_total / row["duration (s)"]
        )
        df.at[index, "PM10_emission_per_s (g/s)"] = (
            PM10_emission_total / row["duration (s)"]
        )
        df.at[index, "NOX_emission_per_s (g/s)"] = (
            NOX_emission_total / row["duration (s)"]
        )

    return df
