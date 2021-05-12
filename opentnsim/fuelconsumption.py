"""Resistance formulations"""

class FuelConsumption:
    """Mixin class: Something that consumes energy.

    P_installed: installed engine power [kW]
    C_b: block coefficient ('fullness') [-]
    nu: kinematic viscosity [m^2/s]
    rho: density of the surrounding water [kg/m^3]
    g: gravitational accelleration [m/s^2]
    x: number of propellors [-]
    eta_0: open water efficiency of propellor [-]
    eta_r: relative rotative efficiency [-]
    eta_t: transmission efficiency [-]
    eta_g: gearing efficiency [-]
    c_stern: determines shape of the afterbody [-]
    one_k2: appendage resistance factor [-]
    """

    
    def __init__(
        self,
        P_installed,
        *args, 
        **kwargs
        ):
        super().__init__(*args, **kwargs)

        """Initialization"""
        self.P_installed = P_installed
        # Scales the tank volume by the installed engine power
        # Starting ship is M8 with volume 58 m3 and P_installed = 1750 kW
        self.volume = 58 / 1750 * self.P_installed

    def energy_carrier_select(self):
        """Defines the different energy carriers and uses their attributes."""
        if self.ec == 'diesel':
            # Energy carrier is diesel
            # Gross energy density from MJ/L to kWh/L
            # Efficiency is product of serie efficiencies in the fuel process,
            #  energy conversion, pre-treatment, after_treatment, power system
            self.energy_density = 31 / 3.6
            self.average_efficiency = 0.38 * 1.00 * 0.99 * 0.88
        elif self.ec == 'CNG':
            # Energy carrier is compressed natural gas
            self.energy_density = 8.5 / 3.6
        elif self.ec == 'LNG':
            # Energy carrier is LNG
            self.energy_density = 15 / 3.6
        elif self.ec == 'methanolICE':
            # Energy carrier is methanol internal combustion engine
            self.energy_density = 14 / 3.6
            self.average_efficiency = 0.38 * 1.00 * 1.00 * 0.90
        elif self.ec == 'methanolfc':
            # Energy carrier is methanol fuel cell
            self.energy_density = 14 / 3.6
            self.average_efficiency = 0.46 * 0.86 * 1.00 * 0.97
        elif self.ec == 'methylether':
            # Energy carrier is methylether
            self.energy_density = 13.5 / 3.6
        elif self.ec == 'ammonia':
            # Energy carrier is ammonia
            self.energy_density = 9.6 / 3.6
            self.average_efficiency = 0.38 * 0.92 * 1.00 * 0.90
        elif self.ec == 'hydrogenc':
            # Energy carrier is compressed hydrogen
            self.energy_density = 3.8 / 3.6
            self.average_efficiency = 0.46 * 0.98 * 1.00 * 0.97
        elif self.ec == 'hydrogenl':
            # Energy carrier is liquid hydrogen
            self.energy_density = 5 / 3.6
            self.average_efficiency = 0.46 * 0.92 * 1.00 * 0.97
        elif self.ec == 'LOHC':
            # Energy carrier is liquid organic hydrogen carrier
            self.energy_density = 5.5 / 3.6
        elif self.ec == 'NaBH':
            # Energy carrier is natriumborohydride
            self.energy_density = 15.3 / 3.6
        elif self.ec == 'linmcbat':
            # Energy carrier is lithium nickel mangaan cobalt battery
            self.energy_density = 0.4 / 3.6
            self.average_efficiency = 0.94 * 1.00 * 1.00 * 0.97

    # Attempt to create an efficiency curve to account 
    # for vessels with different speeds, or the same vessel with different speeds
    

    # def efficiency_curve(self, average_efficiency):
    #     """Formulates the efficiency dependency on the sailing speed."""
    #     if self.speed < 0.3:
    #         n_i = 1.8 * (0.3 * self.speed)**2
    #         if n_i < 0:
    #             n = 0
    #         else:
    #             n = n_i
    #     elif self.speed < 1.0:
    #         n_i = 2.5 * (0.3 * self.speed)**2
    #         if n_i < 0:
    #             n = 0
    #         else:
    #             n = n_i
    #     elif self.speed < 1.4:
    #         n_i = 0.505 * (3.5 * self.speed)**2 - 2 * self.speed - 4
    #         if n_i < 0:
    #             n = 0
    #         else:
    #             n = n_i
    #     else:
    #         n_i = average_efficiency * 150 - 2.5 * self.speed - 2.1 \
    #             * ((self.speed - 5.671))**2 + 1 * (self.speed)**3 \
    #             - 0.18 * (self.speed + 0.25)**4 - 4
    #         if n_i < 0:
    #             n = 0
    #         else:
    #             n = n_i
    #     return n / 100

    def energy_amount(self, energy_density):
        """Returns the stored amount of energy in kWh."""
        self.starting_energy = self.volume * energy_density

    def energy_consumption(self):
        """Determines the fuel consumption per time step."""
        # E_consumption = []
        # for t_i in t:
        #     # Distance in km
        #     x_i = t_i * V_s * 3.6
        #     x.append(x_i)
        #     # Energy demand
        #     E_d = P_b * t_i * 3600
        #     E_demand.append(E_d)
        #     # Calculate the energy consumption
        #     E_c = E_d / n_e
        #     E_consumption.append(E_c / 3600000)
        #     # Bunker Level
        #     E_b = E_0 - E_c / 3600000
        #     E_bunker.append(E_b)

    def bunker_level(self):
        """Tracks the bunker contents."""
        # E_bunker = []
        # for t_i in t:
        #     E_b = E_0 - E_c / 3600000
        #     E_bunker.append(E_b)
