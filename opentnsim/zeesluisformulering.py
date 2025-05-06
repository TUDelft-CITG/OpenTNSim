import math as ma
import numpy as np

class ZeesluisFormulering():

    def __init__(self,
                 L_lock,                            # L_lock: Length lock chamber [m]
                 B_lock,                            # B_lock: Width lock chamber [m]
                 z_lock,                            # z_lock: Bottom level lock chamber [m]
                 z_A=None,                          # z_A: Bottom level at outer harbour basin [m]
                 z_B=None,                          # z_B: Bottom level at inner harbour basin [m]
                 z_sillA=None,                      # z_sillA: Bottom level of sill at side A (sea) [m]
                 z_sillB=None,                      # z_sillB: Bottom level of sill at side B (inland) [m]
                 h_lock=0,                          # h_lock: Water level of lock chamber [m]
                 h_A=0,                             # h_A: Water level at side A (sea) [m]
                 h_B=0,                             # h_B: Water level at side B (inland) [m]
                 S_lock=30,                         # S_lock: Salinity of lock chamber [m]
                 S_A=30,                            # S_A: Salinity at side A (sea) [kg/m3]
                 S_B=0,                             # Salinity at side B (inland) [kg/m3]
                 T=15,                              # General water temperature [degrees Celsius]
                 T_A=None,                          # Water temperature at side A (sea) [degrees Celsius]
                 T_B=None,                          # Water emperature at side B (inland) [degrees Celsius]
                 Eff_A=1,                           # Eff_A: Efficiency countermeasure for exchange current at side A (sea) [-]
                 Eff_B=1,                           # Eff_B: Efficiency countermeasure for exchange current at side B (inland) [-]
                 Q_flushing_low_tide=0,             # Q_flushing_low_tide: Flushing discharge during low water to side A (sea) [m3/s]
                 Q_flushing_high_tide=0,            # Q_flushing_high_tide: Flushing discharge during high water to side A (sea) [m3/s]
                 t_door_opening=4.5,                # t_door_opening: door opening time
                 t_door_closing=4.5,                # t_door_closing: door closing time
                 t_levelling=500,                   # levelling time [s]
                 t_levelling_A=None,                # t_levelling_A: Levelling time to side A [min]
                 t_levelling_B=None,                # t_levelling_B: Levelling time to side B [min]
                 t_door_open = 1800,                # door open time [s]
                 t_door_open_A=None,
                 t_door_open_B=None,
                 V_ship_B2A=0,                      # V_ship_B2A: Volume of vessels bounded for side A (sea) [m3]
                 V_ship_A2B=0,                      # V_ship_A2B: Volume of vessels bounded for side B (inland) [m3]
                 g=9.81,                            # g: Gravitational constant
                 correction_factor_v_exchange_A = 1,# correction_factor_v_exchange_A: Calibration coefficient exchange velocity A [-]
                 correction_factor_v_exchange_B = 1,# correction_factor_v_exchange_B: Calibration coefficient exchange velocity B [-]
                 D_door_any_bubble_screen_A = 0,    # D_door_any_bubble_screen_A: distance of bubble screen in respect to lock doors at side B (sea) [m]
                 D_door_any_bubble_screen_B = 0,    # D_door_any_bubble_screen_B: distance of bubble screen in respect to lock doors at side B (inland) [m]
                 N_cycles = 1,                     # Number of cycles per day [-]
                 calibration_coefficient = 1.0,
                 symmetry_coefficient = 1.0,
                 t_cycle = None):

        self.L_lock = L_lock
        self.B_lock = B_lock
        self.z_lock = z_lock
        if z_A is not None:
            self.z_A = z_A
        else:
            self.z_A = self.z_lock
        if z_B is not None:
            self.z_B = z_B
        else:
            self.z_B = self.z_lock
        if z_sillA is not None:
            self.z_sillA = z_sillA
        else:
            self.z_sillA = self.z_lock
        if z_sillB is not None:
            self.z_sillB = z_sillB
        else:
            self.z_sillB = self.z_lock
        self.h_A = h_A
        self.h_B = h_B
        self.h_lock = h_lock
        self.S_A = S_A
        self.S_B = S_B
        self.T_A = T
        self.T_B = T
        if T_A is not None:
            self.T_A = T_A
        if T_B is not None:
            self.T_B = T_B
        self.S_lock = S_lock
        self.V_ship_A2B = V_ship_A2B
        self.V_ship_B2A = V_ship_B2A
        self.t_door_opening = t_door_opening
        self.t_door_closing = t_door_closing
        self.Eff_A = Eff_A
        self.Eff_B = Eff_B
        self.g = g
        self.Q_flushing_low_tide = Q_flushing_low_tide
        self.Q_flushing_high_tide = Q_flushing_high_tide
        self.correction_factor_v_exchange_A = correction_factor_v_exchange_A
        self.correction_factor_v_exchange_B = correction_factor_v_exchange_B
        self.D_door_any_bubble_screen_A = D_door_any_bubble_screen_A
        self.D_door_any_bubble_screen_B = D_door_any_bubble_screen_B
        self.N_cycles = N_cycles
        self.calibration_coefficient = calibration_coefficient
        self.symmetry_coefficient = symmetry_coefficient

        self.t_levelling = t_levelling
        self.t_levelling_A = t_levelling_A
        self.t_levelling_B = t_levelling_B
        if t_levelling_A is None:
            self.t_levelling_A = self.t_levelling
        if t_levelling_B is None:
            self.t_levelling_B = self.t_levelling

        if not self.t_levelling:
            self.t_levelling = np.mean([self.t_levelling_A, self.t_levelling_B])

        self.t_door_open = t_door_open
        if not t_door_open:
            self.t_cycle = t_cycle
            if t_cycle is None:
                self.t_cycle = 24.0 * 3600.0 / self.N_cycles
            self.t_door_open_avg = 0.5 * (self.t_cycle - self.t_levelling + 2.0 * 0.5 * self.t_door_opening)
            self.t_door_open = self.calibration_coefficient * self.t_door_open_avg

        self.t_door_open_A = t_door_open_A
        if t_door_open_A is None:
            self.t_door_open_A = (2.0 - self.symmetry_coefficient) * self.t_door_open

        self.t_door_open_B = t_door_open_B
        if t_door_open_B is None:
            self.t_door_open_B = self.symmetry_coefficient * self.t_door_open


    def salinity_psu_to_density(self,S_psu,T=10):
        a = 8.24493*10**-1 - 4.0899*10**-3 * T + 7.6438*10**-5 * T**2 - 8.2467*10**-7 * T**3 + 5.3875*10**-9 * T**4
        b = -5.72466*10**-3 + 1.0227*10**-4 * T - 1.6546*10**-6 * T**2
        c = 4.8314*10**-4
        rho_ref = (999.842594 + 6.793952*10**-2 * T - 9.095290*10**-3 * T**2.0 +
                   1.001685*10**-4 * T**3.0 - 1.120083*10**-6 * T**4 + 6.536332*10**-9 * T**5.0)
        density = rho_ref + a * S_psu + b * S_psu**1.5 + c * S_psu**2.0;
        return density


    def salinity_kgm3_to_density(self,S_kgm3,T=10,rtol=10**-5,atol=10**-8):
        S_psu = S_kgm3
        rho = 1000
        for _ in range(1000):
            rho_new = self.salinity_psu_to_density(S_psu,T)
            S_psu = (S_kgm3 / rho_new)*1000
            max_rho = np.max([rho_new,rho])
            if np.abs(rho_new - rho) <= np.max([rtol*max_rho,atol]):
                return rho_new
            rho = rho_new
        return rho_new


    def has_convergance(self,initial_value,next_value,rtol=10**-5,atol=10**-8):
        absolute_max = np.max([np.abs(initial_value), np.abs(next_value)])
        if np.abs(initial_value - next_value) <= np.max([rtol * absolute_max, atol]):
            return True
        else:
            return False


    def levelling_to_B(self,
                       h_lock=None, #Water level in lock chamber [m]
                       h_B=None,    # h_B: Water level at inland water side [m]
                       S_lock=None, # S_lock: Initial salinity in lock [kg/m3]
                       S_B=None,    # S_B: Salinity at inner side (inland) [kg/m3]
                       V_ship=0,    # V_ship_A2B: Volume of upstream-bounded vessels [m3]
                       t_levelling=None,  # T_niv: Levelling time [s]
                       ):

        if h_lock is None:
            h_lock = self.h_A
        if h_B is None:
            h_B = self.h_B
        if S_lock is None:
            S_lock = self.S_lock
        if S_B is None:
            S_B = self.S_B
        if t_levelling is None:
            t_levelling = self.t_levelling_B

        if not V_ship and self.V_ship_A2B:
            V_ship = self.V_ship_A2B

        V_lock_init = self.L_lock * self.B_lock * (h_lock - self.z_lock)  # lock volume levelled with outer harbour [m3]
        V_lock_B = self.L_lock * self.B_lock * (h_B - self.z_lock)  # Lock volume levelled with inner harbour [m3]

        # Levelling phase to inner harbour during higher water at B
        if h_B >= h_lock:
            V_lock2B_nivB = 0 #Levelling volume [m3]
            V_B2lock_nivB = self.L_lock * self.B_lock * (h_B - h_lock) #Levelling volume [m3]
            M_lock2B_nivB = 0
            M_B2lock_nivB = V_B2lock_nivB * S_B  # Mass flux salt from B to lock chamber [kg]
            S_lock_nivB = (S_lock * (V_lock_init - V_ship) + M_B2lock_nivB) / (V_lock_B - V_ship)  # Salinity in lock chamber after levelling [kg/m3]
            Q_lock2B_nivB = 0
            Q_B2lock_nivB = V_B2lock_nivB / t_levelling  # Average discharge water from outer harbour to lock chamber [m3/s]
            S_lock2B_nivB = 0
            S_B2lock_nivB = S_B  # Average salinity of discharge Q_M1 [m3/s]

        # Levelling phase to inner harbour during lower water at B
        else:
            V_lock2B_nivB = self.L_lock * self.B_lock * (h_lock - h_B) #Levelling volume [m3]
            V_B2lock_nivB = 0 #Levelling volume [m3]
            M_lock2B_nivB = V_lock2B_nivB * S_lock
            M_B2lock_nivB = 0
            S_lock_nivB = S_lock # Salinity in lock chamber after levelling [kg/m3]
            Q_lock2B_nivB = V_lock2B_nivB / t_levelling
            Q_B2lock_nivB = 0
            S_lock2B_nivB = S_lock # Average salinity of discharge Q_M1 [m3/s]
            S_B2lock_nivB = 0

        self.S_lock = S_lock_nivB
        self.h_lock = self.h_B

        output = {'S_lock':self.S_lock,
                  'V_lock2B_nivB': V_lock2B_nivB,
                  'V_B2lock_nivB': V_B2lock_nivB,
                  'M_lock2B_nivB': M_lock2B_nivB,
                  'M_B2lock_nivB': M_B2lock_nivB,
                  'S_lock_nivB': S_lock_nivB,
                  'Q_lock2B_nivB': Q_lock2B_nivB,
                  'Q_B2lock_nivB': Q_B2lock_nivB,
                  'S_lock2B_nivB': S_lock2B_nivB,
                  'S_B2lock_nivB': S_B2lock_nivB}

        return output


    def door_opening_phase_B(self,
                             h_A=None,          # Water level at side A (sea) [m]
                             h_B=None,          # Water level at side B (inland) [m]
                             S_lock=None,       # Initial salinity in lock [kg/m3]
                             S_A=None,
                             S_B=None,          # Salinity at side B (inland) [kg/m3]
                             T_A=None,
                             T_B=None,
                             V_ship_exit=0,          # Volume of exiting vessels (upstream-bounded) [m3]
                             V_ship_entry=0,         # Volume of entering vessels (downstream-bounded) [m3]
                             Eff=None,          # Efficiency countermeasure for exchange current
                             t_door_open=None,       # Representative opening time of door [s]
                             Q_flushing=None,   # Flushing discharge during high water at outer harbour basin [m3/s]
                             correction_factor_v_exchange=None,# calibration factor on propagation velocity salt exchange
                             ):

        if h_A is None:
            h_A = self.h_A
        if h_B is None:
            h_B = self.h_B
        if S_lock is None:
            S_lock = self.S_lock
        if S_A is None:
            S_A = self.S_A
        if S_B is None:
            S_B = self.S_B
        if T_A is None:
            T_A = self.T_A
        if T_B is None:
            T_B = self.T_B
        if Eff is None:
            Eff = self.Eff_B
        if t_door_open is None:
            t_door_open = self.t_door_open_B
        if Q_flushing is None:
            if h_B > h_A:
                Q_flushing = self.Q_flushing_low_tide
            else:
                Q_flushing = self.Q_flushing_high_tide
        if correction_factor_v_exchange is None:
            correction_factor_v_exchange = self.correction_factor_v_exchange_B
        if not V_ship_exit and self.V_ship_A2B:
            V_ship_exit = self.V_ship_A2B
        if not V_ship_entry and self.V_ship_B2A:
            V_ship_entry = self.V_ship_B2A

        H_sillB = h_B - self.z_lock - np.max([0,np.min([(self.z_sillB - self.z_B), (self.z_sillB - self.z_lock)])]) #Head at sill [m]
        V_lock_B = self.L_lock * self.B_lock * (h_B - self.z_lock) #lock volume levelled with inner harbour [m3]
        H_effB = h_B - self.z_lock - 0.8 * np.max([0,np.min([(self.z_sillB - self.z_B), (self.z_sillB - self.z_lock)])]) # Effective depth levelled with inner harbour [m]
        V_lock_B_Eff = self.L_lock * self.B_lock * H_effB  # Effective lock volume levelled with inner harbour [m3]
        rho_MZ = 0.5 * (self.salinity_kgm3_to_density(S_A,T_A)+self.salinity_kgm3_to_density(S_B,T_B)) # Average water density lock complex [kg/m3]

        # vessels exiting lock
        M_B2lock_openBa = V_ship_exit * S_B # Mass flux salt from inner to lock chamber after vessels sailed out [kg]
        M_lock_openBa = abs(S_lock * (V_lock_B - V_ship_exit) + M_B2lock_openBa)
        S_lock_openBa = M_lock_openBa / V_lock_B # Salinity in chamber after ship sails out [kg/m3]

        # flushing
        v_flushing = Q_flushing / (self.B_lock*H_sillB)

        # lock exchange current
        S_diff = (S_lock_openBa - S_B)
        v_exchange = correction_factor_v_exchange * 0.5 * ma.sqrt(abs(self.g * 0.8 * S_diff / rho_MZ * H_effB)) # Propagation velocity of exchange current due to density gradient [m/s]
        V_exchange = 0 # Volume of exchange current due to density gradient [m3]

        # bubble screen
        t_exchange_in_front_of_any_bubble_screen = 0 # time of exchange current due to density gradient before current reaches bubble screen [s]
        if self.D_door_any_bubble_screen_B != 0:
            v_in_front_of_any_bubble_screen = v_exchange - np.sign(self.D_door_any_bubble_screen_B)*v_flushing
            v_in_front_of_any_bubble_screen = np.max([v_in_front_of_any_bubble_screen, 1*10**-10])
            t_exchange_in_front_of_any_bubble_screen = np.abs(self.D_door_any_bubble_screen_B)/v_in_front_of_any_bubble_screen
            t_exchange_in_front_of_any_bubble_screen = np.min([t_exchange_in_front_of_any_bubble_screen,t_door_open])
            frac_lock_exchange_in_front_of_any_bubble_screen = np.max([0,(v_exchange - v_flushing) / v_exchange])
            t_exchange_full = 2 * self.L_lock / v_exchange
            V_exchange += frac_lock_exchange_in_front_of_any_bubble_screen*V_lock_B_Eff*ma.tanh((t_exchange_in_front_of_any_bubble_screen / t_exchange_full))

        # gravity-driven lock exchange current (if no bubble screen: Eff = 1)
        v_exchange_behind_any_bubble_screen = Eff*v_exchange
        if v_exchange_behind_any_bubble_screen:
            frac_lock_exchange_behind_any_bubble_screen = np.max([(v_exchange_behind_any_bubble_screen - v_flushing) / v_exchange_behind_any_bubble_screen, 0])
            t_exchange_full = 2*self.L_lock/v_exchange_behind_any_bubble_screen
        else:
            t_exchange_full = np.inf
            frac_lock_exchange_behind_any_bubble_screen = 1
        t_exchange_behind_any_bubble_screen = np.max([t_door_open-t_exchange_in_front_of_any_bubble_screen,0])
        V_exchange += frac_lock_exchange_behind_any_bubble_screen*(V_lock_B_Eff - V_exchange)*ma.tanh((t_exchange_behind_any_bubble_screen / t_exchange_full))

        #flushing volumes
        V_flushing_B2lock_openB = Q_flushing * t_door_open
        V_flushing_max_B2lock_openB = V_lock_B_Eff - V_exchange
        V_flushing_refresh = np.min([V_flushing_B2lock_openB,V_flushing_max_B2lock_openB])
        V_flushing_passthrough = np.max([V_flushing_B2lock_openB-V_flushing_max_B2lock_openB,0])

        # volume fluxes
        V_lock2A_openBb = V_flushing_B2lock_openB
        V_B2lock_openBb = V_exchange + V_flushing_B2lock_openB
        V_lock2B_openBb = V_exchange

        # mass fluxes
        M_lock2A_openBb = V_flushing_refresh * S_lock_openBa + V_flushing_passthrough * S_B
        M_B2lock_openBb = V_B2lock_openBb * S_B
        M_lock2B_openBb = V_lock2B_openBb * S_lock_openBa

        # Mass flux salt to chamber due to exchange current between chamber and inner [kg]
        M_lock_openBb = M_lock_openBa + M_B2lock_openBb - M_lock2A_openBb - M_lock2B_openBb
        S_lock_openBb = M_lock_openBb / V_lock_B

        # vessels entering lock
        M_lock2B_openBc = V_ship_entry * S_lock_openBb  # Mass flux salt from chamber due to approaching downstream-bounded from inner to chamber [kgm3]
        M_lock_openBc = S_lock_openBb * V_lock_B - M_lock2B_openBc
        S_lock_openBc = M_lock_openBc / (V_lock_B - V_ship_entry)  # Salinity in chamber after ship sails into lock [kg/m3]
        self.S_lock = S_lock_openBc

        # totals opening
        M_lock2B_openBtot = -M_B2lock_openBa + M_lock2B_openBb + M_lock2B_openBc - M_B2lock_openBb
        M_lock2A_openBtot = M_lock2A_openBb
        V_lock2A_openBtot = V_lock2A_openBb
        V_lock2B_openBtot = V_ship_entry + V_exchange
        V_B2lock_openBtot = (V_ship_exit + V_exchange + V_flushing_B2lock_openB)

        # averages opening
        if t_door_open:
            Q_B2lock_mn = V_B2lock_openBtot / t_door_open  # Average discharge from inner to chamber during door opening inner [m3/s]
        else:
            Q_B2lock_mn = 0.0
        if t_door_open:
            Q_lock2B_mn = V_lock2B_openBtot / t_door_open
        else:
            Q_lock2B_mn = 0.0
        Q_lock2A_mn = Q_flushing
        S_B2lock_mn = S_B
        if V_lock2B_openBtot:
            S_lock2B_mn = abs(-(M_lock2B_openBtot - V_B2lock_openBtot * S_B) / V_lock2B_openBtot)
        else:
            S_lock2B_mn = S_B

        output = {'S_lock': self.S_lock,
                  'Q_B2lock_mn': Q_B2lock_mn,
                  'Q_lock2B_mn': Q_lock2B_mn,
                  'Q_lock2A_mn': Q_lock2A_mn,
                  'S_B2lock_mn': S_B2lock_mn,
                  'S_lock2B_mn': S_lock2B_mn,
                  'S_lock_openBa':S_lock_openBa,
                  'M_B2lock_openBa':M_B2lock_openBa,
                  'S_lock_openBb':S_lock_openBb,
                  'M_lock2B_openBb':M_lock2B_openBb,
                  'M_B2lock_openBb':M_B2lock_openBb,
                  'M_lock2B_openBc':M_lock2B_openBc,
                  'M_lock2A_openBtot': M_lock2A_openBtot,
                  'M_lock2B_openBtot': M_lock2B_openBtot,
                  'V_lock2B_openBb':V_lock2B_openBb,
                  'V_lock2A_openBtot': V_lock2A_openBtot,
                  'V_lock2B_openBtot': V_lock2B_openBtot,
                  'V_B2lock_openBtot': V_B2lock_openBtot,
                  'Equilibrium time': t_exchange_full,
                  'v_exchange_behind_any_bubble_screen': v_exchange_behind_any_bubble_screen}

        return output


    def levelling_to_A(self,
                       h_lock=None,
                       h_A=None,  # h_B: Water level at side A (sea) [m]
                       S_lock=None,# S_lock: Initial salinity in lock [kg/m3]
                       S_A=None,   # S_A: Salinity at outer side (sea) [kg/m3]
                       V_ship=0, # V_ship_B2A: Volume of upstream-bounded vessels [m3]
                       t_levelling=None  # T_niv: Levelling time [s]
                       ):

        if h_lock is None:
            h_lock = self.h_B
        if h_A is None:
            h_A = self.h_A
        if S_lock is None:
            S_lock = self.S_lock
        if S_A is None:
            S_A = self.S_A
        if t_levelling is None:
            t_levelling = self.t_levelling_A
        if not V_ship and self.V_ship_B2A:
            V_ship = self.V_ship_B2A

        V_lock_init = self.L_lock * self.B_lock * (h_lock - self.z_lock)  # lock volume levelled with outer harbour [m3]
        V_lock_A = self.L_lock * self.B_lock * (h_A - self.z_lock)  # Lock volume levelled with inner harbour [m3]

        # Levelling phase to inner harbour during higher water at A
        if h_A >= h_lock:
            V_lock2A_nivA = 0  # Levelling volume [m3]
            V_A2lock_nivA = self.L_lock * self.B_lock * (h_A - h_lock)  # Levelling volume [m3]
            M_lock2A_nivA = 0
            M_A2lock_nivA = V_A2lock_nivA * S_A  # Mass flux salt from A to lock chamber [kg]
            S_lock_nivA = (S_lock * (V_lock_init - V_ship) + M_A2lock_nivA) / (V_lock_A - V_ship)  # Salinity in lock chamber after levelling [kg/m3]
            Q_lock2A_nivA = 0
            Q_A2lock_nivA = V_A2lock_nivA / t_levelling  # Average discharge water from outer harbour to lock chamber [m3/s]
            S_lock2A_nivA = 0
            S_A2lock_nivA = S_A  # Average salinity of discharge Q_M1 [m3/s]

        # Levelling phase to inner harbour during lower water at A
        else:
            V_lock2A_nivA = self.L_lock * self.B_lock * (h_lock - h_A)  # Levelling volume [m3]
            V_A2lock_nivA = 0  # Levelling volume [m3]
            M_lock2A_nivA = V_lock2A_nivA * S_lock
            M_A2lock_nivA = 0
            S_lock_nivA = S_lock  # Salinity in lock chamber after levelling [kg/m3]
            Q_lock2A_nivA = V_lock2A_nivA / t_levelling
            Q_A2lock_nivA = 0
            S_lock2A_nivA = S_lock  # Average salinity of discharge Q_M1 [m3/s]
            S_A2lock_nivA = 0

        self.S_lock = S_lock_nivA
        self.h_lock = self.h_A

        output = {'S_lock': self.S_lock,
                  'V_lock2A_nivA': V_lock2A_nivA,
                  'V_A2lock_nivA': V_A2lock_nivA,
                  'M_lock2A_nivA': M_lock2A_nivA,
                  'M_A2lock_nivA': M_A2lock_nivA,
                  'S_lock_nivA': S_lock_nivA,
                  'Q_lock2A_nivA': Q_lock2A_nivA,
                  'Q_A2lock_nivA': Q_A2lock_nivA,
                  'S_lock2A_nivA': S_lock2A_nivA,
                  'S_A2lock_nivA': S_A2lock_nivA}

        return output


    def door_opening_phase_A(self,
                             h_A=None,          # Water level at side A (sea) [m]
                             h_B=None,          # Water level at side A (sea) [m]
                             S_lock=None,       # Initial salinity in lock [kg/m3]
                             S_A=None,          # Salinity at side A (sea) [kg/m3]
                             S_B=None,
                             T_A=None,
                             T_B=None,
                             V_ship_exit=0,     # Volume of exiting vessels (upstream-bounded) [m3]
                             V_ship_entry=0,    # Volume of entering vessels (downstream-bounded) [m3]
                             Eff=None,          # Efficiency countermeasure for exchange current
                             t_door_open=None,       # Representative opening time of door [s]
                             Q_flushing=None,   # Flushing discharge during high water at outer harbour basin [m3/s]
                             correction_factor_v_exchange=None,# calibration factor on propagation velocity salt exchange
                             ):

        if h_A is None:
            h_A = self.h_A
        if h_B is None:
            h_B = self.h_B
        if S_lock is None:
            S_lock = self.S_lock
        if S_A is None:
            S_A = self.S_A
        if S_B is None:
            S_B = self.S_B
        if T_A is None:
            T_A = self.T_A
        if T_B is None:
            T_B = self.T_B
        if Eff is None:
            Eff = self.Eff_A
        if t_door_open is None:
            t_door_open = self.t_door_open_A
        if Q_flushing is None:
            if h_B > h_A:
                Q_flushing = self.Q_flushing_low_tide
            else:
                Q_flushing = self.Q_flushing_high_tide
        if correction_factor_v_exchange is None:
            correction_factor_v_exchange = self.correction_factor_v_exchange_A
        if not V_ship_exit and self.V_ship_B2A:
            V_ship_exit = self.V_ship_B2A
        if not V_ship_entry and self.V_ship_A2B:
            V_ship_entry = self.V_ship_A2B

        H_sillA = h_A - self.z_lock - np.max([0,np.min([(self.z_sillA - self.z_A), (self.z_sillA - self.z_lock)])]) #Head at sill [m]
        V_lock_A = self.L_lock * self.B_lock * (h_A - self.z_lock) #lock volume levelled with inner harbour [m3]
        H_effA = h_A - self.z_lock - 0.8 * np.max([0,np.min([(self.z_sillA - self.z_A), (self.z_sillA - self.z_lock)])]) # Effective depth levelled with inner harbour [m]
        V_lock_A_Eff = self.L_lock * self.B_lock * H_effA  # Effective lock volume levelled with inner harbour [m3]
        rho_MZ = 0.5 * (self.salinity_kgm3_to_density(S_A,T_A)+self.salinity_kgm3_to_density(S_B,T_B)) # Average water density lock complex [kg/m3]

        # vessels exiting lock
        M_A2lock_openAa = V_ship_exit * S_A # Mass flux salt from inner to lock chamber after vessels sailed out [kg]
        M_lock_openAa = abs(S_lock * (V_lock_A - V_ship_exit) + M_A2lock_openAa)
        S_lock_openAa = M_lock_openAa / V_lock_A # Salinity in chamber after ship sails out [kg/m3]

        # flushing
        v_flushing = Q_flushing / (self.B_lock*H_sillA)

        # lock exchange current
        S_diff = (S_A - S_lock_openAa)
        v_exchange = correction_factor_v_exchange * 0.5 * ma.sqrt(abs(self.g * 0.8 * S_diff / rho_MZ * H_effA)) # Propagation velocity of exchange current due to density gradient [m/s]
        H_eq = (2 * (((Q_flushing/self.B_lock)**2) * rho_MZ) / (0.8 * self.g * (S_A - S_B))) ** (1/3)  # Height salt wedge in chamber equilibrium during flushing [m]
        H_eq = np.min([H_eq,h_A-self.z_lock])
        f_lock_exchange = (h_A - self.z_lock - H_eq) / (h_A - self.z_lock)

        V_exchange = 0 # Volume of exchange current due to density gradient [m3]

        # bubble screen
        t_exchange_in_front_of_any_bubble_screen = 0 # time of exchange current due to density gradient before current reaches bubble screen [s]
        t_exchange_full = 0.
        if self.D_door_any_bubble_screen_A != 0:
            v_in_front_of_any_bubble_screen = v_exchange - np.sign(self.D_door_any_bubble_screen_A)*v_flushing
            v_in_front_of_any_bubble_screen = np.max([v_in_front_of_any_bubble_screen, 1*10**-10])
            t_exchange_in_front_of_any_bubble_screen = np.abs(self.D_door_any_bubble_screen_A)/v_in_front_of_any_bubble_screen
            t_exchange_in_front_of_any_bubble_screen = np.min([t_exchange_in_front_of_any_bubble_screen,t_door_open])
            t_exchange_full = 2 * self.L_lock * f_lock_exchange / v_exchange
            V_exchange += frac_lock_exchange_in_front_of_any_bubble_screen*V_lock_A_Eff*ma.tanh((t_exchange_in_front_of_any_bubble_screen / t_exchange_full))

        # gravity-driven lock exchange current (if no bubble screen: Eff = 1)
        v_exchange_behind_any_bubble_screen = Eff*v_exchange
        t_exchange_behind_any_bubble_screen = t_door_open - t_exchange_in_front_of_any_bubble_screen
        if v_exchange_behind_any_bubble_screen > v_flushing:
            if v_exchange_behind_any_bubble_screen:
                t_exchange_full = 2 * self.L_lock * f_lock_exchange / (v_exchange_behind_any_bubble_screen - v_flushing)
            else:
                t_exchange_full = np.inf

            if f_lock_exchange:
                V_exchange += f_lock_exchange*(V_lock_A - V_exchange)*ma.tanh((t_exchange_behind_any_bubble_screen / t_exchange_full))

        # flushing volumes
        V_flushing_lock2A_openA = Q_flushing * t_door_open
        V_flushing_max_lock2A_openA = V_lock_A - V_exchange
        V_flushing_refresh = np.min([V_flushing_lock2A_openA,V_flushing_max_lock2A_openA])
        V_flushing_passthrough = np.max([V_flushing_lock2A_openA-V_flushing_max_lock2A_openA,0])

        # flushing mass fluxes
        M_flushing_B2lock_openA = V_flushing_refresh * S_B + V_flushing_passthrough * S_B
        M_flushing_lock2A_openA = V_flushing_refresh * S_lock_openAa + V_flushing_passthrough * S_B

        # volume fluxes
        V_lock2A_openAb = V_exchange + V_flushing_lock2A_openA
        V_A2lock_openAb = V_exchange
        V_B2lock_openAb = V_flushing_lock2A_openA

        # mass fluxes
        M_lock2A_openAb = M_flushing_lock2A_openA + V_exchange * S_lock_openAa
        M_A2lock_openAb = V_A2lock_openAb * S_A
        M_B2lock_openAb = M_flushing_B2lock_openA

        # Mass flux salt to chamber due to exchange current between chamber and inner [kg]
        M_lock_openAb = M_lock_openAa + M_A2lock_openAb - M_lock2A_openAb + M_B2lock_openAb
        S_lock_openAb = M_lock_openAb / V_lock_A

        # vessels entering lock
        M_lock2A_openAc = V_ship_entry * S_lock_openAb  # Mass flux salt from chamber due to approaching downstream-bounded from inner to chamber [kgm3]
        M_lock_openAc = M_lock_openAb - M_lock2A_openAc
        S_lock_openAc = M_lock_openAc / (V_lock_A - V_ship_entry)  # Salinity in chamber after ship sails into lock [kg/m3]
        self.S_lock = S_lock_openAc

        # totals opening
        M_lock2A_openAtot = -M_A2lock_openAa + M_lock2A_openAc + M_lock2A_openAb - M_A2lock_openAb
        M_B2lock_openAtot = M_B2lock_openAb
        V_lock2A_openAtot = V_lock2A_openAb + V_ship_entry
        V_A2lock_openAtot = V_exchange + V_ship_exit
        V_B2lock_openAtot = V_B2lock_openAb

        # averages opening
        Q_A2lock_mn = V_A2lock_openAtot / t_door_open  # Average discharge from inner to chamber during door opening inner [m3/s]
        Q_lock2A_mn = V_lock2A_openAtot / t_door_open
        Q_B2lock_mn = Q_flushing
        S_A2lock_mn = S_A
        S_lock2A_mn = np.nan
        if V_lock2A_openAtot:
            S_lock2A_mn = abs(-(M_lock2A_openAtot + V_A2lock_openAtot * S_A) / V_lock2A_openAtot)

        output = {'S_lock': self.S_lock,
                  'Q_A2lock_mn': Q_A2lock_mn,
                  'Q_lock2A_mn': Q_lock2A_mn,
                  'Q_B2lock_mn': Q_B2lock_mn,
                  'S_A2lock_mn': S_A2lock_mn,
                  'S_lock2A_mn': S_lock2A_mn,
                  'S_lock_openAa':S_lock_openAa,
                  'S_lock_openAb':S_lock_openAb,
                  'M_A2lock_openAa':M_A2lock_openAa,
                  'M_lock2A_openAb':M_lock2A_openAb,
                  'M_lock2A_openAc':M_lock2A_openAc,
                  'M_A2lock_openAb':M_A2lock_openAb,
                  'M_lock2A_openAtot': M_lock2A_openAtot,
                  'M_B2lock_openAtot': M_B2lock_openAtot,
                  'V_ship_exit':V_ship_exit,
                  'V_lock2A_openAb':V_lock2A_openAb,
                  'V_ship_entry':V_ship_entry,
                  'V_lock2A_openAtot': V_lock2A_openAtot,
                  'V_A2lock_openAtot': V_A2lock_openAtot,
                  'V_B2lock_openAtot': V_B2lock_openAtot,
                  'Equilibrium time': t_exchange_full,
                  'v_exchange_behind_any_bubble_screen': v_exchange_behind_any_bubble_screen}

        return output


    def steady_calc(self):
        self.S_lock = sal_lock_4_init = np.mean([self.S_A,self.S_B])
        convergance = False
        while not convergance:
            results_phase1 = self.levelling_to_B()
            S_before_opening_to_B = results_phase1['S_lock']
            results_phase2 = self.door_opening_phase_B()
            S_before_levelling_to_A = results_phase2['S_lock']
            results_phase3 = self.levelling_to_A()
            S_before_opening_to_A = results_phase3['S_lock']
            results_phase4 = self.door_opening_phase_A()
            sal_lock_4 = results_phase4['S_lock']
            convergance = self.has_convergance(sal_lock_4_init,sal_lock_4)
            sal_lock_4_init = results_phase4['S_lock']

        V_B2lock_tot = results_phase1['V_B2lock_nivB'] + results_phase2['V_B2lock_openBtot'] + results_phase4['V_B2lock_openAtot']
        M_lock2A_tot = results_phase2['M_lock2A_openBtot'] + results_phase3['M_lock2A_nivA'] + results_phase4['M_lock2A_openAtot']
        M_lock2B_tot = results_phase1['M_lock2B_nivB'] + results_phase2['M_lock2B_openBtot'] - results_phase4['M_B2lock_openAtot']
        output = {'S_before_opening_to_B':S_before_opening_to_B,
                  'S_before_levelling_to_A':S_before_levelling_to_A,
                  'S_before_opening_to_A':S_before_opening_to_A,
                  'S_before_levelling_to_B':sal_lock_4,
                  'M_lock2B_nivB':results_phase1['M_lock2B_nivB'],
                  'M_lock2B_openBtot':results_phase2['M_lock2B_openBtot'],
                  'M_lock2A_openAtot':results_phase4['M_lock2A_openAtot'],
                  'Equilibium_time_B':results_phase2['Equilibrium time'],
                  'Equilibium_time_A':results_phase4['Equilibrium time'],
                  'V_B2lock_tot':V_B2lock_tot,
                  'M_lock2A_tot':M_lock2A_tot,
                  'M_lock2B_tot':M_lock2B_tot}

        return output