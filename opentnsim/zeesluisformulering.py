import math as ma
import numpy as np

class ZeesluisFormulering():

    def __init__(self,
                 L_K=545,      # L_K: Length lock chamber [m]
                 B_K=70,       # B_K: Width lock chamber [m]
                 Z_K=-17.75,   # Z_K: Bottom level lock chamber [mNAP]
                 Z_VZ=-18.00,  # Z_VZ: Bottom level at outer harbour basin [mNAP]
                 Z_VM=-17.50,  # Z_VM: Bottom level at inner harbour basin [mNAP]
                 Z_DM=-17.00,  # Z_DM: Bottom level at sil of inner harbour basin door [mNAP] (If 0: equal to depth of lock chamber)
                 Z_DZ=-17.00,  # Z_DM: Bottom level at sil of inner harbour basin door [mNAP] (If 0: equal to depth of lock chamber)

                 Eff=1,        # Efficiency countermeasure for exchange current [-]
                 Q_spoelen=0,  # Flushing discharge during high water at outer harbour basin [m3/s]

                 T_D=4.5,      # T_D: Door opening/closing time [min]
                 T_niv=5,      # T_niv: Levelling time [min]
                 V_opw=15000,  # V_opw: Volume of upstream-bounded vessels [m3]
                 V_afw=15000,  # V_afw: Volume of downstream-bounded vessels [m3]
                 h_M=-0.45,    # Water level at inland water side [mNAP]
                 h_Z=0,        # Water level at sea side [mNAP]

                 S_M=7.2,      # Salinity at inner side (inland) [kg/m3]
                 S_Z=14.2,     # Salinity at outer side (sea) [kg/m3]

                 T_bed=24,     # Total operating time of lock complex per day [hours]
                 N_cyc=12,     # N_cyc: Number of lock cycles per day

                 c_dot=1,      # Calibration coefficient representative mean door open time
                 c_M_gem=1,    # Calibration coefficient door open time inequality
                 g=9.81,       # Gravitational constant
                 ):

        self.L_K = L_K
        self.B_K = B_K
        self.Z_K = Z_K
        self.h_M = h_M
        self.h_Z = h_Z
        self.S_M = S_M
        self.S_Z = S_Z
        self.T_bed = T_bed
        self.V_opw = V_opw
        self.V_afw = V_afw
        self.T_D = T_D
        self.T_niv = T_niv
        self.N_cyc = N_cyc
        self.Z_VZ = Z_VZ
        self.Z_VM = Z_VM
        self.Z_DM = Z_DM
        self.Z_DZ = Z_DZ
        self.Eff = Eff
        self.c_dot = c_dot
        self.c_M_gem = c_M_gem
        self.g = g
        self.Q_spoelen = Q_spoelen
        self.Z_DM = Z_DM
        self.Z_DZ = Z_DZ

        if not self.Z_DM:
            self.Z_DM = Z_K

        if not self.Z_DZ:
            self.Z_DZ = Z_K  # Depth at sill of outer harbour basin door [m] (aan te passen) -> Aangenomen gelijk aan diepte kolk


        # General formulas
        self.T_cyc = (self.T_bed * 3600) / self.N_cyc # Average time of full lock cycle [s]
        self.T_op_gem = 0.5 * self.T_cyc - (self.T_niv + self.T_D) * 60 # Average time of door open [s]
        self.T_op = self.c_dot * self.T_op_gem # Representative time of door open [s]
        self.T_op_M = self.c_M_gem * self.T_op # Representative opening time inner harbour door [s]
        self.T_op_Z = (2 - self.c_M_gem) * self.T_op # Representative opening time outer harbour door [s]

        self.V_K_M = self.L_K * self.B_K * (self.h_M - self.Z_K)
        self.V_K_Z = self.L_K * self.B_K * (self.h_Z - self.Z_K)

        self.H_M_Eff = abs((self.h_M - self.Z_DM) + 0.2 * min((self.Z_DM - self.Z_VM), (self.Z_DM - self.Z_K))) # Effective depth levelled with inner harbour [m]
        self.V_K_M_Eff = self.L_K * self.B_K * self.H_M_Eff # Effective lock volume levelled with inner harbour [m3]
        self.H_Z_Eff = abs(((self.h_Z - self.Z_DZ) + 0.2 * min((self.Z_DZ - self.Z_VZ), (self.Z_DZ - self.Z_K)))) # Effective depth levelled with outer harbour [m]
        self.L_K_Eff = abs(self.L_K * ((self.h_Z - self.Z_K) / self.H_Z_Eff)) # Effective length lock levelled with outer harbour [m]
        self.V_K_Z_Eff = self.L_K_Eff * self.B_K * self.H_Z_Eff # Effective lock volume levelled with outer harbour [m3]
        self.rho_MZ = 1000 + 0.8 * (0.5 * (self.S_M + self.S_Z)) # Average water density lock complex [kg/m3]


    def levelling_to_inner_harbour(self,
                                   S_K,  # S_K: Initial salinity in lock [kg/m3]
                                   h_M=None,  # h_M: Water level at inland water side [m]
                                   h_Z=None,  # h_Z: Water level at sea side [m]
                                   S_M=None,  # S_M: Salinity at inner side (inland) [kg/m3]
                                   S_Z=None,  # S_Z: Salinity at outer side (sea) [kg/m3]
                                   V_opw=None,  # V_opw: Volume of upstream-bounded vessels [m3]
                                   T_niv=None,  # T_niv: Levelling time [s]
                                   ):

        if h_M is not None:
            V_K_M = self.L_K * self.B_K * (h_M - self.Z_K) # Lock volume levelled with inner harbour [m3]
            V_K_Z = self.L_K * self.B_K * (h_Z - self.Z_K) # lock volume levelled with outer harbour [m3]

        else:
            h_M = self.h_M
            h_Z = self.h_Z
            S_M = self.S_M
            S_Z = self.S_Z
            V_opw = self.V_opw
            T_niv = self.T_niv
            V_K_M = self.V_K_M
            V_K_Z = self.V_K_Z


        # LW1: Levelling phase to inner harbour during low water at outer harbour
        if h_M >= h_Z: # LW1
            V_niv_HW = 0
            V_niv_LW = self.L_K * self.B_K * (h_M - h_Z) # Levelling volume [m3]
            M_M_LW1 = V_niv_LW * S_M  # Mass flux salt from inner harbour to lock chamber [kg]
            self.S_K_Z_niv = abs((S_K * (V_K_Z - V_opw) + M_M_LW1) / (V_K_M - V_opw))  # Salinity in lock chamber after levelling [kg/m3]
            self.Q_M1 = V_niv_LW / T_niv  # Average discharge water from outer harbour to lock chamber [m3/s]
            self.S_M1 = abs(S_M)  # Average salinity of discharge Q_M1 [m3/s]


        # HW1
        else:
            V_niv_HW = self.L_K * self.B_K * (h_Z - h_M)  # Levelling volume [m3]
            V_niv_LW = 0  # [m3]
            self.S_K_Z_niv = abs(S_K) # Salinity in lock chamber after levelling [kg/m3]

            self.Q_M1 = V_niv_HW / T_niv # Average discharge water from outer harbour to lock chamber [m3/s]
            self.S_M1 = abs(S_Z) # Average salinity of discharge Q_M1 [kg/s]


        self.M_M1 = V_niv_LW * S_M - V_niv_HW * S_K # Total mass flux salt from outer harbour to lock chamber [kgm3]

        return self.S_K_Z_niv, self.M_M1


    def door_opening_phase_inner_harbour(self,
                                         S_K,            # S_K: Initial salinity in lock [kg/m3]
                                         h_M=None,       # h_M: Water level at inland water side [m]
                                         S_M=None,       # S_M: Salinity at inner side (inland) [kg/m3]
                                         S_Z=None,       # S_Z: Salinity at outer side (sea) [kg/m3]
                                         V_opw=None,     # V_opw: Volume of upstream-bounded vessels [m3]
                                         V_afw=None,     # V_afw: Volume of downstream-bounded vessels [m3]
                                         Eff=None,       # Efficiency countermeasure for exchange current
                                         T_op_M=None,    # T_op_M: Representative opening time inner harbour door [s]
                                         Q_spoelen=None, # Q_spoelen: Flushing discharge during high water at outer harbour basin [m3/s]
                                         fc_ci=1.0,      # calibration factor on propagation velocity salt exchange
                                         par=False,       # returning of parameters
                                         ):

        if h_M is not None:
            V_K_M = self.L_K * self.B_K * (h_M - self.Z_K) #lock volume levelled with inner harbour [m3]
            H_M_Eff = abs((h_M - self.Z_DM) + 0.2 * min((self.Z_DM - self.Z_VM), (self.Z_DM - self.Z_K)))  # Effective depth levelled with inner harbour [m]
            V_K_M_Eff = V_K_M #self.L_K * self.B_K * H_M_Eff  # Effective lock volume levelled with inner harbour [m3]
            rho_MZ = 1000 + 0.8 * (0.5 * (S_M + S_Z))  # Average water density lock complex [kg/m3]

        else:
            S_M = self.S_M
            V_opw = self.V_opw
            V_afw = self.V_afw
            Eff = self.Eff
            T_op_M = self.T_op_M
            Q_spoelen = self.Q_spoelen
            rho_MZ = self.rho_MZ
            V_K_M = self.V_K_M

            V_K_M_Eff = self.V_K_M_Eff
            H_M_Eff = self.H_M_Eff

        # vessels exiting lock
        self.M_M2a = V_opw * S_M # Mass flux salt from inner to lock chamber after vessels sailed out to inner [kg]
        self.S_K2a = abs((S_K * (V_K_M - V_opw) + self.M_M2a) / V_K_M) # Salinity in chamber after ship sails out to inner [kg/m3]

        # lock exchange current
        c_i_M = fc_ci * (0.5 * ma.sqrt(abs(self.g * H_M_Eff * ((0.8 * (self.S_K2a - S_M)) / rho_MZ))))  # Propagation velocity of exchange current due to density gradient between chamber and inner [m/s]
        if c_i_M:
            f_LE_sp_M = (c_i_M - (Q_spoelen / (self.B_K * H_M_Eff))) / c_i_M  # Efficiency factor of flushing discharge to delay exchange current between chamber and inner
            T_LE_M = 2 * self.L_K / c_i_M  # Time for exchange current to fully exchange salt from chamber and inner [s]
            V_U_M = V_K_M_Eff * ma.tanh(Eff * (T_op_M / T_LE_M)) * f_LE_sp_M  # Volume of exchange current between chamber and inner [m3]
            self.M_M2b_LE = np.max([(S_M * V_U_M - self.S_K2a * V_U_M), V_K_M_Eff * (S_M - self.S_K2a)])  # Mass flux salt to chamber due to exchange current between chamber and inner [kg]
        else:
            V_U_M = 0
            T_LE_M = 0
            self.M_M2b_LE = 0

        # flushing
        V_sp_KM = Q_spoelen * T_op_M # Volume of flushing discharge during door opening [m3]
        V_sp_max_KM = V_K_M_Eff - V_U_M # Volume of flushing discharge at maximum efficiency during door opening [m3]
        self.M_M2b_sp = V_sp_KM * S_M # Mass flux salt from inner harbour to chamber due to flushing [kg]
        self.M_Z2b_sp = min(V_sp_KM, V_sp_max_KM) * self.S_K2a + max((V_sp_KM - V_sp_max_KM), 0) * S_M # Mass flux salt from chamber to outer due to flushing at opening inner [kg]
        self.S_K2b = abs((self.S_K2a * V_K_M + self.M_M2b_LE + self.M_M2b_sp - self.M_Z2b_sp) / V_K_M) # Salinity in chamber at moment downstream-bounded sail into chamber [kg/m3]

        # vessels entering lock
        self.M_M2c = -V_afw * self.S_K2b # Mass flux salt to chamber due to approaching downstream-bounded from inner to chamber [kgm3]
        self.S_KM = abs((self.S_K2b * V_K_M + self.M_M2c) / (V_K_M - V_afw)) # Salinity in chamber after ship sails into lock [kg/m3]

        # lock opening total
        self.M_M2 = self.M_M2a + self.M_M2b_LE + self.M_M2b_sp + self.M_M2c # Total mass flux salt to chamber during door opening [kgm3]
        self.M_Z2 = self.M_Z2b_sp # Total mass flux salt to outer during opening inner [kgm3]
        # self.S_KM = abs((S_K * (V_K_M_Eff - V_opw) + self.M_M2 - self.M_Z2) / (V_K_M_Eff - V_afw)) # Salinity of chamber after door opening at inner [kg/m3] This step does not take back to back steps into account, but all simultanious

        # avarages
        self.Q_M2_mn = (V_opw + V_U_M + V_sp_KM) / T_op_M # Average discharge from inner to chamber during door opening inner [m3/s]
        self.S_M2_mn = abs(S_M) # Average salinity of discharge from inner to chamber during door opening inner [kg/m3]
        self.Q_M2_pl = (V_afw + V_U_M) / T_op_M # Average discharge from chamber to inner during door opening inner [m3/s]
        self.S_M2_pl = abs(-(self.M_M2 - (V_opw + V_U_M + V_sp_KM) * S_M) / (V_afw + V_U_M)) # Average salinity of discharge from chamber to inner during door opening inner [kg/m3]
        self.Q_Z2 = Q_spoelen # Average discharge from chamber to outer during door opening inner [m3/s]
        if self.Q_Z2:
            self.S_Z2 = abs(self.M_Z2 / (Q_spoelen * T_op_M)) # Average salinity of discharge from chamber to outer during door opening inner [kg/m3]
        else:
            self.S_Z2 = None

        if not par:
            return self.S_KM, self.M_M2

        else:
            return self.S_KM, self.M_M2, self.S_K2a, self.M_M2a, self.S_K2b, self.M_M2b_LE, self.M_M2c, T_LE_M, c_i_M


    def levelling_to_outer_harbour(self,
                                   S_K,        # S_K: Initial salinity in lock [kg/m3]
                                   h_M=None,   # h_M: Water level at inland water side [m]
                                   h_Z=None,   # h_Z: Water level at sea side [m]
                                   S_M=None,   # S_M: Salinity at the inner side (inland) [kg/m3]
                                   S_Z=None,   # S_Z: Salinity at outer side (sea) [kg/m3]
                                   V_afw=None, # V_afw: Volume of upstream-bounded vessels [m3]
                                   T_niv=None  # T_niv: Levelling time [s]
                                   ):

        if h_M is not None:
            V_K_M = self.L_K * self.B_K * (h_M - self.Z_K)  # Lock volume levelled with inner harbour [m3]
            V_K_Z = self.L_K * self.B_K * (h_Z - self.Z_K)  # lock volume levelled with outer harbour [m3]

        else:
            h_M = self.h_M
            h_Z = self.h_Z
            S_M = self.S_M
            S_Z = self.S_Z
            V_afw = self.V_afw
            T_niv = self.T_niv
            V_K_M = self.V_K_M
            V_K_Z = self.V_K_Z

        # LW3: Levelling phase to outer harbour during low water at outer harbour
        if h_M >= h_Z: # LW3
            V_niv_HW = 0
            V_niv_LW = self.L_K * self.B_K * (h_M - h_Z)  # Levelling volume [m3]
            M_Z_LW3 = V_niv_LW * S_K  # Mass flux salt to outer harbour after levelling [kg]
            self.S_K_M_niv = abs(S_K)  # Salinity lock chamber after levelling [kg/m3]
            self.Q_Z3 = V_niv_LW / T_niv
            self.S_Z3 = abs(S_M)

        else:  # HW3
            V_niv_HW = self.L_K * self.B_K * (h_Z - h_M)  # Levelling volume [m3]
            V_niv_LW = 0  # [m3]
            M_Z_HW3 = -V_niv_HW * S_Z  # Mass flux salt from chamber to outer with levelling to outer [kgm3]
            self.S_K_M_niv = abs((S_K * (V_K_M - V_afw) - M_Z_HW3) / (V_K_Z - V_afw))  # Salinity of lock chamber after levelling to outer [kg/m3]
            self.Q_Z3 = V_niv_HW / T_niv  # Average discharge from outer to chamber with levelling to outer [m3/s]
            self.S_Z3 = S_Z  # Average salinity of discharge from outer to chamber with levelling to outer [kg/m3]

        self.M_Z3 = V_niv_LW * self.S_K_M_niv - V_niv_HW * S_Z  # Total mass flux salt from chamber to outer with levelling to outer [kgm3]

        return self.S_K_M_niv, self.M_Z3


    def door_opening_phase_outer_harbour(self,
                                         S_K,            # S_K: Initial salinity in lock [kg/m3]
                                         h_Z=None,       # h_Z: Water level at sea side [mNAP]
                                         S_M=None,       # S_M: Salinity at inner side (inland) [kg/m3]
                                         S_Z=None,       # S_Z: Salinity at outer side (sea) [kg/m3]
                                         V_afw=None,     # V_afw: Volume of downstream-bounded vessels [m3]
                                         V_opw=None,     # V_opw: Volume of upstream-bounded vessels [m3]
                                         Eff=None,       # Eff: Efficiency countermeasure for exchange current
                                         T_op_Z=None,    # T_op_Z: Representative opening time outer harbour door [s]
                                         Q_spoelen=None, # Q_spoelen: Flushing discharge during high water at outer harbour basin [m3/s]
                                         fc_ci=1.0,      # calibration factor on propagation velocity salt exchange
                                         par=None,       # returning of parameters [0,1]
                                         ):

        if h_Z is not None:
            V_K_Z = self.L_K * self.B_K * (h_Z - self.Z_K)  # lock volume levelled with inner harbour [m3]
            H_Z_Eff = abs(((h_Z - self.Z_DZ) + 0.2 * min((self.Z_DZ - self.Z_VZ), (self.Z_DZ - self.Z_K))))  # Effective depth levelled with outer harbour [m]
            L_K_Eff = abs(self.L_K * ((self.h_Z - self.Z_K) / self.H_Z_Eff))  # Effective length lock levelled with outer harbour [m]
            V_K_Z_Eff = V_K_Z #self.L_K * self.B_K * H_Z_Eff  # Effective lock volume levelled with inner harbour [m3]
            rho_MZ = 1000 + 0.8 * (0.5 * (S_M + S_Z))  # Average water density lock complex [kg/m3]

        else:
            S_M = self.S_M
            S_Z = self.S_Z
            V_opw = self.V_opw
            V_afw = self.V_afw
            Eff = self.Eff
            T_op_Z = self.T_op_Z
            Q_spoelen = self.Q_spoelen
            rho_MZ = self.rho_MZ
            V_K_Z_Eff = self.V_K_Z_Eff
            H_Z_Eff = self.H_Z_Eff
            L_K_Eff = self.L_K_Eff
            V_K_Z = self.V_K_Z

        # vessels exiting lock
        self.M_Z4a = -V_afw * S_Z # Mass flux salt from outer chamber after vessels sailed out to outer [kgm3]
        self.S_K4a = abs((S_K * (V_K_Z - V_afw) - self.M_Z4a) / V_K_Z) # Salinity in chamber after ship sails out to outer [kg/m3]

        # lock exchange current
        c_i_Z = fc_ci * (0.5 * ma.sqrt(abs(self.g * H_Z_Eff * ((0.8 * (S_Z - self.S_K4a)) / rho_MZ))))  # Propagation velocity of exchange current due to density gradient between chamber and outer [m/s]
        if c_i_Z:
            H_Eq = (2 * ((Q_spoelen ** 2) * rho_MZ) / ((self.B_K ** 2) * 0.8 * self.g * (S_Z - S_M))) ** 0.333  # Height salt wedge in chamber equilibrium during flushing [m]
            f_LE_sp_Z = (H_Z_Eff - H_Eq) / H_Z_Eff  # Efficiency factor of flushing discharge to delay exchange current between chamber and outer
            T_LE_Z = (2 * f_LE_sp_Z * L_K_Eff) / (Eff * c_i_Z - (Q_spoelen / (self.B_K * H_Z_Eff)))  # Time for exchange current to fully exchange salt from chamber and outer [s]
            V_U_Z = f_LE_sp_Z * V_K_Z_Eff * ma.tanh(Eff * (T_op_Z / T_LE_Z))  # Volume of exchange current between chamber and outer [m3]
            self.M_Z4b_LE = np.max([(V_U_Z * self.S_K4a - V_U_Z * S_Z), V_K_Z_Eff * (self.S_K4a - S_Z)])  # Mass flux salt to chamber due to exchange current between chamber and outer [kgm3]
        else:
            V_U_Z = 0
            T_LE_Z = 0
            self.M_Z4b_LE = 0

        # flushing
        V_sp_KZ = Q_spoelen * T_op_Z # Volume of flushing discharge during door opening
        V_sp_max_KZ = V_K_Z_Eff - V_U_Z # Volume of flushing discharge at maximum efficiency during door opening [m3]
        self.M_M4b_sp = V_sp_KZ * S_M # Mass flux salt from inner to chamber due to flushing [kgm3]
        self.M_Z4b_sp = min(V_sp_KZ, V_sp_max_KZ) * self.S_K4a + max((V_sp_KZ - V_sp_max_KZ), 0) * S_M # Mass flux salt from chamber to outer due to flushing at opening inner [kgm3]
        self.S_K4b = abs((self.S_K4a * V_K_Z - self.M_Z4b_LE + self.M_M4b_sp - self.M_Z4b_sp) / V_K_Z) # Salinity in chamber at moment upstream-bounded sail into chamber [kg/m3]

        # vessels entering lock
        self.M_Z4c = V_opw * self.S_K4b # Mass flux salt from chamber to outer due to approaching upstream-bounded from outer to chamber [kgm3]
        self.S_KZ = abs((self.S_K4b * V_K_Z - self.M_Z4c) / (V_K_Z - V_opw)) # Salinity in chamber after ship sails into lock [kg/m3]

        # lock open total
        self.M_Z4 = self.M_Z4a + self.M_Z4b_LE + self.M_Z4b_sp + self.M_Z4c # Total mass flux salt from chamber to outer during door opening [kgm3]
        self.M_M4 = self.M_Z4b_sp # Total mass flux salt to chamber from inner during opening outer [kgm3]
        # self.S_KZ = abs((S_K * (V_K_Z_Eff - V_afw) + self.M_M4 - self.M_Z4) / (V_K_Z_Eff - V_opw)) # Salinity of chamber after door opening at outer [kg/m3]

        # averages
        self.Q_Z4_mn = (V_afw + V_U_Z) / T_op_Z # Average discharge from outer to chamber during door opening outer [m3/s]
        self.S_Z4_mn = S_Z # Average salinity of discharge from outer to chamber during door opening outer [kg/m3]
        self.Q_Z4_pl = (V_opw + V_U_Z + V_sp_KZ) / T_op_Z # Average discharge from chamber to outer during door opening outer [m3/s]
        self.S_Z4_pl = abs((self.M_M4 - (V_afw + V_U_Z) * S_Z) / (V_opw + V_U_Z + V_sp_KZ)) # Average salinity of discharge from chamber to outer during door opening outer [kg/m3]
        self.Q_M4_mn = Q_spoelen # Average discharge from inner to chamber during door opening outer [m3/s]
        self.S_M4_mn = abs(S_M) # Average salinity of discharge from inner to chamber during door opening outer [kg/m3]

        if par is None:
            return self.S_KZ, self.M_Z4, self.M_M4

        else:
            return self.S_KZ, self.M_Z4, self.M_M4, self.S_K4a, self.M_Z4a, self.S_K4b, self.M_Z4b_LE, self.M_Z4c, T_LE_Z, c_i_Z


    def full_lock_cycle(self):
        S_K = self.S_Z

        self.S_K_Z_niv, self.M_M1 = self.levelling_to_inner_harbour(S_K)
        self.S_KM, self.M_M2 = self.door_opening_phase_inner_harbour(self.S_K_Z_niv)
        self.S_K_M_niv, self.M_Z3 = self.levelling_to_outer_harbour(self.S_KM)
        self.S_KZ, self.M_Z4, self.M_M4 = self.door_opening_phase_outer_harbour(self.S_K_M_niv)

        S_K = self.S_KZ
        return S_K