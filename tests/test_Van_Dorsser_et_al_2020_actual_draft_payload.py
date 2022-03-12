import numpy as np

import pytest

def calculate_actual_T_and_payload(self, h_min, ukc=.3,vesl_type="Dry_DH"):
        """ Calculate actual draft based on Van Dorsser et al 2020
        van Dorsser, C., Vinke, F., Hekkenberg, R. & van Koningsveld, M. (2020). The effect of low water on loading capacity of inland ships. European Journal of Transport and Infrastructure Research, 20(3), 47�70. https://doi.org/10.18757/ejtir.2020.20.3.3981
        """
        #Design draft T_design, refer to Table 5

        Tdesign_coefs = dict({"intercept":0,
                         "c1": 1.7244153371,
                         "c2": 2.2767179246,
                         "c3": 1.3365379898,
                         "c4": -5.9459308905,
                         "c5": 6.2902305560*10**-2,
                         "c6": 7.7398861528*10**-5,
                         "c7": 9.0052384439*10**-3,
                         "c8": 2.8438560877
                         })

        assert vesl_type in ["Container","Dry_SH","Dry_DH","Barge","Tanker"],'Invalid value vesl_type, should be "Container","Dry_SH","Dry_DH","Barge" or "Tanker"'
        if vesl_type == "Container":
            [dum_container, dum_dry,
            dum_barge, dum_tanker] = [1,0,0,0]
        elif vesl_type == "Dry_SH":
            [dum_container, dum_dry,
            dum_barge, dum_tanker] = [0,1,0,0]
        elif vesl_type == "Dry_DH":
            [dum_container, dum_dry,
            dum_barge, dum_tanker] = [0,1,0,0]
        elif vesl_type == "Barge":
            [dum_container, dum_dry,
            dum_barge, dum_tanker] = [0,0,1,0]

        elif vesl_type == "Tanker":
            [dum_container, dum_dry,
            dum_barge, dum_tanker] = [0,0,0,1]

        T_design = Tdesign_coefs['intercept'] + (dum_container * Tdesign_coefs['c1']) + \
                                                (dum_dry * Tdesign_coefs['c2']) + \
                                                (dum_barge * Tdesign_coefs['c3']) +\
                                                (dum_tanker * Tdesign_coefs['c4']) +\
                                                (Tdesign_coefs['c5'] * dum_container * self.L**0.4 * self.B**0.6) +\
                                                (Tdesign_coefs['c6'] * dum_dry * self.L**0.7 * self.B**2.6)+\
                                                (Tdesign_coefs['c7'] * dum_barge * self.L**0.3 * self.B**1.8) +\
                                                (Tdesign_coefs['c8'] * dum_tanker * self.L**0.1 * self.B**0.3)

        #Empty draft T_empty, refer to Table 4
        Tempty_coefs = dict({"intercept": 7.5740820927*10**-2,
                    "c1": 1.1615080992*10**-1,
                    "c2": 1.6865973494*10**-2,
                    "c3": -2.7490565381*10**-2,
                    "c4": -5.1501240744*10**-5,
                    "c5": 1.0257551153*10**-1,
                    "c6": 2.4299435211*10**-1,
                    "c7": -2.1354295627*10**-1,
                    })


        if vesl_type == "Container":
            [dum_container, dum_dry,
            dum_barge, dum_tanker] = [1,0,0,0]
        elif vesl_type == "Dry_SH":
            [dum_container, dum_dry,
            dum_barge, dum_tanker] = [0,0,0,0]
        elif vesl_type == "Dry_DH":
            [dum_container, dum_dry,
            dum_barge, dum_tanker] = [0,1,0,0]
        elif vesl_type == "Barge":
            [dum_container, dum_dry,
            dum_barge, dum_tanker] = [0,0,1,0]

        elif vesl_type == "Tanker":
            [dum_container, dum_dry,
            dum_barge, dum_tanker] = [0,0,0,1]

        # dum_container and dum_dry use the same "c5"   
        T_empty = Tempty_coefs['intercept']  + (Tempty_coefs['c1'] * self.B) + \
                                               (Tempty_coefs['c2'] * ((self.L * T_design) / self.B)) + \
                                               (Tempty_coefs['c3'] * (np.sqrt(self.L * self.B)))  + \
                                               (Tempty_coefs['c4'] * (self.L * self.B * T_design)) +  \
                                               (Tempty_coefs['c5'] * dum_container) + \
                                               (Tempty_coefs['c5'] * dum_dry)   + \
                                               (Tempty_coefs['c6'] * dum_tanker) + \
                                               (Tempty_coefs['c7'] * dum_barge)

        #Actual draft T_actual
        if (T_design <= (h_min - ukc)):
            T_actual = T_design

        elif T_empty > (h_min - ukc):
            T_actual =  (f"No trip possible. Available depth smaller than empty draft: {h_min - T_empty} m")

        elif (T_design > (h_min - ukc)):
            T_actual = h_min -  ukc

        print('The actual draft is', T_actual, 'm')

        #Capacity indexes, refer to Table 3 and eq 2 
        CI_coefs = dict({"intercept": 2.0323139721 * 10**1,

                "c1": -7.8577991460 * 10**1,
                "c2": -7.0671612519 * 10**0,
                "c3": 2.7744056480 * 10**1,
                "c4": 7.5588609922 * 10**-1,
                "c5": 3.6591813315 * 10**1
                })
        # Capindex_1 related to actual draft (especially used for shallow water)
        Capindex_1 = CI_coefs["intercept"] + (CI_coefs["c1"] * T_empty) + (CI_coefs["c2"] * T_empty**2)  +  (
        CI_coefs["c3"] * T_actual) + (CI_coefs["c4"] * T_actual**2)   + ( CI_coefs["c5"] * (T_empty * T_actual))
        # Capindex_2 related to design draft
        Capindex_2 = CI_coefs["intercept"] + (CI_coefs["c1"] * T_empty) + (CI_coefs["c2"] * T_empty**2)   + (
        CI_coefs["c3"] * T_design) + (CI_coefs["c4"] * T_design**2)  + (CI_coefs["c5"] * (T_empty * T_design))
     
        #DWT design capacity, refer to Table 6 and eq 3
        capacity_coefs = dict({"intercept": -1.6687441313*10**1,
             "c1": 9.7404521380*10**-1,
             "c2": -1.1068568208,
             })

        DWT_design = capacity_coefs['intercept'] + (capacity_coefs['c1'] * self.L * self.B * T_design) + (
         capacity_coefs['c2'] * self.L * self.B * T_empty) # designed DWT 
        DWT_actual = (Capindex_1/Capindex_2)*DWT_design # actual DWT of shallow water
        
       
        if T_actual < T_design:
            consumables=0.04 #consumables represents the persentage of fuel weight,which is 4-6% of designed DWT 
                              # 4% for shallow water (Van Dosser  et al. Chapter 8,pp.68).
        else: 
            consumables=0.06 #consumables represents the persentage of fuel weight,which is 4-6% of designed DWT 
                              # 6% for deep water (Van Dosser et al. Chapter 8, pp.68).
        
        fuel_weight=DWT_design*consumables #(Van Dosser et al. Chapter 8, pp.68).
        actual_max_payload = DWT_actual-fuel_weight # payload=DWT-fuel_weight
        print('The actual_max_payload is', actual_max_payload, 'ton')

        # set vessel properties
        self.T = T_actual
      
        # self.container.level = actual_max_payload

        # return T_actual, actual_max_payload


        # Test if the output of "Dry_DH" vessel are the same as Table 7 showing in(Van Dorsser  et al.)
        np.testing.assert_almost_equal(calculate_actual_T_and_payload(self, h_min=1.7, ukc=.3,vesl_type="Dry_DH"), (1.4, 469) , decimal=0, err_msg='', verbose=True)
    
    
    
    
    
    
    