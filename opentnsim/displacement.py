# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:34:31 2024, last update 21-06-2024

This file provides a class object to define ship capacity and displacement at
a given draft. 


Checked against excel file with model validation 
for motorcontainerships and motortankers

@author: cornelis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Ship:
    
    def __init__(self, Length, Beam, Type, Cargo, Hull, Te, Td, LSW, DWTfw, 
                 Force):
        """
        Initialize the ship properties based on known parameters. 
        The more parameters specified, the higher the accuracy of the model.
        
        Parameters
        ----------
        Length : float
            Length overall in meters (required).
        Beam : float
            Beam molded in meters (required).
        Type : str
            Type of the ship, 'motorship', 'dump barge', 'coupling barge' 
            (conditional).
        Cargo : str
            Type of cargo, 'dry', 'container', or 'liquid' (conditional).
        Hull : str
            Hull type, 'single' or 'double' (conditional).
        Te : float, optional
            Empty equal loaded draught in meters.
        Td : float, optional
            Design draught in meters.
        LSW : float, optional
            Light ship weight in metric tonnes.
        DWTfw : float, optional
            Deadweight in metric tonnes in fresh water with density 1.
        Force : str, optional
            Optional parameter to force the use of Hekkenberg (2013) instead 
            of Van Dorsser et al. (2020) for solving certain option 14-16, 
            default is None. Use 'HB' to force method Hekkenberg (2013).
                     
        Attributes
        ----------
        alpha : float
            Scaling factor defined as capacity at 2.50 meters / 100.
        beta : float
            Shape factor defined as LSW / (Awl * Te).
        epsilon : float
            Absolute error term defined in tonnes.
        Awl : float
            Waterline area in square meters measured at Te.
        Cwl : float
            Waterline coefficient defined as Awl / (Length x Beam).
        Cb : float
            Block coefficient defined as LSW / (Length x Beam x Te).
        a, b, c : float
            Shape parameters.
            
        Notes
        -----
        - Type information is required unless all optional parameters Te, 
        Td, LSW, DWTfw are known.
        - Cargo information is required if less than two of the four optional 
        parameters (Te, Td, LSW, DWTfw) are known, or if Te and LSW are known 
        but Td and DWTfw are not.
        - Hull information is required if only one of the optional parameters 
        (Td or DWTfw) is known, or if nothing is known. For modern tankers, 
        the hull parameter is 'double' by default.
        - For accurate representation, it is recommended to include Type, 
        Cargo, and Hull information. If no conditional data is set, default 
        values are 'motorship', 'dry' cargo, and 'double' hull.
        """
        
        # Set main ship properties equal to input
        self.Length = Length
        self.Beam = Beam
        self.Type = Type        
        self.Cargo = Cargo        
        self.Hull = Hull        
        self.Te = Te
        self.Td = Td
        self.LSW = LSW
        self.DWTfw = DWTfw    
        self.Force = Force
        self.Note = []
                
        # Set parameters relative capacity model Van Dorsser et al. (2020)
        self._c0 = 20.323139721
        self._c1 = -78.577991460
        self._c2 = -7.0671612519
        self._c3 = 27.744056480
        self._c4 = 0.75588609922
        self._c5 = 36.591813315
    
        # Run model to complete missing data and run validation
        self._check_main_particulars()
        self._define_parameters()
        self._internal_validation()
        
        
    @classmethod
    def create_new_ship(cls, Length, Beam, Type='motorship', Cargo='dry', 
                        Hull='double',Te=None, Td=None, LSW=None, 
                        DWTfw=None, Force=None):
        """
        Class method to create a new Ship object with optional parameters.
        
        Parameters
        ----------
        Length : float
            Length overall in meters.
        Beam : float
            Beam molded in meters.
        Type : str, optional
            Type of the ship ('motorship', 'dump barge, or 'coupling barge'), 
            default is 'motorship'.
        Cargo : str, optional
            Type of cargo ('dry', 'container', or 'liquid'), default is 'dry'.
        Hull : str, optional
            Hull type ('single' or 'double'), default is 'double'.
        Te : float, optional
            Empty equal loaded draught in meters.
        Td : float, optional
            Design draught in meters.
        LSW : float, optional
            Light ship weight in metric tonnes.
        DWTfw : float, optional
            Deadweight in metric tonnes in fresh water with density 1.
        Force : str, optional
            Optional parameter to force the use of Hekkenberg (2013) instead 
            of Van Dorsser et al. (2020), default is None.
            
        Returns
        -------
        Ship
            An instance of the Ship class with the specified properties.
        """
        new_ship = cls(Length, Beam, Type, Cargo, Hull, Te, Td, LSW, DWTfw, 
                       Force)
        
        return new_ship
        
    
    def _check_main_particulars(self):
        """
        This functions checks if the input data is within the model boundaries.
        If not, cortrections are made to Hull, Cargo and Type propterties. For
        other properties only a notofication is made.
        """
        
        # Check if Length is within model boundaries
        if self.Length is not None:
            if self.Length < 40 or self.Length > 185:
                self.Note.append('Length outside model boundaries' +
                             '. Model supports length from 40 to 185 m.')
        
        # Check if Beam is within model boundaries    
        if self.Beam is not None:
            if self.Beam < 5 or self.Beam > 25:
                self.Note.append('Beam outside model boundaries' +
                             '. Model supports beam from 5 to 25 m.')
        
        # Check if Te (empty draught) is a positive number
        if self.Te is not None:
            if self.Te <= 0:
                self.note.append('Incorrect empty draught. Should be a ' +
                                 'positive number.')   
        
        # Check if Td (design draught) is within model boundaries
        if self.Td is not None:
            if self.Td < 1.5 or self.Td > 5.0:
                self.Note.append('Design draught outside model boundaties' +
                             '. Model supports draughts from 1.5 to 4.5m')
                self.Note.append('for dry cargo ships and up to 5.0m for' +
                                 'Tankers and Dumb Barges.')

        # Check if LSW (light ship weight) is a positive number
        if self.LSW is not None:
            if self.LSW <= 0:
                self.Note.append('Incorrect light ship weight. Should be a ' +
                                 'positive number.')   
        
        # Check if DWTfw (deadweight tonnage) is a positive number
        if self.DWTfw is not None:
            if self.DWTfw <= 0:
                self.Note.append('Incorrect deadweight tonnage. Should be a ' +
                                 'positive number.')               
        
        # Set default Hull type if not specified, and validate the input
        if self.Hull is None:
            self.Hull = 'double'
            self.Note.append("Hull type was not specified. Set tot default" +
                             " value 'double'. Other option is 'single'.")
        else:
            if not self.Hull in ('single', 'double'):
                self.Note.append("Incorrect hull type specified. Use 'single'"
                                 + "or 'double'.")                

        # Set default Cargo type if not specified, and validate the input
        if self.Cargo is None:
            self.Cargo = 'dry'
            self.Note.append("Cargo type was not specified. Set tot default" +
                             " value 'dry'. Other options are 'container', " +
                             "or 'liquid'.")
        else:
            if not self.Cargo in ('dry', 'container', 'liquid'):
                self.Note.append("Incorrect cargo type specified. Use 'dry'"
                                 + ", 'container', or 'liquid'.")  
        
        # Set default Ship Type if not specified, and validate the input
        if self.Type is None:
            self.Type = 'motorship'
            self.Note.append("Ship type was not specified. Set tot default" +
                             " value 'motorship'. Other options are 'dump " + 
                             "barge', or 'coupling barge'.")                
        
        if self.Type is not None:
            if not self.Type in ('motorship', 'dump barge', 'coupling barge'):
                self.Note.append("Incorrect ship type specified. Use "
                       + "'motorship', 'dump barge', or 'coupling barge'.") 
         
        # Additional check for Td if it exceeds 4.5m
        if self.Td is not None:
            if self.Td > 4.5:  
                if self.Cargo == 'dry' and self.Type != 'dump barge':
                    self.Note.append('Design draught outside model boundaties'+
                        '. Model supports draughts from 1.5 to 4.5m for')
                    self.Note.append(' dry cargo ships and coupling barges.')
                    
        return
    
        
    def _define_parameters(self):
        """
        Determine and set the required parameters based on known input data.
        
        This function looks up the appropriate method for setting the 
        parameters based on the known input data. It uses different solutions 
        depending on which parameters are provided.
        """
        
        # solve extended model for known parameters
        if not None in (self.Te, self.Td, self.LSW, self.DWTfw):
            self._solution01()
        elif not None in (self.Td, self.LSW, self.DWTfw):
            self._solution02()
        elif not None in (self.Te, self.LSW, self.DWTfw):
            self._solution03()
        elif not None in (self.Te, self.Td, self.DWTfw):  
            self._solution04()
        elif not None in (self.Te, self.Td, self.LSW):  
            self._solution05()
        elif not None in (self.LSW, self.DWTfw):  
            self._solution06()
        elif not None in (self.Te, self.LSW):  
            self._solution07()
        elif not None in (self.Td, self.LSW):
            self._solution08()
        elif not None in (self.DWTfw, self.Td):
            self._solution09()
        elif not None in (self.DWTfw, self.Te):
            self._solution10()     
        elif not None in (self.Te, self.Td):
            self._solution11() 
        elif self.Te is not None:
            self._solution12()
        elif self.LSW is not None:
            self._solution13()
        elif self.DWTfw is not None:
            self._solution14()
        elif self.Td is not None:
            self._solution15()            
        else:
            self._solution16()
        
        # define simple model parameters
        self._set_abc()
        
    def _set_abc(self):
        """
        Define parameters for the simplified model.
        
        This method calculates the parameters 'a', 'b', and 'c' based on
        other calculated values and constants of relative capacity model 
        of Van Dorsser et al. (2020, table 3).
        """
        self.a = self.alpha * self._c4
        self.b = self.alpha * (self._c3 + self._c5 * self.Te)
        self.c = (self.alpha * (self._c0 + self._c1 * self.Te +
                               self._c2 * self.Te**2) + self.epsilon)
                                                               
    def _solution01(self): 
        """
        Solution when Te, Td, LSW, and DWTfw are known.
        
        This method sets the alpha, beta, epsilon, Cb, Cwl, and Awl parameters.
        """
        self._r1_1() # set alpha
        self._r1_2() # set beta
        self._r1_3() # set epsilon
        self._r1_4() # set Cb
        self._r1_5() # set Cwl
        self._r1_6() # set Awl
        
    def _solution02(self): 
        """
        Solution when Td, LSW, and DWTfw are known.
        
        This method sets the beta, Te, alpha, epsilon, Cb, Cwl, and Awl 
        parameters.
        """
        self.beta = self._default_beta() # set beta
        self._r2_1() # Set Te
        self._r1_1() # set alpha
        self._r1_3() # set epsilon
        self._r1_4() # set Cb
        self._r1_5() # set Cwl
        self._r1_6() # set Awl
        
    def _solution03(self): 
        """
        Solution when Te, LSW, and DWTfw are known.
        
        This method sets the beta, alpha, epsilon, Cb, Cwl, Awl, and Td 
        parameters.
        """ 
        self.beta = self._default_beta() # set beta
        self._r3_1() # set alpha
        self._r1_3() # set epsilon
        self._r1_4() # set Cb
        self._r1_5() # set Cwl
        self._r1_6() # set Awl
        self._r3_2() # set Td      
        
    def _solution04(self):
        """
        Solution when Te, Td, and DWTfw are known.
        
        This method sets the beta, alpha, LSW, epsilon, Cb, Cwl, and Awl 
        parameters.
        """
        self.beta = self._default_beta() # set beta
        self._r1_1() # set alpha
        self._r4_1() # set LSW
        self._r1_3() # set epsilon
        self._r1_4() # set Cb
        self._r1_5() # set Cwl
        self._r1_6() # set Awl
        
    def _solution05(self):
        """
        Solution when Te, Td, and LSW are known.
        
        This method sets the beta, Cb, Cwl, Awl, alpha, epsilon, and DWTfw 
        parameters.
        """
        self.beta = self._default_beta() # set beta
        self._r1_4() # set Cb
        self._r1_5() # set Cwl
        self._r1_6() # set Awl   
        self._r5_1() # set alpha
        self._r1_3() # set epsilon
        self._r5_2() # set DWTfw
        
    def _solution06(self): 
        """
        Solution when LSW and DWTfw are known.
        
        This method sets the beta, Cb, Te, Cwl, Awl, alpha, epsilon, and Td 
        parameters.
        """
        self.beta = self._default_beta() # set beta
        self.Cb = self._default_cb() # set Cb
        self._r6_1() # set Te
        self._r1_5() # set Cwl
        self._r1_6() # set Awl
        self._r5_1() # set alpha
        self._r1_3() # set epsilon
        self._r3_2() # set Td   

    def _solution07(self):
        """
        Solution when LSW and Te are known.
        
        This method sets the beta, Cb, Cwl, Awl, alpha, epsilon, Td, and DWTfw 
        parameters.
        """
        self.beta = self._default_beta() # set beta
        self._r1_4() # set Cb
        self._r1_5() # set Cwl
        self._r1_6() # set Awl          
        self._r5_1() # set alpha
        self._r1_3() # set epsilon
        self._r7_1() # set Td   
        self._r5_2() # set DWTfw
        
    def _solution08(self): 
        """
        Solution when LSW and Td are known.
        
        This method sets the beta, Cb, Te, Cwl, Awl, alpha, epsilon, and DWTfw 
        parameters.
        """
        self.beta = self._default_beta() # set beta
        self.Cb = self._default_cb() # set Cb
        self._r6_1() # set Te
        self._r1_5() # set Cwl
        self._r1_6() # set Awl
        self._r5_1() # set alpha
        self._r1_3() # set epsilon
        self._r5_2() # set DWTfw       
        
    def _solution09(self):
        """
        Solution when DWTfw and Td are known.
        
        This method sets the beta, Cb, Cwl, Awl, Te, alpha, epsilon, and LSW 
        parameters.
        """
        self.beta = self._default_beta() # set beta
        self.Cb = self._default_cb() # set Cb
        self._r1_5() # set Cwl
        self._r1_6() # set Awl
        self._r9_1() # set Te
        self._r5_1() # set alpha
        self._r1_3() # set epsilon  
        self._r4_1() # set LSW
        
    def _solution10(self): 
        """
        Solution when DWTfw and Te are known.
        
        This method sets the beta, Cb, Cwl, Awl, LSW, alpha, epsilon, and Td 
        parameters.
        """
        self.beta = self._default_beta() # set beta
        self.Cb = self._default_cb() # set Cb
        self._r1_5() # set Cwl
        self._r1_6() # set Awl
        self._r10_1() # set LSW
        self._r5_1() # set alpha
        self._r1_3() # set epsilon        
        self._r3_2() # set Td           
        
    def _solution11(self): 
        """
        Solution when Te and Td are known.
        
        This method sets the beta, Cb, Cwl, Awl, LSW, alpha, epsilon, and 
        DWTfw parameters.
        """
        self.beta = self._default_beta() # set beta
        self.Cb = self._default_cb() # set Cb
        self._r1_5() # set Cwl
        self._r1_6() # set Awl
        self._r10_1() # set LSW
        self._r5_1() # set alpha
        self._r1_3() # set epsilon    
        self._r5_2() # set DWTfw   
        
    def _solution12(self): 
        """
        Solution when Te is known.
        
        This method sets the beta, Cb, Td, Cwl, Awl, LSW, alpha, epsilon, and 
        DWTfw parameters.
        """
        self.beta = self._default_beta() # set beta
        self.Cb = self._default_cb() # set Cb
        self._r7_1() # set Td   
        self._r1_5() # set Cwl
        self._r1_6() # set Awl
        self._r10_1() # set LSW
        self._r5_1() # set alpha
        self._r1_3() # set epsilon    
        self._r5_2() # set DWTfw   

    def _solution13(self): 
        """
        Solution when LSW is known.
        
        This method sets the beta, Cb, Td, Cwl, Awl, Te, alpha, epsilon, and 
        DWTfw parameters.
        """
        self.beta = self._default_beta() # set beta
        self.Cb = self._default_cb() # set Cb
        self._r7_1() # set Td  
        self._r1_5() # set Cwl
        self._r1_6() # set Awl
        self._r12_1() # set Te
        self._r5_1() # set alpha
        self._r1_3() # set epsilon    
        self._r5_2() # set DWTfw           
        
    def _solution14(self): 
        """
        Solution when DWTfw is known.
        
        Td does not have to be calculated beforehand and passed as 
        indicated in step 14.1 of the publication, as it is incorporated in 
        the applied functions for estimating the LSW from Annex C, D, and E.
        
        This method sets the LSW and resolves the remaining parameters.
        """
        self._LSW_exogenous() # Set LSW confirm step 14.2
        self._solution06() # resolve the remaining items     
        
    def _solution15(self): 
        """
        Solution when Td is known.
        
        This method sets the LSW and resolves the remaining parameters.
        """
        self._LSW_exogenous() # Set LSW confirm step 14.2
        self._solution08() # resolve the remaining items     

    def _solution16(self): 
        """
        Solution when no additional data is known.
        
        This method sets Td and LSW, and resolves the remaining parameters.
        """
        self.Td = self._Td_exogenous()
        self._LSW_exogenous() # Set LSW confirm step 14.2
        self._solution08() # resolve the remaining items   
        
    def _r1_1(self):
        """
        Set alpha based on result R1.1.
        """
        self.alpha = (self.DWTfw / ((self._c3 + self._c4 * self.Td +
                        self._c4 * self.Te + self._c5 * self.Te) *
                        (self.Td - self.Te)))
    
    def _r1_2(self):
        """
        Set beta based on result R1.2.
        """
        self.beta = (self.LSW / (self.alpha * self.Te * (self._c3 + 2 * 
                     self._c4 * self.Te + self._c5 * self.Te)))
        
    def _r1_3(self):
        """
        Set epsilon based on result R1.3.
        """
        self.epsilon = (-self.alpha * (self._c0 + self._c1 * self.Te +
                       self._c2 * (self.Te**2) + self._c3 * self.Te +
                       self._c4 * (self.Te**2) + self._c5 * (self.Te**2)))
        
    def _r1_4(self):
        """
        Set block coefficient (Cb) based on result R1.4.
        """
        self.Cb = self.LSW / (self.Length * self.Beam * self.Te)
    
    def _r1_5(self):
        """
        Set waterline coefficient (Cwl) based on result R1.5.
        """
        self.Cwl = self.Cb / self.beta
    
    def _r1_6(self):
        """
        Set waterline area (Awl) based on result R1.6.
        """
        self.Awl = self.Cwl * self.Length * self.Beam
    
    def _r2_1(self):
        """
        Set empty draft (Te) based on result 2.1.
        """
        a = ((2 * self._c4 + self._c5) * self.beta * self.DWTfw +
            (self._c4 + self._c5) * self.LSW)
        b = (self.beta * self._c3 * self.DWTfw + (self._c3 - (self._c5 *
            self.Td)) * self.LSW)
        c = -(self._c3 * self.Td + self._c4 * (self.Td**2)) * self.LSW
        self.Te = (-b + ((b**2)-4*a*c)**0.5)/(2*a)
        
    def _r3_1(self):
        """
        Set alpha based on result R3.1.
        """
        self.alpha =(self.LSW / (self.beta * self.Te * (self._c3 + 2 * \
                    self._c4 * self.Te + self._c5 * self.Te)))
            
    def _r3_2(self):
        """
        Set design draught (Td) based on result R3.2.
        """
        a = self._c4
        b = self._c3 + self._c5 * self.Te
        c = (self._c0 + self._c1 * self.Te + self._c2 * self.Te**2 + 
             (self.epsilon - self.DWTfw) / self.alpha)
        self.Td = (-b + ((b**2)-4*a*c)**0.5)/(2*a)
        
    def _r4_1(self):
        """
        Set light ship weight (LSW) based on result R4.1.
        """
        self.LSW =(self.alpha * self.beta * self.Te * (self._c3 +
                   2 * self._c4 * self.Te + self._c5 * self.Te))
        
    def _r5_1(self):
        """
        Set alpha based on result R5.1.
        """
        self.alpha =(self.Awl / (self._c3 + 2 * self._c4 * self.Te +
                     self._c5 * self.Te))
        
    def _r5_2(self):
        """
        Set deadweight in fresh water (DWTfw) based on result R5.2.
        """
        self.DWTfw = (self.alpha * (self._c0 + self._c1 * self.Te +
                      self._c2 * self.Te**2 + self._c3 * self.Td +
                      self._c4 * self.Td**2 + self._c5 * self.Te * self.Td)
                      + self.epsilon)
        
    def _r6_1(self):
        """
        Set empty draught (Te) based on result R6.1.
        """      
        self.Te = self.LSW / (self.Cb * self.Length * self.Beam) # set Te
                
    def _r7_1(self): 
        """
        Set design draught (Td) based on result R7.1 using exogenous method of 
        van Dorsser et al. (2020).
        """
        self.Td = self._Td_exogenous()
               
    def _r9_1(self):
        """
        Set empty draught (Te) based on result R9.1.
        """
        a = (self._c4 + self._c5) * self.Awl
        b = ((2 * self._c4 + self._c5) * self.DWTfw + (self._c3 - self._c5
            * self.Td) * self.Awl)
        c = (self._c3 * self.DWTfw - self._c3 * self.Td * self.Awl -
             self._c4 * (self.Td**2) * self.Awl)
        self.Te = (-b + ((b**2)-4*a*c)**0.5)/(2*a) 
        
    def _r10_1(self):
        """
        Set light ship weight (LSW) based on result R10.1.
        """
        self.LSW = self.beta * self.Awl * self.Te
        
    def _r12_1(self):
        """
        Set empty draught (Te) based on result R12.1.
        """
        self.Te = self.LSW / (self.beta * self.Awl)
        
    def _default_beta(self):
        """
        Set the default beta value based on the ship type.
        
        This method returns the default shape factor (beta) for the ship 
        based on its type. It raises an error if the ship type is incorrect.
        
        Returns
        -------
        beta : float
            The default beta value for the specified ship type.
        """
        if self.Type == 'motorship':
            beta = 0.926
        elif self.Type == 'dump barge':
            beta = 0.961
        elif self.Type == 'coupling barge':
            beta = 0.932
        else:
            print('Please enter correct ship type')
            beta = None
        return beta


    def _default_cb(self):
        """
        Set the default block coefficient (Cb) based on the ship type.
        
        This method returns the default block coefficient (Cb) for the ship 
        based on its type. It raises an error if the ship type is incorrect.
        
        Returns
        -------
        Cb : float
            The default block coefficient for the specified ship type.
        """
        if self.Type == 'motorship':
            Cb = 0.811
        elif self.Type == 'dump barge':
            Cb = 0.866
        elif self.Type == 'coupling barge':
            Cb = 0.848
        else:
            print('Please enter correct ship type')
            Cb = None
        return Cb
                      
    
    def _Td_exogenous(self, margin=0.05):
        """
        Estimate the design draught (Td) based on Van Dorsser et al. (2020), 
        taking into account the lower and upper bounds defined in Annex C.
        
        If it concerns a 'dumb barge' a different method is applied than
        for motorbarges.  
        
        Parameters
        ----------
        margin : float, optional
            Margin to extend the lower and upper bounds, default is 0.05.
            This inplies that lower bound is reduced and upper bound is 
            increased by 5%.
        
        Returns
        -------
        Td : float
            The estimated design draught for the ship.
        """
        
        # For 'dumb barge' use dumb barge parameters
        if self.Type == 'dump barge':
            Td = (1.33653798980786E+00 + self.Length**0.3 * self.Beam**1.8 
                 * 9.00523844393539E-03)
            
            # Apply lower and upper limits of dataset in Table C5
            Td = max(2.80, Td)
            Td = min(4.95, Td)
            return Td

        # For 'coupling barge' use dumb barge parameters
        if self.Type == 'dump barge' or self.Type == 'coupling barge':
            Td = (1.33653798980786E+00 + self.Length**0.3 * self.Beam**1.8 
                 * 9.00523844393539E-03)
            self.Note.append('Design draught based om method for dumb ' +
                'barges. Suggested to match draught of coupled motorship.')
            
            # Apply lower and upper limits of dataset in Table C5
            Td = max((1 - margin) * 2.47, Td)
            Td = min((1 + margin) * 4.46, Td, 5.0)
            return Td       
        
        # For containerships
        if self.Cargo == 'container':
            Td = (1.72441533707496E+00 + (self.Length**0.4) * (self.Beam**0.6)   
                 * 6.29023055595728E-02)
            
            # Apply lower and upper limits of dataset in Table C5
            Td = max((1 - margin) * 2.79, Td)
            Td = min((1 + margin) * 4.27, Td, 5.0)
            return Td  
            
        # For dry cargo ships
        if self.Cargo == 'dry':
            Td = (2.27671792457769E+00 + (self.Length**0.7) * (self.Beam**2.6)   
                 * 7.73988615284090E-05)

            # Apply lower and upper limits of dataset in Table C5
            Td = max((1 - margin) * 2.17, Td)
            Td = min((1 + margin) * 3.72, Td)
            return Td  
        
        # For tankers
        if self.Cargo == 'liquid':
            Td = (-5.94593089048782E+00 + (self.Length**0.1) * (self.Beam**0.3)    
                 * 2.84385608765655E+00)
            
            # Apply lower and upper limits of dataset in Table C5
            Td = max((1 - margin) * 2.82, Td)
            Td = min((1 + margin) * 5.01, Td, 5.0)
            return Td  
    
    
    def _LSW_exogenous(self):
        """
        Define the Light Ship Weight (LSW) based on compliance with Van Dorsser 
        et al. (2020) and Hekkenberg (2013) models.
        
        This method selects the appropriate method to estimate LSW depending on 
        the ship type, hull type, and whether the parameters fall within the 
        validated dataset ranges of common ship types. 
        
        If parameters are within the range of common ship types the method of
        Van Dorsser et al. (2020) is applied, if not Hekkenberg (2013) is used.
        """
        
        # For 'dump barge' and 'coupling barge', use new method from Annex E
        # This method does not require verification of fit in Table C5 range
        if self.Type == 'dump barge' or self.Type == 'coupling barge':
            self.LSW = self._LSW_AnnexE()
        
        # Deal with single hull type vessels
        if self.Hull == 'single': 
            
            if self.Cargo == 'container':
                self.Hull == 'double'
                self.Note.append('Warning: single hull containership is not '+
                             'supported by the model. Hull set to double.')
            
            # Check if single hull ship fits within validated dataset            
            if hasattr(self, 'Te') and self.Te is not None:
                # include Te in validation
                supported = (self._check_length_VD() and self._check_beam_VD() and
                             self._check_Td_VD() and self._check_LB_VD())
            else:
                # exclude Te from validation
                supported = (self._check_length_VD() and self._check_beam_VD() and
                            self._check_LB_VD())
                
            # If the model of van Dorsser does not support single hull ships of
            # the indicated size and type than set hull type to double
            if not supported:
                if self.Cargo == 'dry':
                    self.Hull = 'double'
                    self.Note.append(
                        'Warning: single hull dry cargo not supported ' +
                        'by the model for this size. Hull set to double.') 
                elif self.Cargo == 'liquid':
                    self.Hull = 'double'
                    self.Note.append(
                        'Warning: single hull tanker not supported ' +
                        'by the model for this size. Hull set to double.')                 
        
        # Check if hull type is well specified, else replace by 'double'
        if not self.Hull in ('single', 'double'):
            # Hull type not properly specified. Set to double
            self.Hull = 'double'
            self.Note.append(
                'Warning: invalid hull type specification. Hull type set ' +
                'to double.')       
           
        # Continue to check if double hull items fit within validated dataset            
        if hasattr(self, 'Te') and self.Te is not None:
            # Include Te in validation
            supported = (self._check_length_VD() and self._check_beam_VD() and
                         self._check_Td_VD() and self._check_LB_VD())
        else:
            # Exclude Te from validation
            supported = (self._check_length_VD() and self._check_beam_VD() and
                        self._check_LB_VD())
   
        # Apply appropriate estimation method based on supported
        if supported and self.Force != 'HB':          
            
            # Use method of van Dorsser et al. (2020)
            self.LSW = self._LSW_AnnexC() 
            
        else:
            # Use method of Hekkenberg (2013)
            self.LSW = self._LSW_AnnexD() 
            
            
    def _check_length_VD(self, margin=0.05):
        """
        Check if the length fits within the empirical dataset of validated 
        ships.
        
        Parameters
        ----------
        margin : float, optional
            Margin to extend the lower and upper bounds, default is 0.05.
        
        Returns
        -------
        Fit : bool
            True if the length fits within the empirical dataset, False 
            otherwise.
        """
             
        if self.Hull == 'double':
            if self.Cargo == 'dry':
                Fit = (self.Length >= ((1-margin) * 84.88) and 
                             self.Length <= ((1+margin) * 135.00))
            if self.Cargo == 'container':
                Fit = (self.Length >= ((1-margin) * 63.00) and 
                             self.Length <= ((1+margin) * 135.00))
            if self.Cargo == 'liquid':
                Fit = (self.Length >= ((1-margin) * 85.59) and 
                             self.Length <= ((1+margin) * 135.00))
 
        elif self.Hull == 'single':
            if self.Cargo == 'dry':
                Fit = (self.Length >= ((1-margin) * 46.22) and 
                             self.Length <= ((1+margin) * 109.95))
            if self.Cargo == 'liquid':
                Fit = (self.Length >= ((1-margin) * 84.60) and 
                             self.Length <= ((1+margin) * 110.00))      
        return Fit
    
    
    def _check_beam_VD(self, margin=0.05):
        """
        Check if the beam fits within the empirical dataset of validated ships.
        
        Parameters
        ----------
        margin : float, optional
            Margin to extend the lower and upper bounds, default is 0.05.
        
        Returns
        -------
        Fit : bool
            True if the beam fits within the empirical dataset, False 
            otherwise.
        """
             
        if self.Hull == 'double':
            if self.Cargo == 'dry':
                Fit = (self.Beam >= ((1-margin) * 8.20) and 
                             self.Beam <= ((1+margin) * 11.45))
            if self.Cargo == 'container':
                Fit = (self.Beam >= ((1-margin) * 7.03) and 
                             self.Beam <= ((1+margin) * 17.10))
            if self.Cargo == 'liquid':
                Fit = (self.Beam >= ((1-margin) * 9.50) and 
                             self.Beam <= ((1+margin) * 17.55))
        elif self.Hull == 'single':
            if self.Cargo == 'dry':
                Fit = (self.Beam >= ((1-margin) * 6.33) and 
                             self.Beam <= ((1+margin) * 11.40))
            if self.Cargo == 'liquid':
                Fit = (self.Beam >= ((1-margin) * 9.50) and 
                             self.Beam <= ((1+margin) * 11.36))           
        return Fit
    
    
    def _check_Td_VD(self, margin=0.05):
        """
        Check if the design draught (Td) fits within the empirical dataset of
        validated ships.
        
        Parameters
        ----------
        margin : float, optional
            Margin to extend the lower and upper bounds, default is 0.05.
        
        Returns
        -------
        Fit : bool
            True if the design draught fits within the empirical dataset,
            False otherwise.
        """
             
        if self.Hull == 'double':
            if self.Cargo == 'dry':
                Fit = (self.Td >= ((1-margin) * 2.47) and 
                             self.Td <= ((1+margin) * 3.72))
            if self.cargo == 'container':
                Fit = (self.Td >= ((1-margin) * 2.79) and 
                             self.Td <= ((1+margin) * 4.27))
            if self.Cargo == 'liquid':
                Fit = (self.Td >= ((1-margin) * 2.89) and 
                             self.Td <= ((1+margin) * 5.01))
        elif self.Hull == 'single':
            if self.Cargo == 'dry':
                Fit = (self.Td >= ((1-margin) * 2.17) and 
                             self.Td <= ((1+margin) * 3.22))
            if self.Cargo == 'liquid':
                Fit = (self.Td >= ((1-margin) * 2.82) and 
                             self.Td <= ((1+margin) * 4.02))           
        return Fit    


    def _check_LB_VD(self, margin=0.05):
        """
        Check if the length-to-beam ratio (L/B) fits within the empirical 
        dataset of validated ships.
        
        Parameters
        ----------
        margin : float, optional
            Margin to extend the lower and upper bounds, default is 0.05.
        
        Returns
        -------
        Fit : bool
            True if the length-to-beam ratio fits within the empirical dataset, 
            False otherwise.
        """
                     
        if self.Hull == 'double':
            if self.Cargo == 'dry':
                Fit = ((self.Length/self.Beam) >= ((1-margin) * 7.8) and 
                             (self.Length/self.Beam) <= ((1+margin) * 11.8))
            if self.Cargo == 'container':
                Fit = ((self.Length/self.Beam) >= ((1-margin) * 7.5) and 
                             (self.Length/self.Beam) <= ((1+margin) * 11.8))
            if self.Cargo == 'liquid':
                Fit = ((self.Length/self.Beam) >= ((1-margin) * 7.5) and 
                             (self.Length/self.Beam) <= ((1+margin) * 11.7))
        elif self.Hull == 'single':
            if self.Cargo == 'dry':
                Fit = ((self.Length/self.Beam) >= ((1-margin) * 7.3) and 
                             (self.Length/self.Beam) <= ((1+margin) * 11.1))
            if self.Cargo == 'liquid':
                Fit = ((self.Length/self.Beam) >= ((1-margin) * 8.9) and 
                             (self.Length/self.Beam) <= ((1+margin) * 9.8))
            
        return Fit  
            
    
    def _LSW_AnnexC(self): 
        """
        Define the Light Ship Weight (LSW) based on the method of van Dorsser
        et al. (2020, Table 4).
        
        This method calculates the LSW using known Cb values and other 
        parameters.
        
        Returns
        -------
        LSW : float
            The calculated light ship weight.
        """
        
        # Set Td if not yet existing in self
        if hasattr(self, 'Td') and self.Td is not None:
            Td = self.Td
        else:
            Td = self._Td_exogenous()
        
        # Calculate Te based on provided formula
        Te = (7.57408209269816E-02 + 1.16150809917024E-01 * self.Beam +
              1.68659734940172E-02 * (self.Length * Td)/self.Beam +
              -2.74905653810124E-02 * (self.Length * self.Beam)**0.5 +
              -5.15012407436910E-05 * (self.Length * self.Beam * Td))
      
        # Apply dummy adjustment for dumb barges
        if self.Type != 'motorship':
            Te += -2.13542956265673E-01
        
        # Apply dummy adjustment for double hull barges
        elif self.Hull == 'double':
            if self.Cargo == 'dry' or self.Cargo == 'container':
                Te += 1.02575511529418E-01
            elif self.Cargo == 'liquid':
                Te += 2.42994352105025E-01
        
        # No dummy adjustment applied to single hull ships
        
        # Set Cb if not yet existing in self
        if hasattr(self, 'Cb') and self.Cb is not None:
            Cb = self.Cb
        else:
            Cb = self._default_cb()
        
        # Calculate LSW
        LSW = Cb * Te * self.Length * self.Beam
        
        return LSW
            
    
    def _LSW_AnnexD(self): 
        """
        Define the Light Ship Weight (LSW) based on the corrected method of 
        Hekkenberg (2013).
        
        This method calculates the LSW using a correction on accommodation
        and outfitting for small ships.
        
        Returns
        -------
        LSW : float
            The calculated light ship weight.
        """
        
        # Set Td if not yet existing in self
        if hasattr(self, 'Td') and self.Td is not None:
            Td = self.Td
        else:
            Td = self._Td_exogenous()
        
        # Define steel weight
        if self.Cargo == 'dry':
            
            # Define steelweight for transverse framing
            SW_trans = (-2.597E+01 + #c1
                   2.320E-01 * self.Length * self.Beam + # c2
                   -1.552E-03 * self.Length**2 * Td + # c3
                   4.444E-02 * self.Length * self.Beam * Td + # c4
                   8.134E-07 * self.Length**3.5 * self.Beam + # c5
                   1.024E+00 * self.Length**1.3 * Td**0.7 / self.Beam + # c6
                   7.691E+02 * 1/(self.Beam**2 * Td**1.5)) # c7
            
            # Define steelweight for longgitudinal framing
            SW_long = (4.985E+01 + #c1
                   2.290E-01 * self.Length * self.Beam + # c2
                   -1.234E-05 * self.Length**2 * Td + # c3
                   1.910E-02 * self.Length * self.Beam * Td + # c4
                   9.584E-07 * self.Length**3.5 * self.Beam + # c5
                   2.880E-01 * self.Length**1.3 * Td**0.7 / self.Beam + # c6
                   -1.066E+03 * 1/(self.Beam**2 * Td**1.5)) # c7    
            
            # Set steelweight as min of longitudinal and transverse framing
            SW = min(SW_trans, SW_long)
            
        elif self.Cargo == 'container':
            
            # Define steelweight for transverse framing
            SW_trans = (-2.200E+01 + #c1
                   2.540E-01 * self.Length * self.Beam + # c2
                   -1.975E-03 * self.Length**2 * Td + # c3
                   4.473E-02 * self.Length * self.Beam * Td + # c4
                   1.059E-06 * self.Length**3.5 * self.Beam + # c5
                   9.600E-01 * self.Length**1.3 * Td**0.7 / self.Beam + # c6
                   6.676E+02 * 1/(self.Beam**2 * Td**1.5)) # c7
            
            # Define steelweight for longgitudinal framing
            SW_long = (5.107E+01 + #c1
                   2.440E-01 * self.Length * self.Beam + # c2
                   -1.772E-04 * self.Length**2 * Td + # c3
                   1.588E-02 * self.Length * self.Beam * Td + # c4
                   1.100E-06 * self.Length**3.5 * self.Beam + # c5
                   3.120E-01 * self.Length**1.3 * Td**0.7 / self.Beam + # c6
                   -1.164E+03 * 1/(self.Beam**2 * Td**1.5)) # c7  
            
            # Set steelweight as min of longitudinal and transverse framing
            SW = min(SW_trans, SW_long)     
        elif self.Cargo == 'liquid':
            SW = (-4.220E+02 + #c1
                   -7.694E-04 * self.Length**2 * Td + # c3
                   7.311E-02 * self.Length * self.Beam * Td + # c4
                   1.157E-06 * self.Length**3.5 * self.Beam + # c5
                   -7.922E+03 * 1/((self.Length * self.Beam * Td)**0.5)) # c7
        else:
            SW = None
            self.Note = 'Invalid cargo type definition'
        
        # Define accomodation, machinery, equipment and outfitting weight
        AMEO = (4.325E-01 * max(self.Length/4 * (self.Beam-2), 100) + # c9
                2.804E+01 + # c10
                4.605E+00 * Td + # c11
                2.097E-02 * self.Length * self.Beam + # c12
                2.240E-03 * self.Length * self.Beam * Td + # c13
               -4.258E+05 * 1/(self.Length**3)) # c14
        
        # Make sure AMEO is not larger than 0.25 x SW
        AMEO = min(AMEO, 0.25 * SW)
        
        # Define piping weight
        if self.Cargo == 'dry' or self.Cargo == 'container':
            PIPE = (-2.723E+00 + # c15
                    6.232E-02 * self.Length + # c16
                    5.048E-02 * self.Beam + # c17
                    9.968E-02 * self.Td + # c18
                    1.343E-04 * self.Length * self.Beam * Td) # c19
        elif self.Cargo == 'liquid':
            PIPE = (-3,949E+00 + # c15
                    8,191E-02 * self.Length + # c16
                   -4,407E-01 * self.Beam + # c17
                    1,065E-03 * self.Length * self.Beam * Td + # c19
                    6,966E-02 * self.Length**0.6 * self.Beam + # c20
                    1,228E+04 * self.Beam / (self.Length**3))
        else:
            PIPE = None
            
        LSW = SW + AMEO + PIPE
        
        return LSW


    def _LSW_AnnexE(self): 
        """
        Define the Light Ship Weight (LSW) using the new method for dumb and 
        coupling barges.
        
        This method is based on the relation to steel weight from Hekkenberg 
        (2013) and LBTd.
        
        Returns
        -------
        LSW : float
            The calculated light ship weight.
        """

        # Method has not been defined and validated for tank barges
        if self.Cargo == 'liquid':
            self.Note = ('Warning: dry cargo method applied to define LSW of'+
            ' tank barges.')
            
        # Set Td if not yet existing in self
        if hasattr(self, 'Td') and self.Td is not None:
            Td = self.Td
        else:
            Td = self._Td_exogenous()
        
        # Define steelweight for transverse framing
        SW_trans = (-2.597E+01 + #c1
               2.320E-01 * self.Length * self.Beam + # c2
               -1.552E-03 * self.Length**2 * Td + # c3
               4.444E-02 * self.Length * self.Beam * Td + # c4
               8.134E-07 * self.Length**3.5 * self.Beam + # c5
               1.024E+00 * self.Length**1.3 * Td**0.7 / self.Beam + # c6
               7.691E+02 * 1/(self.Beam**2 * Td**1.5)) # c7
        # Define steelweight for longgitudinal framing
        SW_long = (4.985E+01 + #c1
               2.290E-01 * self.Length * self.Beam + # c2
               -1.234E-05 * self.Length**2 * Td + # c3
               1.910E-02 * self.Length * self.Beam * Td + # c4
               9.584E-07 * self.Length**3.5 * self.Beam + # c5
               2.880E-01 * self.Length**1.3 * Td**0.7 / self.Beam + # c6
               -1.066E+03 * 1/(self.Beam**2 * Td**1.5)) # c7        
        # Set steelweight as min of longitudinal and transverse framing
        SW = min(SW_trans, SW_long)    

        LSW = 0.473 * SW + 0.09254 * self.Length * self.Beam * Td

        return LSW           


    def _internal_validation(self):
        """
        Perform internal validation of the ship data to ensure epsilon, beta, 
        and Cb values are within acceptable limits.
        
        This procedure checks if the calculated values of epsilon, beta, and Cb
        are within the acceptable ranges based on the type of ship.
    
        Returns
        -------
        None
        """
        
        # Check beta and Cb
        if self.Type == 'motorship':
            self.check_beta = 0.863 <= self.beta <= 0.989
            self.check_Cb = 0.734 <= self.Cb <= 0.887
        elif self.Type == 'dump barge':
            self.check_beta = 0.917 <= self.beta <= 1.000
            self.check_Cb = 0.797 <= self.Cb <= 0.936
        elif self.Type == 'coupling barge':
            self.check_beta = 0.863 <= self.beta <= 1.000
            self.check_Cb = 0.734 <= self.Cb <= 0.936
        else:
            self.Note = 'Incorrect barge type assigned'
            self.check_beta = None
            self.check_Cb = None
            
        # Check epsilon
        alpha = abs(self.alpha)
        self.check_epsilon = -5 * alpha <= self.epsilon <= 5 * alpha
        
        # Internal validation
        self.Validation_ok = (self.check_Cb and self.check_beta and 
                              self.check_epsilon)
        
        return
       
        
    def capacity_at_draught(self, actual_draught, density_water=1):
        """
        Define deadweight capacity for a given water density if draught is 
        within the design boundaries. Else, return None.
        
        Parameters
        ----------
        actual_draught : float
            The actual draught of the ship.
        density_water : float, optional
            The density of water, default is 1.
        
        Returns
        -------
        DWT : float or None
            The deadweight capacity at the given draught and water density,
            or None if the draught is outside the design boundaries.
        """
        if actual_draught >= self.Te and actual_draught <= self.Td:
            Ta = actual_draught
            rho = density_water
            CAP_Ta = self.a * Ta**2 + self.b * Ta + self.c
            DWT = (rho - 1) * self.LSW + rho * CAP_Ta
        else:
            print('Actual_draught does not fit within ship dimensions.')
            DWT = None
            
        return DWT
    
    
    def displacement_at_draught(self, actual_draught, density_water=1):
        """
        Define displacement for a given water density if draught is within the 
        design boundaries. Else, return None.
        
        Parameters
        ----------
        actual_draught : float
            The actual draught of the ship.
        density_water : float, optional
            The density of water, default is 1.
        
        Returns
        -------
        DISPL : float or None
            The displacement at the given draught and water density,
            or None if the draught is outside the design boundaries.
        """          
        DWT = self.capacity_at_draught(actual_draught, density_water)
        
        if DWT is not None:
            DISPL = DWT + self.LSW
        else:
            DISPL = None
            
        return DISPL
    
    
    def draught_at_capacity(self, actual_capacity, density_water=1):
        """
        Define draught at a given deadweight capacity and water density.
        Using the following relation:
            
        CAP = (DWT_act_sw - (rho - 1) * LSW) / rho
        T = (-b + sqrt(b^2 - 4 a c + 4 a CAP))/(2 a)
        
        Parameters
        ----------
        actual_capacity : float
            The actual deadweight capacity of the ship.
        density_water : float, optional
            The density of water, default is 1.
        
        Returns
        -------
        T : float or None
            The draught at the given deadweight capacity and water density,
            or None if the capacity is outside the ship's design boundaries.
        """
        
        # Gebruik rho voor de dichtheid
        rho = density_water

        # Definieer maximale capaciteit bij gegeven dichtheid
        DWTmax = self.capacity_at_draught(self.Td, density_water)
        
        if 0 <= actual_capacity <= DWTmax:
            DWTact_sw = actual_capacity
            CAP = (DWTact_sw - (rho - 1) * self.LSW) / rho
            T = ((-self.b + (self.b**2 - 4*self.a*self.c + 4*self.a *
                           CAP)**0.5) / (2*self.a))
            return T
        else:
            print('Actual_capacity does not fit within ship dimensions.')
            return None
        
         
    def print_properties(self, ship_name='Ship'):
        """
        Print the ship properties.
        
        This function prints out the various properties and parameters of the ship.
        
        Parameters
        ----------
        ship_name : str, optional
            The name of the ship, default is 'Ship'.
        
        Returns
        -------
        None
        """
        print(ship_name, 'details:')
        print(f"Type: {self.Type}") 
        print(f"Cargo: {self.Cargo} cargo") 
        print(f"Hull: {self.Hull} hull")
        print()
        print(f"Length: {round(self.Length, 2)} meter")
        print(f"Beam: {round(self.Beam, 2)} meter")
        print(f"Te: {round(self.Te, 2)} meter") 
        print(f"Td: {round(self.Td, 2)} meter") 
        print(f"LSW: {round(self.LSW, 1)} ton")
        print(f"DWTfw: {round(self.DWTfw, 1)} ton")
        print()
        print(f"Cwl: {round(self.Cwl, 3)} [-]")
        print(f"Awl: {round(self.Awl, 2)} m2")
        print(f"Cb: {round(self.Cb, 3)} [-]")
        print(f"beta: {round(self.beta, 3)} [-]")
        print(f"alpha: {round(self.alpha, 2)} mton/index point")
        print(f"epsilon: {round(self.epsilon, 1)} ton")
        print()
        print('DWTfw = a * T^2 + b * T + c')
        print('with T: actual draught')
        print(f"a:  {self.a:.10e} mton/m2")
        print(f"b:  {self.b:.10e} mton/m")
        print(f"c: {self.c:.10e} mton") 
        if len(self.Note) > 0:
            print('Notes:')
            for row in self.Note:
                print(row)  
        print()
        print('Check_Cb:',self.check_Cb)
        print('Check_beta', self.check_beta)
        print('Check_epsilon:', self.check_epsilon)
        print('Internal validation ok:', self.Validation_ok)
        
    
    def draught_displacement_table(self, density_water=1):
        """
        Generate a DataFrame with draught, deadweight tonnage (DWT), 
        and displacement values.
    
        This function calculates the DWT and displacement for a range of 
        draught values, using the provided water density. The draught values 
        range from the vessel's empty draught (Te) to the design draught (Td),
        in steps of 0.01.
    
        Parameters
        ----------
        density_water : float, optional
            The density of water used in the calculations. The default is 1.
    
        Returns
        -------
        df : pandas.DataFrame
            A DataFrame with draught values as the index and two columns:
            - 'DWT': Deadweight tonnage values corresponding to the draught.
            - 'DISPL': Displacement values corresponding to the draught.
        """
        import math
        
        lower = math.ceil(self.Te * 100)/100
        upper = round(self.Td, 2)
        step = 0.01
        
        # Generate index values
        index_values = np.round(np.arange(lower, upper + step, step), 2)
        
        # Calculate DWT and DISPL values
        dwt_values = [self.capacity_at_draught(draught, density_water) 
                      for draught in index_values]
        displ_values = [self.displacement_at_draught(draught)
                        for draught in index_values]
        
        # Put data in a dataframe
        df = pd.DataFrame({'DWT': dwt_values, 
                           'DISPL': displ_values}, index=index_values)
        
        return df
    
    
    def plot_draught_displacement_figure(self, name='Ship', density_water=1):
        """
        Plot a figure showing the draught, deadweight tonnage (DWT), 
        and displacement values.
        
        This function generates a plot with draught values on the x-axis and 
        both DWT and displacement values on the y-axis. The data is retrieved 
        from the draught_displacement_table function. The plot includes a title
        and a subtitle indicating the water density used in the calculations.
        
        Parameters
        ----------
        name : str, optional
            The name of the ship to be included in the title of the plot. 
                The default is 'Ship'.
        density_water : float, optional
            The density of water used in the calculations. The default is 1.
        
        Returns
        -------
        None
        """
        
        # Get DataFrame with data to be plotted
        df = Ship.draught_displacement_table(self, density_water)
        
        # Plot de data
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['DWT'], label='DWT', color='b')
        plt.plot(df.index, df['DISPL'], label='DISPL', color='r')
        plt.xlabel('Draught (m)')
        plt.ylabel('Value (tonnes)')
        if name is not None:
            if len(name) > 0:
                title = name + ': DWT and DISPL vs Draught\n'
            else:
                title = 'DWT and DISPL vs Draught\n'
        
        plt.title(title, fontsize=16)  # Voeg een nieuwe regel toe voor de subtitel
        plt.suptitle('Water density = ' + str(density_water), y=0.92, 
                     fontsize=10)
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return 


def all_model_solutions(Length, Beam, Type, Cargo, Hull, Te, Td, LSW, DWTfw):
    """
    Generate a table with results for all model solutions in a pandas dataframe.
    
    This procedure provides a comprehensive table containing the results for 
    all model solutions. To obtain accurate results, all input data fields 
    must be entered and cannot be None.
    
    Parameters
    ----------
    Length : float
        Length overall in meters.
    Beam : float
        Beam molded in meters.
    Type : str
        Type of the ship ('motorship', 'dump barge', 'coupling barge').
    Cargo : str
        Type of cargo ('dry', 'container', 'liquid').
    Hull : str
        Hull type ('single' or 'double').
    Te : float
        Empty equal loaded draught in meters.
    Td : float
        Design draught in meters.
    LSW : float
        Light ship weight in metric tonnes.
    DWTfw : float
        Deadweight in fresh water with density 1.
    
    Returns
    -------
    solutions : pd.DataFrame
        A dataframe containing the results for all model solutions.
    """
    
    solutions = pd.DataFrame()
    
    # Set base values
    solutions.loc['Type', 'Real'] = Type
    solutions.loc['Cargo', 'Real'] = Cargo
    solutions.loc['Hull', 'Real'] = Hull
    solutions.loc['Length', 'Real'] = Length
    solutions.loc['Beam', 'Real'] = Beam
    solutions.loc['Te', 'Real'] = Te
    solutions.loc['Td', 'Real'] = Td
    solutions.loc['LSW', 'Real'] = LSW
    solutions.loc['DWTfw', 'Real'] = DWTfw    
    
    # set parameters for model solutions
    for sol in ['1','2','3','4','5', '6', '7', '8', '9', '10', '11', '12',
                   '13', '14', '14_hb', '15', '15_hb', '16', '16_hb']:
        
        # Set parameter Te for solutions
        if sol in ['1', '3', '4', '5', '7', '10', '11', '12']:
            _Te = Te
        else:
            _Te = None
            
        # Set parameter Td for solutions            
        if sol in ['1', '2', '4', '5', '8', '9', '11', '15']:
            _Td = Td
        else:
            _Td = None
            
        # Set parameter LSW for solutions 
        if sol in ['1', '2', '3', '5', '6', '7', '8', '13']:
            _LSW = LSW
        else:
            _LSW = None
            
        # Set parameter DWTfw for solutions 
        if sol in ['1', '2', '3', '4', '6', '9', '10', '14']:
            _DWTfw = DWTfw
        else:
            _DWTfw = None
            
        # Set forced use of HB model for solutions 
        if sol in ['14_hb', '15_hb', '16_hb']:
            _Force = 'HB'
        else:
            _Force = None
        
        # Get model output
        sol_ship = Ship.create_new_ship(Length, Beam, Type, Cargo, Hull,
                             _Te, _Td, _LSW, _DWTfw, _Force)
        
        # Set data
        solutions.loc['Type', sol] = sol_ship.Type
        solutions.loc['Cargo', sol] = sol_ship.Cargo
        solutions.loc['Hull', sol] = sol_ship.Hull
        solutions.loc['Length', sol] = sol_ship.Length
        solutions.loc['Beam', sol] = sol_ship.Beam
        solutions.loc['Te', sol] = sol_ship.Te
        solutions.loc['Td', sol] = sol_ship.Td
        solutions.loc['LSW', sol] = sol_ship.LSW
        solutions.loc['DWTfw', sol] = sol_ship.DWTfw
        
        solutions.loc['Cwl', sol] = sol_ship.Cwl
        solutions.loc['Awl', sol] = sol_ship.Awl
        solutions.loc['Cb', sol] = sol_ship.Cb
        solutions.loc['beta', sol] = sol_ship.beta
        solutions.loc['alpha', sol] = sol_ship.alpha
        solutions.loc['epsilon', sol] = sol_ship.epsilon
        
        solutions.loc['a', sol] = sol_ship.a
        solutions.loc['b', sol] = sol_ship.b
        solutions.loc['c', sol] = sol_ship.c

        solutions.loc['check_cb', sol] = sol_ship.check_Cb
        solutions.loc['check_beta', sol] = sol_ship.check_beta
        solutions.loc['check_epsilon', sol] = sol_ship.check_epsilon    
        
    return solutions


# Example of hou to use file
# Ship = Ship.create_new_ship(110, 11.45, 'motorship', 'container', 'double')
# Ship.print_properties()
# Ship.draught_displacement_table()
# Ship.plot_draught_displacement_figure()