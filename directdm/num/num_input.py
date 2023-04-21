#!/usr/bin/env python3

import sys
import numpy as np
import warnings
from directdm.run import rge


#----------------------------------------------#
#                                              #
# Numerical input. All masses in units of GeV. #
#                                              #
#----------------------------------------------#

# Nov 14, 2020: Updated from PDG 2020 and most recent lattice results


class Num_input(object):
    def __init__(self, my_input_dict=None):
        """ The numerical input for DirectDM

        Default values can be overriden by providing the optional 
        dictionary 'my_input_dict' which specifies the numerical 
        values of (a subset of) input parameters.
        """

        self.input_parameters = {}

        ### Couplings ###

        # The strong coupling constant
        self.input_parameters['asMZ'] = 0.1179

        # The Fermi constant
        self.input_parameters['GF'] = 1.1663787e-5

        # The inverse of QED alpha @ MZ
        self.input_parameters['aMZinv'] = 127.952

        # The inverse of QED alpha @ mtau
        self.input_parameters['amtauinv'] = 133.472

        # The inverse of QED alpha at low scales
        self.input_parameters['alowinv'] = 137.035999084

        # sin-squared of the weak mixing angle (MS-bar)
        self.input_parameters['sw2_MSbar'] = 0.23121


        ### Boson masses ###

        # Z boson mass 
        self.input_parameters['Mz'] = 91.1876

        # Higgs boson mass
        self.input_parameters['Mh'] = 125.10

        # W boson mass
        self.input_parameters['Mw'] = 80.379


        ### Lepton masses ###

        # tau mass
        self.input_parameters['mtau'] = 1.77686

        # mu mass
        self.input_parameters['mmu'] = 105.6583715e-3

        # electron mass
        self.input_parameters['me'] = 0.000510998928


        ### Baryon masses ###

        # proton mass
        self.input_parameters['mproton'] = 938.272081e-3

        # neutron mass
        self.input_parameters['mneutron'] = 939.565413e-3


        ### Meson masses ###

        # 
        self.input_parameters['mpi0'] =134.98e-3 

        # 
        self.input_parameters['meta'] = 547.862e-3


        ### Quark masses ###

        # PDG top quark pole mass
        self.input_parameters['mt_pole'] = 172.76

        # bottom quark mass, MS-bar
        self.input_parameters['mb_at_mb'] = 4.18

        # charm quark mass, MS-bar
        self.input_parameters['mc_at_mc'] = 1.27

        # strange quark mass, MS-bar at 2 GeV
        self.input_parameters['ms_at_2GeV'] = 0.093

        # down quark mass, MS-bar at 2 GeV
        self.input_parameters['md_at_2GeV'] = 0.00467

        # up quark mass, MS-bar at 2 GeV
        self.input_parameters['mu_at_2GeV'] = 0.00216


        ### Low-energy constants for chiral EFT ###

        # The strange electric charge radius squared [1/GeV^2]
        self.input_parameters['rs2'] = -0.114

        # gA
        self.input_parameters['gA'] = 1.2756

        # Delta u + Delta d
        self.input_parameters['DeltauDeltad'] = 0.440

        # Deltas
        self.input_parameters['Deltas'] = -0.035

        # mG
        self.input_parameters['mG'] = 0.836

        # sigmaup
        self.input_parameters['sigmaup'] = 17e-3

        # sigmadp
        self.input_parameters['sigmadp'] = 32e-3

        # sigmaun
        self.input_parameters['sigmaun'] = 15e-3

        # sigmadn
        self.input_parameters['sigmadn'] = 36e-3

        # sigmas
        self.input_parameters['sigmas'] = 52.9e-3

        # B0mu
        self.input_parameters['B0mu'] = 5.8e-3

        # B0md
        self.input_parameters['B0md'] = 12.4e-3

        # B0ms
        self.input_parameters['B0ms'] = 0.249

        # Nuclear dipole moments #

        # mup
        self.input_parameters['mup'] = 2.793

        # mun
        self.input_parameters['mun'] = -1.913

        # mus
        self.input_parameters['mus'] = -0.036

        # nuclear tensor charges (at 2 GeV) #

        # gTu
        self.input_parameters['gTu'] = 0.784

        # gTd
        self.input_parameters['gTd'] = -0.204

        # gTs
        self.input_parameters['gTs'] = -2.7e-2

        # BT10up
        self.input_parameters['BT10up'] = 3.0

        # BT10dp
        self.input_parameters['BT10dp'] = 0.24

        # BT10un
        self.input_parameters['BT10un'] = 0.24

        # BT10dn
        self.input_parameters['BT10dn'] = 3.0

        # BT10s
        self.input_parameters['BT10s'] = 0.

        # C-even twist-two M.E. [arxiv:1409:8290]
        self.input_parameters['f2up'] = 0.346
        self.input_parameters['f2dp'] = 0.192
        self.input_parameters['f2sp'] = 0.034

        self.input_parameters['f2un'] = 0.192
        self.input_parameters['f2dn'] = 0.346
        self.input_parameters['f2sn'] = 0.034

        self.input_parameters['f2g'] = 0.419


        #------------------------------------------------------------------------#
        # Update primary input parameters with user-specified values (optional): #
        #------------------------------------------------------------------------#

        if my_input_dict is None:
            pass
        else:
            # Issue a user warning if a key is not defined:
            for input_key in my_input_dict.keys():
                if input_key in self.input_parameters.keys():
                    pass
                else:
                    raise Exception(input_key + ' is not a valid key for an input parameter. Typo?')
            # Create the dictionary.
            self.input_parameters.update(my_input_dict)


        ###----------------------###
        ### Dependent parameters ###
        ###----------------------###

        self.dependent_parameters = {}

        # The QCD and e/w MSbar mass at mu = mt_pole
        # https://arxiv.org/abs/1212.4319

        # The 6-flavor strong coupling at mu = mt_pole. 
        # Strictly speaking, the decoupling formulas at m(mu) instead of m(m)
        # should be used. This is a tiny discrepancy which is neglected for the moment.
        as6 = rge.AlphaS(self.input_parameters['asMZ'],\
                         self.input_parameters['Mz']).\
                         run({'mtmt': self.input_parameters['mt_pole'],\
                              'mbmb': self.input_parameters['mb_at_mb'],\
                              'mcmc': self.input_parameters['mc_at_mc']},\
                             {'mut': self.input_parameters['mt_pole'],\
                              'mub': self.input_parameters['mb_at_mb'],\
                              'muc': self.input_parameters['mc_at_mc']},\
                             self.input_parameters['mt_pole'], 6, 3)
        self.dependent_parameters['mt_at_mt_pole'] = self.input_parameters['mt_pole']\
            * (1 - 4/3*as6/np.pi - 9.125*(as6/np.pi)**2 - 80.405*(as6/np.pi)**3\
               + 0.0664 - 0.00115 * (self.input_parameters['Mh'] - 125))

        # The running coupling (at LO, there are no finite threshold effects)

        self.dependent_parameters['as_at_mb'] = rge.AlphaS(self.input_parameters['asMZ'],\
                              self.input_parameters['Mz']).\
                              run({'mtmt': self.input_parameters['mt_pole'],\
                                   'mbmb': self.input_parameters['mb_at_mb'],\
                                   'mcmc': self.input_parameters['mc_at_mc']},\
                                  {'mut': self.input_parameters['mt_pole'],\
                                   'mub': self.input_parameters['mb_at_mb'],\
                                   'muc': self.input_parameters['mc_at_mc']},\
                                  self.input_parameters['mb_at_mb'], 5, 1)

        self.dependent_parameters['as_at_2GeV'] = rge.AlphaS(self.input_parameters['asMZ'],\
                                self.input_parameters['Mz']).\
                                run({'mtmt': self.input_parameters['mt_pole'],\
                                     'mbmb': self.input_parameters['mb_at_mb'],\
                                     'mcmc': self.input_parameters['mc_at_mc']},\
                                    {'mut': self.input_parameters['mt_pole'],\
                                     'mub': self.input_parameters['mb_at_mb'],\
                                     'muc': 2},\
                                    2, 3, 1)

        # The running masses

        def mt(mu, mut, mub, muc, nf, loop):
            return rge.M_Quark_MSbar('t', self.dependent_parameters['mt_at_mt_pole'],\
                                     self.input_parameters['mt_pole'],\
                                     self.input_parameters['asMZ'],\
                                     self.input_parameters['Mz']).run(mu,\
                                        {'mtmt': 163.48,\
                                         'mbmb': self.input_parameters['mb_at_mb'],\
                                         'mcmc': self.input_parameters['mc_at_mc']},\
                                        {'mut': mut, 'mub': mub, 'muc': muc}, nf, loop)

        def mb(mu, mub, muc, nf, loop):
            return rge.M_Quark_MSbar('b', self.input_parameters['mb_at_mb'],\
                                     self.input_parameters['mb_at_mb'],\
                                     self.input_parameters['asMZ'],\
                                     self.input_parameters['Mz']).run(mu,\
                                        {'mbmb': self.input_parameters['mb_at_mb'],\
                                         'mcmc': self.input_parameters['mc_at_mc']},\
                                        {'mub': mub, 'muc': muc}, nf, loop)

        def mc(mu, mub, muc, nf, loop):
            return rge.M_Quark_MSbar('c', self.input_parameters['mc_at_mc'],\
                                     self.input_parameters['mc_at_mc'],\
                                     self.input_parameters['asMZ'],\
                                     self.input_parameters['Mz']).run(mu,\
                                        {'mbmb': self.input_parameters['mb_at_mb'],\
                                         'mcmc': self.input_parameters['mc_at_mc']},\
                                        {'mub': mub, 'muc': muc}, nf, loop)

        def ms(mu, mub, muc, nf, loop):
            return rge.M_Quark_MSbar('s', self.input_parameters['ms_at_2GeV'],\
                                     2, self.input_parameters['asMZ'],\
                                     self.input_parameters['Mz']).run(mu,\
                                        {'mbmb': self.input_parameters['mb_at_mb'],\
                                         'mcmc': self.input_parameters['mc_at_mc']},\
                                        {'mub': mub, 'muc': muc}, nf, loop)

        def md(mu, mub, muc, nf, loop):
            return rge.M_Quark_MSbar('d', self.input_parameters['md_at_2GeV'],\
                                     2, self.input_parameters['asMZ'],\
                                     self.input_parameters['Mz']).run(mu,\
                                        {'mbmb': self.input_parameters['mb_at_mb'],\
                                         'mcmc': self.input_parameters['mc_at_mc']},\
                                        {'mub': mub, 'muc': muc}, nf, loop)

        def mu(mu, mub, muc, nf, loop):
            return rge.M_Quark_MSbar('u', self.input_parameters['mu_at_2GeV'],\
                                     2, self.input_parameters['asMZ'],\
                                     self.input_parameters['Mz']).run(mu,\
                                        {'mbmb': self.input_parameters['mb_at_mb'],\
                                         'mcmc': self.input_parameters['mc_at_mc']},\
                                        {'mub': mub, 'muc': muc}, nf, loop)



        # top quark mass, MS-bar (converted to MSbar QCD and EW, and run to MZ at 1-loop QCD)
        self.dependent_parameters['mt_at_MZ'] = mt(self.input_parameters['Mz'],\
                                                   self.input_parameters['mt_pole'],\
                                                   self.input_parameters['mb_at_mb'],\
                                                   self.input_parameters['mc_at_mc'], 6, 1)

        self.dependent_parameters['mb_at_MZ'] = mb(self.input_parameters['Mz'],\
                                                   self.input_parameters['mb_at_mb'],\
                                                   self.input_parameters['mc_at_mc'], 5, 1)
        self.dependent_parameters['mc_at_MZ'] = mc(self.input_parameters['Mz'],\
                                                   self.input_parameters['mb_at_mb'],\
                                                   self.input_parameters['mc_at_mc'], 5, 1)
        self.dependent_parameters['ms_at_MZ'] = ms(self.input_parameters['Mz'],\
                                                   self.input_parameters['mb_at_mb'],\
                                                   self.input_parameters['mc_at_mc'], 5, 1)
        self.dependent_parameters['md_at_MZ'] = md(self.input_parameters['Mz'],\
                                                   self.input_parameters['mb_at_mb'],\
                                                   self.input_parameters['mc_at_mc'], 5, 1)
        self.dependent_parameters['mu_at_MZ'] = mu(self.input_parameters['Mz'],\
                                                   self.input_parameters['mb_at_mb'],\
                                                   self.input_parameters['mc_at_mc'], 5, 1)

        self.dependent_parameters['ms_at_mb'] = ms(self.input_parameters['mb_at_mb'],\
                                                   self.input_parameters['mb_at_mb'],\
                                                   self.input_parameters['mc_at_mc'], 5, 1)
        self.dependent_parameters['md_at_mb'] = md(self.input_parameters['mb_at_mb'],\
                                                   self.input_parameters['mb_at_mb'],\
                                                   self.input_parameters['mc_at_mc'], 5, 1)
        self.dependent_parameters['mu_at_mb'] = mu(self.input_parameters['mb_at_mb'],\
                                                   self.input_parameters['mb_at_mb'],\
                                                   self.input_parameters['mc_at_mc'], 5, 1)


        # Deltaup
        self.dependent_parameters['Deltaup'] = (   self.input_parameters['gA']\
                                                   + self.input_parameters['DeltauDeltad'])/2

        # Deltadp
        self.dependent_parameters['Deltadp'] = ( - self.input_parameters['gA']\
                                                 + self.input_parameters['DeltauDeltad'])/2

        # Deltaun
        self.dependent_parameters['Deltaun'] = self.dependent_parameters['Deltadp']

        # Deltadn
        self.dependent_parameters['Deltadn'] = self.dependent_parameters['Deltaup']

        # muup
        self.dependent_parameters['muup'] = self.input_parameters['mup'] + self.input_parameters['mun']/2

        # mudp
        self.dependent_parameters['mudp'] = self.input_parameters['mup'] + 2*self.input_parameters['mun']

        # mudn
        self.dependent_parameters['mudn'] = self.dependent_parameters['muup']

        # muun
        self.dependent_parameters['muun'] = self.dependent_parameters['mudp']

        # ap
        self.dependent_parameters['ap'] = self.input_parameters['mup'] - 1.

        # an
        self.dependent_parameters['an'] = self.input_parameters['mun']

        # F2sp
        self.dependent_parameters['F2sp'] = self.input_parameters['mus']


        # Input parameters at MZ for electroweak RG evolution

        # SU2 coupling
        self.dependent_parameters['g2_at_MZ']\
          = np.sqrt(4*np.pi/self.input_parameters['aMZinv']/self.input_parameters['sw2_MSbar'])

        # U1 coupling
        self.dependent_parameters['g1_at_MZ']\
          = np.sqrt(self.dependent_parameters['g2_at_MZ']**2/(1/self.input_parameters['sw2_MSbar'] - 1))

        # SU3 coupling
        self.dependent_parameters['g3_at_MZ']\
          = np.sqrt(4*np.pi*self.input_parameters['asMZ'])

        # charm Yukawa
        self.dependent_parameters['yc_at_MZ']\
          = np.sqrt(np.sqrt(2)*self.input_parameters['GF'])*np.sqrt(2) * self.dependent_parameters['mc_at_MZ']

        # bottom Yukawa
        self.dependent_parameters['yb_at_MZ']\
          = np.sqrt(np.sqrt(2)*self.input_parameters['GF'])*np.sqrt(2) * self.dependent_parameters['mb_at_MZ']

        # tau Yukawa
        self.dependent_parameters['ytau_at_MZ']\
          = np.sqrt(np.sqrt(2)*self.input_parameters['GF'])*np.sqrt(2) * self.input_parameters['mtau']

        # top Yukawa
        self.dependent_parameters['yt_at_MZ']\
          = np.sqrt(np.sqrt(2)*self.input_parameters['GF'])*np.sqrt(2) * self.dependent_parameters['mt_at_MZ']

        # Higgs quartic coupling
        self.dependent_parameters['lam_at_MZ']\
          = 2*np.sqrt(2) * self.input_parameters['GF'] * self.input_parameters['Mh']**2


        # Update the dictionary with the dependent parameters
        self.input_parameters.update(self.dependent_parameters)


