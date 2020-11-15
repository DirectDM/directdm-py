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

# The parameter with 'd' in front denotes the corresponding uncertainty
# (currently not used anywhere in the code)

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
        self.input_parameters['dasMZ'] = 0.0010

        # The Fermi constant
        self.input_parameters['GF'] = 1.1663787e-5
        self.input_parameters['dGF'] = 0.0000006e-5

        # The inverse of QED alpha @ MZ
        self.input_parameters['aMZinv'] = 127.952
        self.input_parameters['daMZinv'] = 0.009

        # The inverse of QED alpha @ mtau
        self.input_parameters['amtauinv'] = 133.472
        self.input_parameters['damtauinv'] = 0.007

        # The inverse of QED alpha at low scales
        self.input_parameters['alowinv'] = 137.035999084
        self.input_parameters['dalowinv'] = 0.000000021

        # sin-squared of the weak mixing angle (MS-bar)
        self.input_parameters['sw2_MSbar'] = 0.23121
        self.input_parameters['dsw2_MSbar'] = 0.00004


        ### Boson masses ###

        # Z boson mass 
        self.input_parameters['Mz'] = 91.1876
        self.input_parameters['dMz'] = 0.0021

        # Higgs boson mass
        self.input_parameters['Mh'] = 125.10
        self.input_parameters['dMh'] = 0.14

        # W boson mass
        self.input_parameters['Mw'] = 80.379
        self.input_parameters['dMw'] = 0.012


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
        self.input_parameters['dmproton'] = 0.000006e-3

        # neutron mass
        self.input_parameters['mneutron'] = 939.565413e-3
        self.input_parameters['dmneutron'] = 0.000006e-3


        ### Meson masses ###

        # 
        self.input_parameters['mpi0'] =134.98e-3 
        self.input_parameters['dmpi0'] = 0.

        # 
        self.input_parameters['meta'] = 547.862e-3
        self.input_parameters['dmeta'] = 0.017e-3


        ### Quark masses ###

        # top quark mass, e/w onshell, QCD MS-bar
        self.input_parameters['mt_at_mt_QCD'] = 160.
        self.input_parameters['dmt_at_mt_QCD'] = 5.

        # top quark pole mass
        self.input_parameters['mt_pole'] = 173.0
        self.input_parameters['dmt_pole'] = 0.4

        # bottom quark mass, MS-bar
        self.input_parameters['mb_at_mb'] = 4.18
        self.input_parameters['dmb_at_mb'] = 0.04

        # charm quark mass, MS-bar
        self.input_parameters['mc_at_mc'] = 1.275
        self.input_parameters['dmc_at_mc'] = 0.03

        # strange quark mass, MS-bar at 2 GeV
        self.input_parameters['ms_at_2GeV'] = 0.093

        # down quark mass, MS-bar at 2 GeV
        self.input_parameters['md_at_2GeV'] = 0.00467

        # up quark mass, MS-bar at 2 GeV
        self.input_parameters['mu_at_2GeV'] = 0.00216

        # top quark mass, MS-bar (converted to MSbar QCD and EW, and run to MZ at 1-loop QCD)
        self.input_parameters['mt_at_MZ'] = 182.


        ### Low-energy constants for chiral EFT ###

        # gA
        self.input_parameters['gA'] = 1.2756
        self.input_parameters['dgA'] = 0.0013

        # Delta u + Delta d
        self.input_parameters['DeltauDeltad'] = 0.440
        self.input_parameters['dDeltauDeltad'] = 0.044

        # Deltas
        self.input_parameters['Deltas'] = -0.035
        self.input_parameters['dDeltas'] = 0.009

        # mG
        self.input_parameters['mG'] = 0.836
        self.input_parameters['dmG'] = 0.017

        # sigmaup
        self.input_parameters['sigmaup'] = 17e-3
        self.input_parameters['dsigmaup'] = 5e-3

        # sigmadp
        self.input_parameters['sigmadp'] = 32e-3
        self.input_parameters['dsigmadp'] = 10e-3

        # sigmaun
        self.input_parameters['sigmaun'] = 15e-3
        self.input_parameters['dsigmaun'] = 5e-3

        # sigmadn
        self.input_parameters['sigmadn'] = 36e-3
        self.input_parameters['dsigmadn'] = 10e-3

        # sigmas
        self.input_parameters['sigmas'] = 52.9e-3
        self.input_parameters['dsigmas'] = 7.0e-3

        # B0mu
        self.input_parameters['B0mu'] = 5.8e-3
        self.input_parameters['dB0mu'] = 0.5e-3

        # B0md
        self.input_parameters['B0md'] = 12.4e-3
        self.input_parameters['dB0md'] = 0.5e-3

        # B0ms
        self.input_parameters['B0ms'] = 0.249
        self.input_parameters['dB0ms'] = 0.009

        # Nuclear dipole moments #

        # mup
        self.input_parameters['mup'] = 2.793

        # mun
        self.input_parameters['mun'] = -1.913

        # mus
        self.input_parameters['mus'] = -0.036
        self.input_parameters['dmus'] = 0.021

        # nuclear tensor charges (at 2 GeV) #

        # gTu
        self.input_parameters['gTu'] = 0.784
        self.input_parameters['dgTu'] = 0.030

        # gTd
        self.input_parameters['gTd'] = -0.204
        self.input_parameters['dgTd'] = 0.015

        # gTs
        self.input_parameters['gTs'] = 2.7e-2
        self.input_parameters['dgTs'] = 1.6e-2

        # BT10up
        self.input_parameters['BT10up'] = 3.0
        self.input_parameters['dBT10up'] = 3.0/2

        # BT10dp
        self.input_parameters['BT10dp'] = 0.24
        self.input_parameters['dBT10dp'] = 0.24/2

        # BT10un
        self.input_parameters['BT10un'] = 0.24
        self.input_parameters['dBT10un'] = 0.24/2

        # BT10dn
        self.input_parameters['BT10dn'] = 3.0
        self.input_parameters['dBT10dn'] = 3.0/2

        # BT10s
        self.input_parameters['BT10s'] = 0.
        self.input_parameters['dBT10s'] = 0.2

        # C-even twist-two M.E. [arxiv:1409:8290]
        self.input_parameters['f2up'] = 0.346
        self.input_parameters['df2up'] = 0.007
        self.input_parameters['f2dp'] = 0.192
        self.input_parameters['df2dp'] = 0.006
        self.input_parameters['f2sp'] = 0.034
        self.input_parameters['df2sp'] = 0.003

        self.input_parameters['f2un'] = 0.192
        self.input_parameters['df2un'] = 0.006
        self.input_parameters['f2dn'] = 0.346
        self.input_parameters['df2dn'] = 0.007
        self.input_parameters['f2sn'] = 0.034
        self.input_parameters['df2sn'] = 0.003

        self.input_parameters['f2g'] = 0.419
        self.input_parameters['df2g'] = 0.019


        ### Dependent parameters ###

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

        self.input_parameters['mb_at_MZ'] = mb(self.input_parameters['Mz'],\
                                               self.input_parameters['mb_at_mb'],\
                                               self.input_parameters['mc_at_mc'], 5, 1)
        self.input_parameters['mc_at_MZ'] = mc(self.input_parameters['Mz'],\
                                               self.input_parameters['mb_at_mb'],\
                                               self.input_parameters['mc_at_mc'], 5, 1)
        self.input_parameters['ms_at_MZ'] = ms(self.input_parameters['Mz'],\
                                               self.input_parameters['mb_at_mb'],\
                                               self.input_parameters['mc_at_mc'], 5, 1)
        self.input_parameters['md_at_MZ'] = md(self.input_parameters['Mz'],\
                                               self.input_parameters['mb_at_mb'],\
                                               self.input_parameters['mc_at_mc'], 5, 1)
        self.input_parameters['mu_at_MZ'] = mu(self.input_parameters['Mz'],\
                                               self.input_parameters['mb_at_mb'],\
                                               self.input_parameters['mc_at_mc'], 5, 1)


        # Deltaup
        self.input_parameters['Deltaup'] = (   self.input_parameters['gA']\
                                             + self.input_parameters['DeltauDeltad'])/2
        self.input_parameters['dDeltaup'] = np.sqrt(   self.input_parameters['dgA']**2\
                                                     + self.input_parameters['DeltauDeltad']**2)/2

        # Deltadp
        self.input_parameters['Deltadp'] = ( - self.input_parameters['gA']\
                                             + self.input_parameters['DeltauDeltad'])/2
        self.input_parameters['dDeltadp'] = np.sqrt(   self.input_parameters['dgA']**2\
                                                     + self.input_parameters['DeltauDeltad']**2)/2

        # Deltaun
        self.input_parameters['Deltaun'] = self.input_parameters['Deltadp']
        self.input_parameters['dDeltaun'] = self.input_parameters['dDeltadp']

        # Deltadn
        self.input_parameters['Deltadn'] = self.input_parameters['Deltaup']
        self.input_parameters['dDeltadn'] = self.input_parameters['dDeltaup']

        # muup
        self.input_parameters['muup'] = self.input_parameters['mup'] + self.input_parameters['mun']/2

        # mudp
        self.input_parameters['mudp'] = self.input_parameters['mup'] + 2*self.input_parameters['mun']

        # mudn
        self.input_parameters['mudn'] = self.input_parameters['muup']

        # muun
        self.input_parameters['muun'] = self.input_parameters['mudp']

        # ap
        self.input_parameters['ap'] = self.input_parameters['mup'] - 1.

        # an
        self.input_parameters['an'] = self.input_parameters['mun']

        # F2sp
        self.input_parameters['F2sp'] = self.input_parameters['mus']




        if my_input_dict is None:
            pass
        else:
            # Issue a user warning if a key is not defined:
            for input_key in my_input_dict.keys():
                if input_key in self.input_parameters.keys():
                    pass
                else:
                    warnings.warn(input_key + ' is not a valid key for an input parameter. Typo?')
            # Create the dictionary.
            self.input_parameters.update(my_input_dict)

