#!/usr/bin/env python3

        
class F1:
    def __init__(self, quark, nucleon, input_dict):
        """ The nuclear form factor F1
        
        Return the nuclear form factor F1

        Arguments
        ---------
        quark = 'u', 'd', 's' -- the quark flavor (up, down, strange)

        nucleon = 'p', 'n' -- the nucleon (proton or neutron)
        """

        self.quark = quark
        self.nucleon = nucleon
        self.ip = input_dict

    def value_zero_mom(self):
        """ Return the value of the form factor at zero momentum transfer """

        if self.nucleon == 'p':
            if self.quark == 'u':
                return 2
            if self.quark == 'd':
                return 1
            if self.quark == 's':
                return 0
        if self.nucleon == 'n':
            if self.quark == 'u':
                return 1
            if self.quark == 'd':
                return 2
            if self.quark == 's':
                return 0

    def first_deriv_zero_mom(self):
        """ Return the value of the first derivative of the form factor
            w.r.t. q^2 at zero momentum transfer (only strange quark) """

        if self.nucleon == 'p':
            if self.quark == 's':
                return self.ip['rs2'] / 6
        if self.nucleon == 'n':
            if self.quark == 's':
                return self.ip['rs2'] / 6


class F2(object):
    def __init__(self, quark, nucleon, input_dict):
        """ The nuclear form factor F2

        Return the nuclear form factor F2

        Arguments
        ---------
        quark = 'u', 'd', 's' -- the quark flavor (up, down, strange)

        nucleon = 'p', 'n' -- the nucleon (proton or neutron)

        input_dict (optional) -- a dictionary of hadronic input parameters
                                 (default is Num_input().input_parameters)
        """
        self.quark = quark
        self.nucleon = nucleon
        self.ip = input_dict


    def value_zero_mom(self):
        """ Return the value of the form factor at zero momentum transfer """

        if self.nucleon == 'p':
            if self.quark == 'u':
                return 2*self.ip['ap'] + self.ip['an'] + self.ip['F2sp']
            if self.quark == 'd':
                return 2*self.ip['an'] + self.ip['ap'] + self.ip['F2sp']
            if self.quark == 's':
                return self.ip['F2sp']
        if self.nucleon == 'n':
            if self.quark == 'u':
                return 2*self.ip['an'] + self.ip['ap'] + self.ip['F2sp']
            if self.quark == 'd':
                return 2*self.ip['ap'] + self.ip['an'] + self.ip['F2sp']
            if self.quark == 's':
                return self.ip['F2sp']


class FA(object):
    def __init__(self, quark, nucleon, input_dict):
        """ The nuclear form factor FA at zero momentum transfer

        Return the nuclear form factor FA, evaluated at zero momentum transfer.

        Arguments
        ---------
        quark = 'u', 'd', 's' -- the quark flavor (up, down, strange)

        nucleon = 'p', 'n' -- the nucleon (proton or neutron)

        input_dict (optional) -- a dictionary of hadronic input parameters
                                 (default is Num_input().input_parameters)
        """
        self.quark = quark
        self.nucleon = nucleon
        self.ip = input_dict


    def value_zero_mom(self):
        """ Return the value of the form factor at zero momentum transfer """

        if self.nucleon == 'p':
            if self.quark == 'u':
                return self.ip['Deltaup']
            if self.quark == 'd':
                return self.ip['Deltadp']
            if self.quark == 's':
                return self.ip['Deltas']
        if self.nucleon == 'n':
            if self.quark == 'u':
                return self.ip['Deltaun']
            if self.quark == 'd':
                return self.ip['Deltadn']
            if self.quark == 's':
                return self.ip['Deltas']


class FPprimed(object):
    def __init__(self, quark, nucleon, input_dict):
        """ The nuclear form factor FPprimed

        Return the nuclear form factor FPprimed

        Arguments
        ---------
        quark = 'u', 'd', 's' -- the quark flavor (up, down, strange)

        nucleon = 'p', 'n' -- the nucleon (proton or neutron)

        input_dict (optional) -- a dictionary of hadronic input parameters
                                 (default is Num_input().input_parameters)
        """
        self.quark = quark
        self.nucleon = nucleon
        self.ip = input_dict


    def value_pion_pole(self):
        """ Return the coefficient of the pion pole

        The pion pole is given, in terms of the spatial momentum q, by 1 / (q^2 + mpi0^2)
        """

        self.mN = (self.ip['mproton'] + self.ip['mneutron'])/2

        if self.nucleon == 'p':
            if self.quark == 'u':
                return self.mN**2 * 2 * self.ip['gA']
            if self.quark == 'd':
                return - self.mN**2 * 2 * self.ip['gA']
            if self.quark == 's':
                return 0
        if self.nucleon == 'n':
            if self.quark == 'u':
                return - self.mN**2 * 2 * self.ip['gA']
            if self.quark == 'd':
                return self.mN**2 * 2 * self.ip['gA']
            if self.quark == 's':
                return 0

    def value_eta_pole(self):
        """ Return the coefficient of the pion pole

        The eta pole is given, in terms of the spatial momentum q, by 1 / (q^2 + meta^2)
        """

        self.mN = (self.ip['mproton'] + self.ip['mneutron'])/2

        if self.nucleon == 'p':
            if self.quark == 'u':
                return self.mN**2 * 2 * (self.ip['Deltaup'] + self.ip['Deltadp'] - 2*self.ip['Deltas'])/3
            if self.quark == 'd':
                return self.mN**2 * 2 * (self.ip['Deltaup'] + self.ip['Deltadp'] - 2*self.ip['Deltas'])/3
            if self.quark == 's':
                return - self.mN**2 * 4 * (self.ip['Deltaup'] + self.ip['Deltadp'] - 2*self.ip['Deltas'])/3
        if self.nucleon == 'n':
            if self.quark == 'u':
                return self.mN**2 * 2 * (self.ip['Deltaup'] + self.ip['Deltadp'] - 2*self.ip['Deltas'])/3
            if self.quark == 'd':
                return self.mN**2 * 2 * (self.ip['Deltaup'] + self.ip['Deltadp'] - 2*self.ip['Deltas'])/3
            if self.quark == 's':
                return - self.mN**2 * 4 * (self.ip['Deltaup'] + self.ip['Deltadp'] - 2*self.ip['Deltas'])/3


class FS(object):
    def __init__(self, quark, nucleon, input_dict):
        """ The nuclear form factor FS

        Return the nuclear form factor FS

        Arguments
        ---------
        quark = 'u', 'd', 's' -- the quark flavor (up, down, strange)

        nucleon = 'p', 'n' -- the nucleon (proton or neutron)

        input_dict (optional) -- a dictionary of hadronic input parameters
                                 (default is Num_input().input_parameters)
        """
        self.quark = quark
        self.nucleon = nucleon
        self.ip = input_dict


    def value_zero_mom(self):
        """ Return the value of the form factor at zero momentum transfer """

        if self.nucleon == 'p':
            if self.quark == 'u':
                return self.ip['sigmaup']
            if self.quark == 'd':
                return self.ip['sigmadp']
            if self.quark == 's':
                return self.ip['sigmas']
        if self.nucleon == 'n':
            if self.quark == 'u':
                return self.ip['sigmaun']
            if self.quark == 'd':
                return self.ip['sigmadn']
            if self.quark == 's':
                return self.ip['sigmas']


class FP(object):
    def __init__(self, quark, nucleon, input_dict):
        """ The nuclear form factor FP

        Return the nuclear form factor FP

        Arguments
        ---------
        quark = 'u', 'd', 's' -- the quark flavor (up, down, strange)

        nucleon = 'p', 'n' -- the nucleon (proton or neutron)

        input_dict (optional) -- a dictionary of hadronic input parameters
                                 (default is Num_input().input_parameters)
        """
        self.quark = quark
        self.nucleon = nucleon
        self.ip = input_dict


    def value_pion_pole(self):
        """ Return the coefficient of the pion pole

        The pion pole is given, in terms of the spatial momentum q, by 1 / (q^2 + mpi0^2)
        """

        self.mN = (self.ip['mproton'] + self.ip['mneutron'])/2

        if self.nucleon == 'p':
            if self.quark == 'u':
                return self.mN**2 * self.ip['gA'] * self.ip['B0mu'] / self.mN 
            if self.quark == 'd':
                return - self.mN**2 * self.ip['gA'] * self.ip['B0md'] / self.mN 
            if self.quark == 's':
                return 0
        if self.nucleon == 'n':
            if self.quark == 'u':
                return - self.mN**2 * self.ip['gA'] * self.ip['B0mu'] / self.mN 
            if self.quark == 'd':
                return self.mN**2 * self.ip['gA'] * self.ip['B0md'] / self.mN 
            if self.quark == 's':
                return 0

    def value_eta_pole(self):
        """ Return the coefficient of the pion pole

        The eta pole is given, in terms of the spatial momentum q, by 1 / (q^2 + meta^2)
        """

        self.mN = (self.ip['mproton'] + self.ip['mneutron'])/2

        if self.nucleon == 'p':
            if self.quark == 'u':
                return self.mN**2 * (self.ip['Deltaup'] + self.ip['Deltadp'] - 2*self.ip['Deltas'])/3/self.mN * self.ip['B0mu']
            if self.quark == 'd':
                return self.mN**2 * (self.ip['Deltaup'] + self.ip['Deltadp'] - 2*self.ip['Deltas'])/3/self.mN * self.ip['B0md']
            if self.quark == 's':
                return - 2 * self.mN**2 * (self.ip['Deltaup'] + self.ip['Deltadp'] - 2*self.ip['Deltas'])/3/self.mN * self.ip['B0ms']
        if self.nucleon == 'n':
            if self.quark == 'u':
                return self.mN**2 * (self.ip['Deltaup'] + self.ip['Deltadp'] - 2*self.ip['Deltas'])/3/self.mN * self.ip['B0mu']
            if self.quark == 'd':
                return self.mN**2 * (self.ip['Deltaup'] + self.ip['Deltadp'] - 2*self.ip['Deltas'])/3/self.mN * self.ip['B0md']
            if self.quark == 's':
                return - 2 * self.mN**2 * (self.ip['Deltaup'] + self.ip['Deltadp'] - 2*self.ip['Deltas'])/3/self.mN * self.ip['B0ms']


class FT0(object):
    def __init__(self, quark, nucleon, input_dict):
        """ The nuclear form factor FT0

        Return the nuclear form factor FT0

        Arguments
        ---------
        quark = 'u', 'd', 's' -- the quark flavor (up, down, strange)

        nucleon = 'p', 'n' -- the nucleon (proton or neutron)

        input_dict (optional) -- a dictionary of hadronic input parameters
                                 (default is Num_input().input_parameters)
        """
        self.quark = quark
        self.nucleon = nucleon
        self.ip = input_dict


    def value_zero_mom(self):
        """ Return the value of the form factor at zero momentum transfer """

        if self.nucleon == 'p':
            if self.quark == 'u':
                return self.ip['mu_at_2GeV'] * self.ip['gTu']
            if self.quark == 'd':
                return self.ip['md_at_2GeV'] * self.ip['gTd']
            if self.quark == 's':
                return self.ip['ms_at_2GeV'] * self.ip['gTs']
        if self.nucleon == 'n':
            if self.quark == 'u':
                return self.ip['mu_at_2GeV'] * self.ip['gTd']
            if self.quark == 'd':
                return self.ip['md_at_2GeV'] * self.ip['gTu']
            if self.quark == 's':
                return self.ip['ms_at_2GeV'] * self.ip['gTs']


class FT1(object):
    def __init__(self, quark, nucleon, input_dict):
        """ The nuclear form factor FT1

        Return the nuclear form factor FT1

        Arguments
        ---------
        quark = 'u', 'd', 's' -- the quark flavor (up, down, strange)

        nucleon = 'p', 'n' -- the nucleon (proton or neutron)

        input_dict (optional) -- a dictionary of hadronic input parameters
                                 (default is Num_input().input_parameters)
        """
        self.quark = quark
        self.nucleon = nucleon
        self.ip = input_dict


    def value_zero_mom(self):
        """ Return the value of the form factor at zero momentum transfer """

        if self.nucleon == 'p':
            if self.quark == 'u':
                return - self.ip['mu_at_2GeV'] * self.ip['BT10up']
            if self.quark == 'd':
                return - self.ip['md_at_2GeV'] * self.ip['BT10dp']
            if self.quark == 's':
                return - self.ip['ms_at_2GeV'] * self.ip['BT10s']
        if self.nucleon == 'n':
            if self.quark == 'u':
                return - self.ip['mu_at_2GeV'] * self.ip['BT10un']
            if self.quark == 'd':
                return - self.ip['md_at_2GeV'] * self.ip['BT10dn']
            if self.quark == 's':
                return - self.ip['ms_at_2GeV'] * self.ip['BT10s']


class FG(object):
    def __init__(self, nucleon, input_dict):
        """ The nuclear form factor FG

        Return the nuclear form factor FG

        Arguments
        ---------
        nucleon = 'p', 'n' -- the nucleon (proton or neutron)

        input_dict (optional) -- a dictionary of hadronic input parameters
                                 (default is Num_input().input_parameters)
        """
        self.nucleon = nucleon
        self.ip = input_dict


    def value_zero_mom(self):
        """ Return the value of the form factor at zero momentum transfer """

        if self.nucleon == 'p':
            return -2*self.ip['mG']/27
        if self.nucleon == 'n':
            return -2*self.ip['mG']/27


class FGtilde(object):
    def __init__(self, nucleon, input_dict):
        """ The nuclear form factor FGtilde

        Return the nuclear form factor FGtilde

        Arguments
        ---------
        nucleon = 'p', 'n' -- the nucleon (proton or neutron)

        input_dict (optional) -- a dictionary of hadronic input parameters
                                 (default is Num_input().input_parameters)
        """
        self.nucleon = nucleon
        self.ip = input_dict


    def value_zero_mom(self):
        """ Return the value of the form factor at zero momentum transfer """

        self.mtilde = 1/(1/self.ip['mu_at_2GeV'] + 1/self.ip['md_at_2GeV'] + 1/self.ip['ms_at_2GeV'])
        self.mN = (self.ip['mproton'] + self.ip['mneutron'])/2
        
        if self.nucleon == 'p':
            return -self.mN * self.mtilde * (self.ip['Deltaup']/self.ip['mu_at_2GeV']\
                                              + self.ip['Deltadp']/self.ip['md_at_2GeV']\
                                              + self.ip['Deltas']/self.ip['ms_at_2GeV'])
        if self.nucleon == 'n':
            return -self.mN * self.mtilde * (self.ip['Deltaun']/self.ip['mu_at_2GeV']\
                                              + self.ip['Deltadn']/self.ip['md_at_2GeV']\
                                              + self.ip['Deltas']/self.ip['ms_at_2GeV'])

    def value_pion_pole(self):
        """ Return the coefficient of the pion pole

        The pion pole is given, in terms of the spatial momentum q, by q^2 / (q^2 + mpi0^2)
        """

        self.mtilde = 1/(1/self.ip['mu_at_2GeV'] + 1/self.ip['md_at_2GeV'] + 1/self.ip['ms_at_2GeV'])
        self.mN = (self.ip['mproton'] + self.ip['mneutron'])/2

        if self.nucleon == 'p':
            return self.mN * self.mtilde * self.ip['gA'] * (1/self.ip['mu_at_2GeV'] - 1/self.ip['md_at_2GeV']) / 2
        if self.nucleon == 'n':
            return - self.mN * self.mtilde * self.ip['gA'] * (1/self.ip['mu_at_2GeV'] - 1/self.ip['md_at_2GeV']) / 2

    def value_eta_pole(self):
        """ Return the coefficient of the eta pole

        The eta pole is given, in terms of the spatial momentum q, by q^2 / (q^2 + meta^2)
        """

        self.mtilde = 1/(1/self.ip['mu_at_2GeV'] + 1/self.ip['md_at_2GeV'] + 1/self.ip['ms_at_2GeV'])
        self.mN = (self.ip['mproton'] + self.ip['mneutron'])/2

        if self.nucleon == 'p':
            return self.mN * self.mtilde * (self.ip['Deltaup'] + self.ip['Deltadp'] - 2*self.ip['Deltas'])\
                * (1/self.ip['mu_at_2GeV'] + 1/self.ip['md_at_2GeV'] - 2/self.ip['ms_at_2GeV']) / 6
        if self.nucleon == 'n':
            return self.mN * self.mtilde * (self.ip['Deltaun'] + self.ip['Deltadn'] - 2*self.ip['Deltas'])\
                * (1/self.ip['mu_at_2GeV'] + 1/self.ip['md_at_2GeV'] - 2/self.ip['ms_at_2GeV']) / 6


class FTwist2:
    def __init__(self, flavor, nucleon, input_dict):
        """ The twist-two nuclear form factors
        
        Return the twist-two nuclear form factors

        Arguments
        ---------
        flavor = 'u', 'd', 's', 'g' -- the "quark" flavor (up, down, strange, or gluon contribution)

        nucleon = 'p', 'n' -- the nucleon (proton or neutron)

        input_dict (optional) -- a dictionary of hadronic input parameters
                                 (default is Num_input().input_parameters)
        """

        self.flavor = flavor
        self.nucleon = nucleon
        self.ip = input_dict


    def value_zero_mom(self):
        """ Return the value of the form factor at zero momentum transfer """

        self.mp = self.ip['mproton']
        self.mn = self.ip['mneutron']

        if self.nucleon == 'p':
            if self.flavor == 'u':
                return 3/4 * self.mp * self.ip['f2up']
            if self.flavor == 'd':
                return 3/4 * self.mp * self.ip['f2dp']
            if self.flavor == 's':
                return 3/4 * self.mp * self.ip['f2sp']
            if self.flavor == 'g':
                return 3/4 * self.mp * self.ip['f2g']
        if self.nucleon == 'n':
            if self.flavor == 'u':
                return 3/4 * self.mn * self.ip['f2un']
            if self.flavor == 'd':
                return 3/4 * self.mn * self.ip['f2dn']
            if self.flavor == 's':
                return 3/4 * self.mn * self.ip['f2sn']
            if self.flavor == 'g':
                return 3/4 * self.mn * self.ip['f2g']


