#!/usr/bin/env python3

from directdm.num.num_input import Num_input

class F1:
    def __init__(self, quark, nucleon):
        """ The nuclear form factor F1
        
        Return the nuclear form factor F1

        Arguments
        ---------
        quark = 'u', 'd', 's' -- the quark flavor (up, down, strange)

        nucleon = 'p', 'n' -- the nucleon (proton or neutron)
        """

        self.quark = quark
        self.nucleon = nucleon

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


class F2(object):
    def __init__(self, quark, nucleon, input_dict=None):
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

        if input_dict is None:
            self.input_dict = Num_input().input_parameters
            # One should include a warning in case the dictionary
            # does not contain all necessary keys
        else:
            self.input_dict = input_dict

    def value_zero_mom(self):
        """ Return the value of the form factor at zero momentum transfer """

        ip = self.input_dict
        
        if self.nucleon == 'p':
            if self.quark == 'u':
                return 2*ip['ap'] + ip['an'] + ip['F2sp']
            if self.quark == 'd':
                return 2*ip['an'] + ip['ap'] + ip['F2sp']
            if self.quark == 's':
                return ip['F2sp']
        if self.nucleon == 'n':
            if self.quark == 'u':
                return 2*ip['an'] + ip['ap'] + ip['F2sp']
            if self.quark == 'd':
                return 2*ip['ap'] + ip['an'] + ip['F2sp']
            if self.quark == 's':
                return ip['F2sp']


class FA(object):
    def __init__(self, quark, nucleon, input_dict=None):
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

        if input_dict is None:
            self.input_dict = Num_input().input_parameters
            # One should include a warning in case the dictionary
            # does not contain all necessary keys
        else:
            self.input_dict = input_dict

    def value_zero_mom(self):
        """ Return the value of the form factor at zero momentum transfer """
        ip = self.input_dict

        if self.nucleon == 'p':
            if self.quark == 'u':
                return ip['Deltaup']
            if self.quark == 'd':
                return ip['Deltadp']
            if self.quark == 's':
                return ip['Deltas']
        if self.nucleon == 'n':
            if self.quark == 'u':
                return ip['Deltaun']
            if self.quark == 'd':
                return ip['Deltadn']
            if self.quark == 's':
                return ip['Deltas']


class FPprimed(object):
    def __init__(self, quark, nucleon, input_dict=None):
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

        if input_dict is None:
            self.input_dict = Num_input().input_parameters
            # One should include a warning in case the dictionary
            # does not contain all necessary keys
        else:
            self.input_dict = input_dict

    def value_pion_pole(self):
        """ Return the coefficient of the pion pole

        The pion pole is given, in terms of the spatial momentum q, by 1 / (q^2 + mpi0^2)
        """
        ip = self.input_dict
        self.mN = (ip['mproton'] + ip['mneutron'])/2

        if self.nucleon == 'p':
            if self.quark == 'u':
                return self.mN**2 * 2 * ip['gA']
            if self.quark == 'd':
                return - self.mN**2 * 2 * ip['gA']
            if self.quark == 's':
                return 0
        if self.nucleon == 'n':
            if self.quark == 'u':
                return - self.mN**2 * 2 * ip['gA']
            if self.quark == 'd':
                return self.mN**2 * 2 * ip['gA']
            if self.quark == 's':
                return 0

    def value_eta_pole(self):
        """ Return the coefficient of the pion pole

        The eta pole is given, in terms of the spatial momentum q, by 1 / (q^2 + meta^2)
        """
        ip = self.input_dict
        self.mN = (ip['mproton'] + ip['mneutron'])/2

        if self.nucleon == 'p':
            if self.quark == 'u':
                return self.mN**2 * 2 * (ip['Deltaup'] + ip['Deltadp'] - 2*ip['Deltas'])/3
            if self.quark == 'd':
                return self.mN**2 * 2 * (ip['Deltaup'] + ip['Deltadp'] - 2*ip['Deltas'])/3
            if self.quark == 's':
                return - self.mN**2 * 4 * (ip['Deltaup'] + ip['Deltadp'] - 2*ip['Deltas'])/3
        if self.nucleon == 'n':
            if self.quark == 'u':
                return self.mN**2 * 2 * (ip['Deltaup'] + ip['Deltadp'] - 2*ip['Deltas'])/3
            if self.quark == 'd':
                return self.mN**2 * 2 * (ip['Deltaup'] + ip['Deltadp'] - 2*ip['Deltas'])/3
            if self.quark == 's':
                return - self.mN**2 * 4 * (ip['Deltaup'] + ip['Deltadp'] - 2*ip['Deltas'])/3


class FS(object):
    def __init__(self, quark, nucleon, input_dict=None):
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

        if input_dict is None:
            self.input_dict = Num_input().input_parameters
            # One should include a warning in case the dictionary
            # does not contain all necessary keys
        else:
            self.input_dict = input_dict

    def value_zero_mom(self):
        """ Return the value of the form factor at zero momentum transfer """
        ip = self.input_dict

        if self.nucleon == 'p':
            if self.quark == 'u':
                return ip['sigmaup']
            if self.quark == 'd':
                return ip['sigmadp']
            if self.quark == 's':
                return ip['sigmas']
        if self.nucleon == 'n':
            if self.quark == 'u':
                return ip['sigmaun']
            if self.quark == 'd':
                return ip['sigmadn']
            if self.quark == 's':
                return ip['sigmas']


class FP(object):
    def __init__(self, quark, nucleon, input_dict=None):
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

        if input_dict is None:
            self.input_dict = Num_input().input_parameters
            # One should include a warning in case the dictionary
            # does not contain all necessary keys
        else:
            self.input_dict = input_dict

    def value_pion_pole(self):
        """ Return the coefficient of the pion pole

        The pion pole is given, in terms of the spatial momentum q, by 1 / (q^2 + mpi0^2)
        """
        ip = self.input_dict
        self.mN = (ip['mproton'] + ip['mneutron'])/2

        if self.nucleon == 'p':
            if self.quark == 'u':
                return self.mN**2 * ip['gA'] * ip['B0mu'] / self.mN 
            if self.quark == 'd':
                return - self.mN**2 * ip['gA'] * ip['B0md'] / self.mN 
            if self.quark == 's':
                return 0
        if self.nucleon == 'n':
            if self.quark == 'u':
                return - self.mN**2 * ip['gA'] * ip['B0mu'] / self.mN 
            if self.quark == 'd':
                return self.mN**2 * ip['gA'] * ip['B0md'] / self.mN 
            if self.quark == 's':
                return 0

    def value_eta_pole(self):
        """ Return the coefficient of the pion pole

        The eta pole is given, in terms of the spatial momentum q, by 1 / (q^2 + meta^2)
        """
        ip = self.input_dict
        self.mN = (ip['mproton'] + ip['mneutron'])/2

        if self.nucleon == 'p':
            if self.quark == 'u':
                return self.mN**2 * (ip['Deltaup'] + ip['Deltadp'] - 2*ip['Deltas'])/3/self.mN * ip['B0mu']
            if self.quark == 'd':
                return self.mN**2 * (ip['Deltaup'] + ip['Deltadp'] - 2*ip['Deltas'])/3/self.mN * ip['B0md']
            if self.quark == 's':
                return - 2 * self.mN**2 * (ip['Deltaup'] + ip['Deltadp'] - 2*ip['Deltas'])/3/self.mN * ip['B0ms']
        if self.nucleon == 'n':
            if self.quark == 'u':
                return self.mN**2 * (ip['Deltaup'] + ip['Deltadp'] - 2*ip['Deltas'])/3/self.mN * ip['B0mu']
            if self.quark == 'd':
                return self.mN**2 * (ip['Deltaup'] + ip['Deltadp'] - 2*ip['Deltas'])/3/self.mN * ip['B0md']
            if self.quark == 's':
                return - 2 * self.mN**2 * (ip['Deltaup'] + ip['Deltadp'] - 2*ip['Deltas'])/3/self.mN * ip['B0ms']


class FT0(object):
    def __init__(self, quark, nucleon, input_dict=None):
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

        if input_dict is None:
            self.input_dict = Num_input().input_parameters
            # One should include a warning in case the dictionary
            # does not contain all necessary keys
        else:
            self.input_dict = input_dict

    def value_zero_mom(self):
        """ Return the value of the form factor at zero momentum transfer """
        ip = self.input_dict

        if self.nucleon == 'p':
            if self.quark == 'u':
                return ip['mu_at_2GeV'] * ip['gTu']
            if self.quark == 'd':
                return ip['md_at_2GeV'] * ip['gTd']
            if self.quark == 's':
                return ip['ms_at_2GeV'] * ip['gTs']
        if self.nucleon == 'n':
            if self.quark == 'u':
                return ip['mu_at_2GeV'] * ip['gTd']
            if self.quark == 'd':
                return ip['md_at_2GeV'] * ip['gTu']
            if self.quark == 's':
                return ip['ms_at_2GeV'] * ip['gTs']


class FT1(object):
    def __init__(self, quark, nucleon, input_dict=None):
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

        if input_dict is None:
            self.input_dict = Num_input().input_parameters
            # One should include a warning in case the dictionary
            # does not contain all necessary keys
        else:
            self.input_dict = input_dict

    def value_zero_mom(self):
        """ Return the value of the form factor at zero momentum transfer """
        ip = self.input_dict

        if self.nucleon == 'p':
            if self.quark == 'u':
                return - ip['mu_at_2GeV'] * ip['BT10up']
            if self.quark == 'd':
                return - ip['md_at_2GeV'] * ip['BT10dp']
            if self.quark == 's':
                return - ip['ms_at_2GeV'] * ip['BT10s']
        if self.nucleon == 'n':
            if self.quark == 'u':
                return - ip['mu_at_2GeV'] * ip['BT10un']
            if self.quark == 'd':
                return - ip['md_at_2GeV'] * ip['BT10dn']
            if self.quark == 's':
                return - ip['ms_at_2GeV'] * ip['BT10s']


class FG(object):
    def __init__(self, nucleon, input_dict=None):
        """ The nuclear form factor FG

        Return the nuclear form factor FG

        Arguments
        ---------
        nucleon = 'p', 'n' -- the nucleon (proton or neutron)

        input_dict (optional) -- a dictionary of hadronic input parameters
                                 (default is Num_input().input_parameters)
        """
        self.nucleon = nucleon

        if input_dict is None:
            self.input_dict = Num_input().input_parameters
            # One should include a warning in case the dictionary
            # does not contain all necessary keys
        else:
            self.input_dict = input_dict

    def value_zero_mom(self):
        """ Return the value of the form factor at zero momentum transfer """
        ip = self.input_dict

        if self.nucleon == 'p':
            return -2*ip['mG']/27
        if self.nucleon == 'n':
            return -2*ip['mG']/27


class FGtilde(object):
    def __init__(self, nucleon, input_dict=None):
        """ The nuclear form factor FGtilde

        Return the nuclear form factor FGtilde

        Arguments
        ---------
        nucleon = 'p', 'n' -- the nucleon (proton or neutron)

        input_dict (optional) -- a dictionary of hadronic input parameters
                                 (default is Num_input().input_parameters)
        """
        self.nucleon = nucleon

        if input_dict is None:
            self.input_dict = Num_input().input_parameters
            # One should include a warning in case the dictionary
            # does not contain all necessary keys
        else:
            self.input_dict = input_dict

    def value_zero_mom(self):
        """ Return the value of the form factor at zero momentum transfer """
        ip = self.input_dict
        self.mtilde = 1/(1/ip['mu_at_2GeV'] + 1/ip['md_at_2GeV'] + 1/ip['ms_at_2GeV'])
        self.mN = (ip['mproton'] + ip['mneutron'])/2
        
        if self.nucleon == 'p':
            return -self.mN * self.mtilde * (ip['Deltaup']/ip['mu_at_2GeV']\
                                              + ip['Deltadp']/ip['md_at_2GeV']\
                                              + ip['Deltas']/ip['ms_at_2GeV'])
        if self.nucleon == 'n':
            return -self.mN * self.mtilde * (ip['Deltaun']/ip['mu_at_2GeV']\
                                              + ip['Deltadn']/ip['md_at_2GeV']\
                                              + ip['Deltas']/ip['ms_at_2GeV'])

    def value_pion_pole(self):
        """ Return the coefficient of the pion pole

        The pion pole is given, in terms of the spatial momentum q, by q^2 / (q^2 + mpi0^2)
        """
        ip = self.input_dict
        self.mtilde = 1/(1/ip['mu_at_2GeV'] + 1/ip['md_at_2GeV'] + 1/ip['ms_at_2GeV'])
        self.mN = (ip['mproton'] + ip['mneutron'])/2

        if self.nucleon == 'p':
            return self.mN * self.mtilde * ip['gA'] * (1/ip['mu_at_2GeV'] - 1/ip['md_at_2GeV']) / 2
        if self.nucleon == 'n':
            return - self.mN * self.mtilde * ip['gA'] * (1/ip['mu_at_2GeV'] - 1/ip['md_at_2GeV']) / 2

    def value_eta_pole(self):
        """ Return the coefficient of the eta pole

        The eta pole is given, in terms of the spatial momentum q, by q^2 / (q^2 + meta^2)
        """
        ip = self.input_dict
        self.mtilde = 1/(1/ip['mu_at_2GeV'] + 1/ip['md_at_2GeV'] + 1/ip['ms_at_2GeV'])
        self.mN = (ip['mproton'] + ip['mneutron'])/2

        if self.nucleon == 'p':
            return self.mN * self.mtilde * (ip['Deltaup'] + ip['Deltadp'] - 2*ip['Deltas'])\
                * (1/ip['mu_at_2GeV'] + 1/ip['md_at_2GeV'] - 2/ip['ms_at_2GeV']) / 6
        if self.nucleon == 'n':
            return self.mN * self.mtilde * (ip['Deltaun'] + ip['Deltadn'] - 2*ip['Deltas'])\
                * (1/ip['mu_at_2GeV'] + 1/ip['md_at_2GeV'] - 2/ip['ms_at_2GeV']) / 6


