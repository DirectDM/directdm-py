#!/usr/bin/env python3

import sys
import numpy as np
import scipy as sp
from scipy.integrate import ode
from scipy.integrate import odeint
from scipy.special import zetac
from scipy.interpolate import interp1d
from scipy.linalg import expm
from ..num.num_input import Num_input


### The Riemann Zeta Function

def my_zeta(x):
    return zetac(x)+1


############################
### QCD running          ###
############################

# The QCD beta function

class QCD_beta(object):
    """ The QCD beta function """
    def __init__(self, nf, loop):
        self.nf = nf
        self.loop = loop

    def chet(self):
        """ Conventions as in Chetyrkin, Kuehn, Steinhauser, arXiv:hep-ph/0004189 """
        if self.loop == 1:
            return (11 - 2/3 * self.nf)/4
        if self.loop == 2:
            return (102 - 38/3 * self.nf)/16
        if self.loop == 3:
            return (2857/2 - 5033/18 * self.nf + 325/54 * self.nf**2)/64

    def trad(self):
        """ The more traditional normalization """
        if self.loop == 1:
            return (11 - 2/3 * self.nf)
        if self.loop == 2:
            return (102 - 38/3 * self.nf)
        if self.loop == 3:
            return (2857/2 - 5033/18 * self.nf + 325/54 * self.nf**2)


# The QCD anomalous dimension for the quark mass

class QCD_gamma(object):
    """ The QCD gamma function """
    def __init__(self, nf, loop):
        self.nf = nf
        self.loop = loop

    def trad(self):
        if self.loop == 1:
            return 8


class AlphaS(object):
    """ The strong coupling constant """

    def __init__(self, nf, loop):
        self.nf = nf
        self.loop = loop

        #----------------------#
        #--- Some constants ---#
        #----------------------#

        ip = Num_input()
        self.MZ = ip.Mz
        self.asMZ = ip.asMZ
        self.mc_at_2GeV = ip.mc_at_2GeV #GeV
        self.mb_at_mb = ip.mb_at_mb #GeV

    def decouple_down_MSbar(self, alphasatmu, mu, mh):
        """ Decoupling of the strong coupling from nf to (nf - 1) at scale mu, at heavy quark mass mh

        Input is alphas(mu,nf), output is alphas(mu,nf-1)
        """
        if self.loop == 1:
            return alphasatmu * (1 - 1/6 * np.log(mu**2/mh**2) * (alphasatmu/np.pi))
        if self.loop == 2:
            return alphasatmu * (1 - 1/6 * np.log(mu**2/mh**2) * (alphasatmu/np.pi)
                                   + (11/72 - 11/24 * np.log(mu**2/mh**2) + 1/36 * np.log(mu**2/mh**2)**2) * (alphasatmu/np.pi)**2)
        if self.loop == 3:
            return alphasatmu * (1 - 1/6 * np.log(mu**2/mh**2) * (alphasatmu/np.pi)
                                   + (11/72 - 11/24 * np.log(mu**2/mh**2) + 1/36 * np.log(mu**2/mh**2)**2) * (alphasatmu/np.pi)**2
                                   + (564731/124416 - 82043/27648 * my_zeta(3) - 955/576 * np.log(mu**2/mh**2) + 53/576 * np.log(mu**2/mh**2)**2
                                      - 1/216 * np.log(mu**2/mh**2)**3 + (self.nf-1) * ( -2633/31104 + 67/576 * np.log(mu**2/mh**2)
                                      - 1/36 * np.log(mu**2/mh**2)**2 ) ) * (alphasatmu/np.pi)**3 )


    def decouple_up_MSbar(self, alphasatmu, mu, mh):
        """ Decoupling of the strong coupling from (nf-1) to nf at scale mu, at heavy quark mass mh

        Input is alphas(mu,nf-1), output is alphas(mu,nf)
        """
        if self.loop == 1:
            return alphasatmu * (1 + 1/6 * np.log(mu**2/mh**2) * (alphasatmu/np.pi))
        if self.loop == 2:
            return alphasatmu * (1 + 1/6 * np.log(mu**2/mh**2) * (alphasatmu/np.pi)
                                   + (- 11/72 + 11/24 * np.log(mu**2/mh**2) + 1/36 * np.log(mu**2/mh**2)**2) * (alphasatmu/np.pi)**2)
        if self.loop == 3:
            return alphasatmu * (1 - 1/6 * np.log(mu**2/mh**2) * (alphasatmu/np.pi)
                                   + (- 11/72 + 11/24 * np.log(mu**2/mh**2) + 1/36 * np.log(mu**2/mh**2)**2) * (alphasatmu/np.pi)**2
                                   + (- 564731/124416 + 82043/27648 * my_zeta(3) + 2645/1728 * np.log(mu**2/mh**2) + 167/576 * np.log(mu**2/mh**2)**2
                                      + 1/216 * np.log(mu**2/mh**2)**3 + (self.nf-1) * ( 2633/31104 - 67/576 * np.log(mu**2/mh**2)
                                      + 1/36 * np.log(mu**2/mh**2)**2 ) ) * (alphasatmu/np.pi)**3 )

    def __dalphasdmu(self, mu, alphas, nf):
        if self.loop == 1:
            return 2 * np.pi / mu * ( - QCD_beta(nf, 1).chet() * (alphas/np.pi)**2 )
        if self.loop == 2:
            return 2 * np.pi / mu * ( - QCD_beta(nf, 1).chet() * (alphas/np.pi)**2 - QCD_beta(nf, 2).chet() * (alphas/np.pi)**3 )
        if self.loop == 3:
            return 2 * np.pi / mu * ( - QCD_beta(nf, 1).chet() * (alphas/np.pi)**2 - QCD_beta(nf, 2).chet() * (alphas/np.pi)**3 - QCD_beta(nf, 3).chet() * (alphas/np.pi)**4 )

    def solve_rge(self, as_at_mu, mu, mu0):
        """The running strong coupling
    
        Run from scale mu to scale mu0, initial condition alphas(mu) = as_at_mu
        """
        def deriv(alphas, mu):
            return self.__dalphasdmu(mu, alphas, self.nf)
        r = odeint(deriv, as_at_mu, np.array([mu, mu0]))
        return list(r)[1][0]

    def __solve_rge_nf(self, as_at_mu, mu, mu0, nf):
        """The running strong coupling
    
        Run from scale mu to scale mu0, with initial condition alphas(mu) = as_at_mu, and nf active flavors
        """
        def deriv(alphas, mu):
            return self.__dalphasdmu(mu, alphas, nf)
        r = odeint(deriv, as_at_mu, np.array([mu, mu0]))
        return list(r)[1][0]

    def run(self, mu0):
        """ Run the strong coupling 

        Start value is alphas(MZ), decoupling at quark thresholds is performed automatically. 
        """
        if self.nf == 5:
            return self.solve_rge(self.asMZ, self.MZ, mu0)
        if self.nf == 4:
            as5_mb = self.__solve_rge_nf(self.asMZ, self.MZ, self.mb_at_mb, 5)
            as4_mb = self.decouple_down_MSbar(as5_mb, self.mb_at_mb, self.mb_at_mb)
            return self.__solve_rge_nf(as4_mb, self.mb_at_mb, mu0, 4)
        if self.nf == 3:
            as5_mb = self.__solve_rge_nf(self.asMZ, self.MZ, self.mb_at_mb, 5)
            as4_mb = self.decouple_down_MSbar(as5_mb, self.mb_at_mb, self.mb_at_mb)
            as4_mc = self.__solve_rge_nf(as4_mb, self.mb_at_mb, 2, 4)
            as3_mc = self.decouple_down_MSbar(as4_mc, 2, self.mc_at_2GeV)
            return self.__solve_rge_nf(as4_mc, 2, mu0, 3)

### Future: class should be C_QCD, given the Wilson coefficient at different scales

class RGE(object):

    def __init__(self, adm, nf):
        self.adm = adm
        self.nf = nf

    def U0(self, asmuh, asmul):
        """The leading order (QCD) RG evolution matrix in f-flavor QCD -- matrix exponentiation """
        b0 = QCD_beta(self.nf, 1).trad()
        return expm(np.log(asmuh/asmul) * np.array(np.transpose(self.adm[0]))/(2*b0))

    def U0_as2(self, asmuh, asmul):
        """The leading order (QCD) RG evolution matrix in f-flavor QCD, for ADM that starts at order alphas^2 """
        b0 = QCD_beta(self.nf, 1).trad()
        return expm((asmuh-asmul)/(4*np.pi) * np.array(np.transpose(self.adm[0]))/(2*b0))


