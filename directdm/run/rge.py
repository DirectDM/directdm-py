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
        self.mc_at_3GeV = ip.mc_at_3GeV #GeV
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
            as4_mc = self.__solve_rge_nf(as4_mb, self.mb_at_mb, 3, 4)
            as3_mc = self.decouple_down_MSbar(as4_mc, 3, self.mc_at_3GeV)
            return self.__solve_rge_nf(as4_mc, 3, mu0, 3)

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



class CmuEW(object):
    def __init__(self, Wilson, ADM, muh, mul, Y, d, s1, s2, s3, st):
        """ Calculate the running of the Wilson coefficients in the unbroken EW theory

        The running takes into account the gauge coupling g1, g2, g3, and the top Yukawa yt.

        Wilson should be a list of initial conditions for the Wilson coefficients.

        ADM should be a list / array of four anomalous dimension matrices for g1, g2, g3, yt.

        muh is the initial scale.

        mul is the final scale.

        s1, s2, s3, st = 1 / 0 switches g1, g2, gs, yt on / off
        """
        self. Wilson = Wilson
        self.ADM = ADM
        self.muh = muh
        self.mul = mul
        self.Y = Y
        self.d = d
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.st = st

        # Input parameters

        ip = Num_input()

        self.alpha = 1/ip.aMZinv
        self.el = np.sqrt(4*np.pi*self.alpha)
        self.MW = ip.Mw
        self.MZ = ip.Mz
        self.Mh = ip.Mh
        self.cw = self.MW/self.MZ
        self.sw = np.sqrt(1-self.cw**2)
        self.g1 = self.el/self.cw
        self.g2 = self.el/self.sw
        self.asMZ = ip.asMZ
        self.gs = np.sqrt(4*np.pi*self.asMZ)
        self.yt = ip.mt_pole/246*np.sqrt(2)

        # The initial values of the couplings at MZ
        self.ginit = [self.g1*self.s1, self.g2*self.s2, self.gs*self.s3, self.yt*self.st]


    def _dgdmu(self, g, mu, Y, d):
        """ Calculate the log derivative of the couplings g1, g2, g3, yt w.r.t. to mu, at scale mu
        
        Takes a 4-vector (list) of couplings g = [g1,g2,g3,yt]

        Takes the DM quantum numbers d, Y (so far only 1 multiplet)

        Returns the derivative -- again a 4-vector
        """
        N = 1
        # The 4x4 matrix of beta functions (Arason et al., Phys.Rev. D46 (1992) 3945-3965, and our calculation)
        beta = [[-41/6-Y**2*d*N/3,0,0,0],
                [0,19/6-4*(d**2-1)/4*d*N/9,0,0],
                [0,0,7,0],
                [17/12,9/4,8,-9/2]]
        deriv_list = [sum([ -g[k]*beta[k][i]*g[i]**2 / mu / (4*np.pi)**2 for i in range(4)]) for k in range(4)]
        return deriv_list

    def _alphai(self, g_init, mu_init, mu2, Y, d):
        """ Calculate the one-loop running of alpha1, alpha2, alpha3, alphat in 6-flavor theory
        
        Run from mu_init to mu2. ginit are the couplings defined at scale mu_init.

        Careful, takes g's as input and returns alpha's
        """

        def deriv(g,mu):
            return self._dgdmu(g, mu, Y, d)
        r = odeint(deriv, g_init, np.array([mu_init, mu2]))
        # Now take just final numbers and make alpha's out of the g's
        alpha = list(map(lambda x: x**2/4/np.pi, r[1]))
        return alpha

    def _alphai_interpolate(self, g_init, mu_init, mu1, mu2, mu0, Y, d):
        """ Calculate the one-loop running of alpha1, alpha2, alpha3, alphat in 6-flavor theory as an interpolating function from mu1 to mu2
        
        Interpolate the running between mu1 and mu2 at mu0. ginit are the couplings defined at scale mu_init.

        Careful, it takes g's as input and returns alpha's
        """
        def deriv(g,mu):
            return self._dgdmu(g, mu, Y, d)
        def domain(mu):
            return np.array([mu_init, mu])
        points = 50
        assert mu1 != mu2, "This is alphai_interpolate: mu1 and mu2 have to be different!"
        r = np.array([list(map(lambda x: x**2/4/np.pi, odeint(deriv, g_init, domain(mu))[1])) for mu in np.linspace(mu1, mu2, points)])
        # Create interpolating function for the running couplings
        int_fun_list = np.array([interp1d(np.linspace(mu1, mu2, points), r.T[k], kind='cubic')(mu0) for k in range(4)])
        return int_fun_list

    def run(self):
        def deriv(C, mu):
            return sum([np.dot(C,self.ADM[k])*self._alphai(self.ginit, self.MZ, mu, self.Y, self.d)[k]/4/np.pi/mu for k in range(4)])
        r = odeint(deriv, self.Wilson, np.array([self.muh, self.mul]), full_output=1)
        return list(r)
