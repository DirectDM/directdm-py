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



class CmuEW(object):
    def __init__(self, Wilson, ADM, muh, mul, Y, d):
        """ Calculate the running of the Wilson coefficients in the unbroken EW theory

        The running takes into account the gauge coupling g1, g2, g3,
        the tau, bottom, and top Yukawas, ytau, yb, yt, and the Higgs self coupling lambda.

        Wilson should be a list of initial conditions for the Wilson coefficients.

        ADM should be a list / array of the seven DM anomalous dimension matrices proportional to g1^2, g2^2, g3^2, ytau^2, yb^2, yt^2, lambda.

        muh is the initial scale.

        mul is the final scale.
        """
        self. Wilson = Wilson
        self.ADM = ADM
        self.muh = muh
        self.mul = mul
        self.Y = Y
        self.d = d

        # Input parameters

        ip = Num_input()

        self.alpha  = 1/ip.aMZinv
        self.el     = np.sqrt(4*np.pi*self.alpha)
        self.MW     = ip.Mw
        self.MZ     = ip.Mz
        self.Mh     = ip.Mh
        self.cw     = self.MW/self.MZ
        self.sw     = np.sqrt(1-self.cw**2)
        self.g1     = self.el/self.cw
        self.g2     = self.el/self.sw
        self.asMZ   = ip.asMZ
        self.gs     = np.sqrt(4*np.pi*self.asMZ)
        self.ytau   = ip.mtau/246*np.sqrt(2)
        self.yb     = ip.mb_at_MZ/246*np.sqrt(2)
        self.yt     = ip.mt_pole/246*np.sqrt(2)
        self.lam = self.g2**2 * self.Mh**2 / self.MW**2 / 2

        # The initial values of the couplings at MZ
        # Need to think more carefully what to use as input!!!
        self.ginit = [self.g1, self.g2, self.gs, self.ytau, self.yb, self.yt, self.lam]


    def _dgdmu(self, g, mu, Y, d):
        """ Calculate the log derivative [i.e. dg/dlog(mu)] of the couplings g1, g2, g3, ytau, yb, yt, lam w.r.t. to mu, at scale mu
        
        Take a 7-vector (list) of couplings g = [g1, g2, g3, ytau, yb, yt, lambda]

        Take the DM quantum numbers d, Y (so far only 1 multiplet)

        Return the derivative -- again a 7-vector
        """
        N = 1
        # The 7x7 matrix of beta functions (Arason et al., Phys.Rev. D46 (1992) 3945-3965, and our calculation)
        # Note the different sign and normalization conventions. 

        # g1, g2, g3, ytau, yb, ty
        g6 = np.array(g[:-1])
        g6_squared = np.array(list(map(lambda x: x**2, g[:-1])))
        beta = np.array([[41/6+Y**2*d*N/3, 0,                        0,  0,    0,    0  ],
                         [0,               -19/6+4*(d**2-1)/4*d*N/9, 0,  0,    0,    0  ],
                         [0,               0,                        -7, 0,    0,    0  ],
                         [-15/4,           -9/4,                     0,  5/2,  3,    3  ],
                         [-5/12,           -9/4,                     -8, 1,    9/2,  3/2],
                         [-17/12,          -9/4,                     -8, 1,    3/2,  9/2]])

        # g1, g2, g3, ytau, yb, ty, lambda
        beta_lam_1 = np.array([-3, -9, 0, 4, 12, 12, 12])

        # g1^2, g2^2, g3^2, ytau^2, yb^2, yt^2
        beta_lam_2 = np.array([[3/4,  9/20, 0,  0,  0,   0],
                               [9/20, 9/4,  0,  0,  0,   0],
                               [0,    0,    0,  0,  0,   0],
                               [0,    0,    0,  -4, 0,   0],
                               [0,    0,    0,  0,  -12, 0],
                               [0,    0,    0,  0,  0,   -12]])

        deriv_list = np.hstack(( np.multiply(g6, np.dot(beta, g6_squared)),\
                                 np.multiply(g[6], np.dot(beta_lam_1[:-1], g6_squared))\
                                 + beta_lam_1[6]*g[6]**2 + np.dot(g6_squared, np.dot(beta_lam_2, g6_squared)) )) / mu / (4*np.pi)**2

        return deriv_list

    def _alphai(self, g_init, mu_init, mu2, Y, d):
        """ Calculate the one-loop running of alpha1, alpha2, alpha3, alphatau, alphab, alphat, lambda in the six-flavor theory
        
        Run from mu_init to mu2. ginit are the couplings defined at scale mu_init.

        Take g's as input and return alpha's
        """

        def deriv(g,mu):
            return self._dgdmu(g, mu, Y, d)
        r = odeint(deriv, g_init, np.array([mu_init, mu2]))
        # Now take just final numbers and make alpha's out of the g's:
        alpha = list(map(lambda x: x**2/4/np.pi, r[1]))
        return alpha

    # def _alphai_interpolate(self, g_init, mu_init, mu1, mu2, mu0, Y, d):
    #     """ Calculate the one-loop running of alpha1, alpha2, alpha3, alphatau, alphab, alphat, lam in the six-flavor theory 
    #     as an interpolating function from mu1 to mu2
        
    #     Interpolate the running between mu1 and mu2 at mu0. ginit are the couplings defined at scale mu_init.

    #     Take g's as input and return alpha's

    #     I THINK THIS METHOD IS NOT USED ANYWHERE AND SHOULD BE REMOVED!!!
    #     """
    #     def deriv(g,mu):
    #         return self._dgdmu(g, mu, Y, d)
    #     def domain(mu):
    #         return np.array([mu_init, mu])
    #     # Maybe we should play with this:
    #     points = 50

    #     assert mu1 != mu2, "This is alphai_interpolate: mu1 and mu2 have to be different!"

    #     r = np.array([list(map(lambda x: x**2/4/np.pi, odeint(deriv, g_init, domain(mu))[1])) for mu in np.linspace(mu1, mu2, points)])

    #     # Create interpolating function for the running couplings
    #     int_fun_list = np.array([interp1d(np.linspace(mu1, mu2, points), r.T[k], kind='cubic')(mu0) for k in range()])
    #     return int_fun_list

    def run(self):
        def deriv(C, mu):
            return sum([np.dot(C,self.ADM[k])*self._alphai(self.ginit, self.MZ, mu, self.Y, self.d)[k]/4/np.pi/mu for k in range(7)])
        r = odeint(deriv, self.Wilson, np.array([self.muh, self.mul]), full_output=1)
        return list(r)
