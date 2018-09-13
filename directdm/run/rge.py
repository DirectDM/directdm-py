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

    def __init__(self, asMZ, MZ):
        self.asMZ = asMZ
        self.MZ = MZ

    def decouple_down_MSbar(self, alphasatmu, mu, mh, nf, loop):
        """ Decoupling of the strong coupling from nf to (nf - 1) at scale mu, at heavy quark mass mh = mh(mh)

        Input is alphas(mu,nf), output is alphas(mu,nf-1)
        """
        if loop == 1:
            return alphasatmu
        if loop == 2:
            return alphasatmu * (1 - 1/6 * np.log(mu**2/mh**2) * (alphasatmu/np.pi))
        if loop == 3:
            return alphasatmu * (1 - 1/6 * np.log(mu**2/mh**2) * (alphasatmu/np.pi)
                                   + (11/72 - 19/24 * np.log(mu**2/mh**2) + 1/36 * np.log(mu**2/mh**2)**2) * (alphasatmu/np.pi)**2)
        if loop == 4:
            return alphasatmu * (1 - 1/6 * np.log(mu**2/mh**2) * (alphasatmu/np.pi)
                                   + (11/72 - 19/24 * np.log(mu**2/mh**2) + 1/36 * np.log(mu**2/mh**2)**2) * (alphasatmu/np.pi)**2
                                   + (564731/124416 - 82043/27648 * my_zeta(3) - 6793/1728 * np.log(mu**2/mh**2) - 131/576 * np.log(mu**2/mh**2)**2
                                      - 1/216 * np.log(mu**2/mh**2)**3 + (nf-1) * ( -2633/31104 + 281/1728 * np.log(mu**2/mh**2)
                                      ) ) * (alphasatmu/np.pi)**3 )

    # def decouple_up_MSbar(self, alphasatmu, mu, mh):
    #     """ Decoupling of the strong coupling from (nf-1) to nf at scale mu, at heavy quark mass mh = mh(mh)

    #     Input is alphas(mu,nf-1), output is alphas(mu,nf)
    #     """
    #     if loop == 1:
    #         return alphasatmu
    #     if loop == 2:
    #         return alphasatmu * (1 + 1/6 * np.log(mu**2/mh**2) * (alphasatmu/np.pi))
    #     if loop == 3:
    #         return alphasatmu * (1 + 1/6 * np.log(mu**2/mh**2) * (alphasatmu/np.pi)
    #                                + (- 11/72 + 19/24 * np.log(mu**2/mh**2) + 1/36 * np.log(mu**2/mh**2)**2) * (alphasatmu/np.pi)**2)
    #     if loop == 4:
    #         return alphasatmu * (1 - 1/6 * np.log(mu**2/mh**2) * (alphasatmu/np.pi)
    #                                + (- 11/72 + 19/24 * np.log(mu**2/mh**2) + 1/36 * np.log(mu**2/mh**2)**2) * (alphasatmu/np.pi)**2
    #                                + (- 564731/124416 + 82043/27648 * my_zeta(3) + 2191/576 * np.log(mu**2/mh**2) + 511/576 * np.log(mu**2/mh**2)**2
    #                                   + 1/216 * np.log(mu**2/mh**2)**3 + (self.nf-1) * ( 2633/31104 - 281/1728 * np.log(mu**2/mh**2)
    #                                   ) ) * (alphasatmu/np.pi)**3 )

    def __dalphasdmu(self, mu, alphas, nf, loop):
        if loop == 1:
            return 2 * np.pi / mu * ( - QCD_beta(nf, 1).chet() * (alphas/np.pi)**2 )
        if loop == 2:
            return 2 * np.pi / mu * ( - QCD_beta(nf, 1).chet() * (alphas/np.pi)**2 - QCD_beta(nf, 2).chet() * (alphas/np.pi)**3 )
        if loop == 3:
            return 2 * np.pi / mu * ( - QCD_beta(nf, 1).chet() * (alphas/np.pi)**2 - QCD_beta(nf, 2).chet() * (alphas/np.pi)**3\
                                      - QCD_beta(nf, 3).chet() * (alphas/np.pi)**4 )

    def __solve_rge_nf(self, as_at_mu, mu, mu0, nf, loop):
        """The running strong coupling
    
        Run from scale mu to scale mu0, with nf active flavors
        """
        def deriv(alphas, mu):
            return self.__dalphasdmu(mu, alphas, nf, loop)
        r = odeint(deriv, as_at_mu, np.array([mu, mu0]))
        return list(r)[1][0]

    def run(self, dict_mh, dict_mu, mu0, nf, loop):
        """ Run the strong coupling with decoupling at flavor thresholds

        A dictionary of scales for each heavy quark mass mq(mq) should be given. E.g.

        {'mbmb': 4.18, 'mcmc': 1.275}

        A dictionary of scales for each threshold should be given. E.g.

        {'mub': 5, 'muc': 1.3}

        (Depending on nf one or zero can be given)

        The decoupling is always at mq(mq)
        """
        if nf == 5:
            return self.__solve_rge_nf(self.asMZ, self.MZ, mu0, 5, loop)
        if nf == 4:
            as5_mub = self.__solve_rge_nf(self.asMZ, self.MZ, dict_mu['mub'], 5, loop)
            as4_mub = self.decouple_down_MSbar(as5_mub, dict_mu['mub'], dict_mh['mbmb'], 5, loop)
            return self.__solve_rge_nf(as4_mub, dict_mu['mub'], mu0, 4, loop)
        if nf == 3:
            as5_mub = self.__solve_rge_nf(self.asMZ, self.MZ, dict_mu['mub'], 5, loop)
            as4_mub = self.decouple_down_MSbar(as5_mub, dict_mu['mub'], dict_mh['mbmb'], 5, loop)
            as4_muc = self.__solve_rge_nf(as4_mub, dict_mu['mub'], dict_mu['muc'], 4, loop)
            as3_muc = self.decouple_down_MSbar(as4_muc, dict_mu['muc'], dict_mh['mcmc'], 4, loop)
            return self.__solve_rge_nf(as3_muc, dict_mu['muc'], mu0, 3, loop)

### Future: class should be CmuQCD, giving the Wilson coefficient at different scales

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
        the tau, charm, bottom, and top Yukawas, ytau, yc, yb, yt, and the Higgs self coupling lambda.

        Wilson should be a list of initial conditions for the Wilson coefficients.

        ADM should be a list / array of the eight DM anomalous dimension matrices proportional to g1^2, g2^2, g3^2, ytau^2, yc^2, yb^2, yt^2, lambda.

        muh is the initial scale.

        mul is the final scale.
        """
        self.Wilson = Wilson
        self.ADM = ADM
        self.muh = muh
        self.mul = mul
        self.Y = Y
        self.d = d

        # Input parameters

        ip = Num_input()

        self.g1   = ip.g1_at_MZ
        self.g2   = ip.g2_at_MZ
        self.gs   = ip.g3_at_MZ
        self.ytau = ip.ytau_at_MZ
        self.yc   = ip.yc_at_MZ
        self.yb   = ip.yb_at_MZ
        self.yt   = ip.yt_at_MZ
        self.lam  = ip.lam_at_MZ

        # The initial values of the couplings at MZ
        self.MZ    = ip.Mz
        self.ginit = [self.g1, self.g2, self.gs, self.yc, self.ytau, self.yb, self.yt, self.lam]


    def _dgdmu(self, g, mu, Y, d):
        """ Calculate the log derivative [i.e. dg/dlog(mu)] of the couplings g1, g2, g3, yc, ytau, yb, yt, lam w.r.t. to mu, at scale mu
        
        Take a 8-vector (list) of couplings g = [g1, g2, g3, yc, ytau, yb, yt, lambda]

        Take the DM quantum numbers d, Y (so far only 1 multiplet)

        Return the derivative -- again a 8-vector
        """
        N = 1
        # The 8x8 matrix of beta functions (Arason et al., Phys.Rev. D46 (1992) 3945-3965, and our calculation)
        # Note the different sign and normalization conventions. (g1_Arason = 5/3 * g1_Denner; lambda_Arason = 1/4 * lambda_Denner)

        # g1, g2, g3, yc, ytau, yb, yt
        g7 = np.array(g[:-1])
        g7_squared = np.array(list(map(lambda x: x**2, g[:-1])))
        beta = np.array([[41/6+Y**2*d*N/3, 0,                         0,  0,   0,    0,    0  ],
                         [0,               -19/6+4*(d**2-1)/4*d*N/9,  0,  0,   0,    0,    0  ],
                         [0,               0,                        -7, 0,   0,    0,    0  ],
                         [-17/12,          -9/4,                     -8, 9/2, 1,    3/2,  9/2],
                         [-15/4,           -9/4,                      0,  3,   5/2,  3,    3  ],
                         [-5/12,           -9/4,                     -8, 3/2, 1,    9/2,  3/2],
                         [-17/12,          -9/4,                     -8, 9/2, 1,    3/2,  9/2]])

        # g1, g2, g3, yc, ytau, yb, yt, lambda
        beta_lam_1 = np.array([-3/4, -9/4, 0, 3, 1, 3, 3, 3/4])

        # g1^2, g2^2, g3^2, yc^2, ytau^2, yb^2, yt^2
        beta_lam_2 = np.array([[3/4,  3/4,  0,  0,   0,  0,   0  ],
                               [3/4,  9/4,  0,  0,   0,  0,   0  ],
                               [0,    0,    0,  0,   0,  0,   0  ],
                               [0,    0,    0,  -12, 0,  0,   0  ],
                               [0,    0,    0,  0,   -4, 0,   0  ],
                               [0,    0,    0,  0,   0,  -12, 0  ],
                               [0,    0,    0,  0,   0,  0,   -12]])

        deriv_list = np.hstack(( np.multiply(g7, np.dot(beta, g7_squared)),\
                                 np.multiply(g[7], np.dot(beta_lam_1[:-1], g7_squared))\
                                 + beta_lam_1[7]*g[7]**2 + np.dot(g7_squared, np.dot(beta_lam_2, g7_squared)) )) / mu / (4*np.pi)**2

        return deriv_list

    def _alphai(self, g_init, mu_init, mu2, Y, d):
        """ Calculate the one-loop running of alpha1, alpha2, alpha3, alphac, alphatau, alphab, alphat, lambda in the six-flavor theory
        
        Run from mu_init to mu2. ginit are the couplings defined at scale mu_init.

        Take g's as input and return alpha's
        """


        # Runge-Kutta:        
        def deriv(mu, g):
            return self._dgdmu(g, mu, Y, d)
        g0, mu0 = g_init, mu_init
        dmu = mu2 - mu_init
#        r = ode(deriv).set_integrator('lsoda')
        r = ode(deriv).set_integrator('dopri5')
#        r = ode(deriv).set_integrator('vode')
#        r = ode(deriv).set_integrator('dop853')
        r.set_initial_value(g0, mu0)
        solution = r.integrate(r.t+dmu)
        alpha = np.hstack((np.array(list(map(lambda x: x**2/4/np.pi, solution[:-1]))), np.array([solution[-1]/4/np.pi])))

        return alpha


    def run(self):

        def deriv(mu, C):
            return sum([np.dot(C,self.ADM[k])*self._alphai(self.ginit, self.MZ, mu, self.Y, self.d)[k]/4/np.pi/mu for k in range(8)])

        C0, mu0 = self.Wilson, self.muh
        dmu = self.mul - self.muh
        r = ode(deriv).set_integrator('dopri5')
#        r = ode(deriv).set_integrator('lsoda')
#        r = ode(deriv).set_integrator('dop853')
#        r = ode(deriv).set_integrator('vode')
        r.set_initial_value(C0, mu0)
        C = np.array(r.integrate(r.t+dmu))
        return(C)

