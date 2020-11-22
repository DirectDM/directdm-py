#!/usr/bin/env python3

import numpy as np
import scipy.integrate as spint
from directdm.num.num_input import Num_input



#-----------------------------------------------#
# The Higgs penguin function from Hisano et al. #
#-----------------------------------------------#

class Higgspenguin(object):
    def __init__(self, dchi, input_dict=None):
        """ Hisano's two-loop Higgs penguin function [arxiv:1104.0228] 

            input_dict (optional) -- a dictionary of hadronic input parameters
                                     (default is Num_input().input_parameters)
        """
        self.dchi = dchi

        if input_dict is None:
            self.input_dict = Num_input().input_parameters
            # One should include a warning in case the dictionary
            # does not contain all necessary keys
        else:
            self.input_dict = input_dict


        self.alpha = 1/self.input_dict['aMZinv']
        self.Mh = self.input_dict['Mh']
        self.MW = self.input_dict['Mw']
        self.MZ = self.input_dict['Mz']
        self.sw = np.sqrt(self.input_dict['sw2_MSbar'])
        self.cw = np.sqrt(1-self.sw**2)
        self.mt = self.input_dict['mt_at_MZ']

        # Hisano definitions
        self.auV = 1/4 - 2/3 * self.sw**2
        self.adV = - 1/4 + 1/3 * self.sw**2
        self.auA = - 1/4
        self.adA = 1/4

    def _my_sqrt(self, z):
        """ sqrt for negative entries """
        if z >= 0:
            return np.sqrt(z)
        elif z < 0:
            return 1j*np.sqrt(-z)
        else:
            raise Exception("imaginary argument of square root")

    def _Gt1(self, z, y):
        return np.real_if_close( - (self._my_sqrt(z)*(12*y**2 - z*y + z**2))/(3*(4*y-z)**2) \
                                 + (z**(3/2)*(48*y**3 - 20*z*y**2 + 12*z**2*y - z**3))/(6*(4*y-z)**3) * np.log(z) \
                                 + 2*z**(3/2)*y**2*(4*y-7*z)/(3*(4*y-z)**3) * np.log(4*y) \
                                 - (z**(3/2) * self._my_sqrt(y) * (16*y**3 - 4*(2+7*z)*y**2 + 14*(2+z) + 5*z))/\
                                   (3*(4*y-z)**3*self._my_sqrt(1-y)) * np.arctan(self._my_sqrt((1-y)/y)) \
                                 - ((48*y**3 - z**3)*(z**2-2*z+4) - 4*z*(5*z**2 - 10*z + 44)*y**2 + 12*z**3*(z-2)*y)/\
                                   (3*(4*y-z)**3*self._my_sqrt(4-z)) * np.arctan(self._my_sqrt((4-z)/z)))

    def _Gt2(self, z, y):
        return np.real_if_close(   (self._my_sqrt(z)*(2*y-z))/(4*y-z) - (z**(3/2)*(8*y**2 - 8*y*z +z**2))/(2*(4*y-z)**2) * np.log(z) \
                                 - 4*z**(3/2)*y**2/(4*y-z)**2 * np.log(4*y) \
                                 - 4*z**(3/2)*y**2/(4*y-z)**2 * np.arctan(self._my_sqrt((1-y)/y)) \
                                 - (8*z*(z**2-2*z+1)*y - (z**2-2*z+4)*(8*y**2+z**2))/\
                                   ((4*y-z)**2*self._my_sqrt(4-z)) * np.arctan(self._my_sqrt((4-z)/z)))

    def _gtnolog(self, z, y):
        return self.auV**2 * self._Gt1(z, y) + self.auA**2 * self._Gt2(z, y)

    def _I1(self, y, z):
        def integrand(t):
            return ((self._my_sqrt(t+4)-self._my_sqrt(t)) * (np.log(self._my_sqrt(t+4*y)+self._my_sqrt(t))\
                     - np.log(self._my_sqrt(t+4*y)-self._my_sqrt(t))))/\
                   ((t+z)**2 * (t+4*y)**(5/2) * t)
        return spint.quad(integrand, 0, np.inf)[0]

    def _I2(self, y, z):
        def integrand(t):
            return ((t + 2 - self._my_sqrt(t+4)*self._my_sqrt(t)) * (np.log(self._my_sqrt(t+4*y)+self._my_sqrt(t))\
                     - np.log(self._my_sqrt(t+4*y)-self._my_sqrt(t))))/\
                   ((t+z)**2 * (t+4*y)**(5/2) * t**(1/2))/2
        return spint.quad(integrand, 0, np.inf)[0]

    def _I3(self, y, z):
        def integrand(t):
            return ((self._my_sqrt(t+4)-self._my_sqrt(t)) * (np.log(self._my_sqrt(t+4*y)+self._my_sqrt(t))\
                     - np.log(self._my_sqrt(t+4*y)-self._my_sqrt(t))))/\
                   ((t+z)**2 * (t+4*y)**(5/2))
        return spint.quad(integrand, 0, np.inf)[0]

    def _I4(self, y, z):
        def integrand(t):
            return ((t + 2 - self._my_sqrt(t+4)*self._my_sqrt(t)) * t**(1/2) * (np.log(self._my_sqrt(t+4*y)+self._my_sqrt(t))\
                     - np.log(self._my_sqrt(t+4*y)-self._my_sqrt(t))))/\
                   ((t+z)**2 * (t+4*y)**(5/2))/2
        return spint.quad(integrand, 0, np.inf)[0]

    def _gtlog(self, z, y):
        A1 = -2*self.auV**2 + 4*self.auA**2
        A2 = -self.auV**2 + self.auA**2
        return 4*z**(3/2)*y**2 * (A1 * y * (self._I1(y,z) + self._I2(y,z)) + A2 * (self._I3(y,z) + self._I4(y,z)))

    def _gt(self, z, y):
        return self._gtnolog(z, y) + self._gtlog(z, y)

    def _gB1(self, x):
        return np.real_if_close( -1/24 * self._my_sqrt(x) * (x*np.log(x) - 2) + ((x**2-2*x+4) \
                                       * np.arctan(2*self._my_sqrt(1-x/4)/self._my_sqrt(x)))/(24*self._my_sqrt(1-x/4)))

    def _gB31(self, x, y):
        return np.real_if_close( - x**(3/2)/(12*(y-x)) - x**(3/2)*y**2/(24*(y-x)**2) * np.log(y) - x**(5/2)*(x-2*y)/(24*(y-x)**2) * np.log(x) \
                                 - x**(3/2)*y**(1/2)*(y+2)*self._my_sqrt(4-y)/(12*(y-x)**2) * np.arctan(self._my_sqrt((4-y)/y)) \
                                 + x*(x**3 - 2*(y+1)*x**2 + 4*(y+1)*x + 4*y)/(12*(y-x)**2*self._my_sqrt(4-x)) * np.arctan(self._my_sqrt((4-x)/x))) 

    def _gB32(self, x, y):
        return np.real_if_close(  - x**(3/2)*y/(12*(y-x)**2) - x**(5/2)*y**2/(24*(y-x)**3) * np.log(y) + x**(5/2)*y**2/(24*(y-x)**3) * np.log(x) \
                                 + x**(3/2)*y**(1/2)*(-6*y+x*y**2-2*x*y-2*x)/(12*(y-x)**3*self._my_sqrt(4-y)) * np.arctan(self._my_sqrt((4-y)/y)) \
                                 - x*y*(x**2*y-2*x*y-6*x-2*y)/(12*(y-x)**3*self._my_sqrt(4-x)) * np.arctan(self._my_sqrt((4-x)/x)))

    def _gB3(self, x, y):
        return self._gB31(x,y) + self._gB32(x,y)

    def _gW(self, w, y):
        return 2*self._gB1(w) + self._gB3(w,y)

    def _gZ(self, z, y):
        return ( 2*(self.auV**2 + self.auA**2) + 3*(self.adV**2 + self.adA**2) - 2*(self.auV**2 - self.auA**2) - 2*(self.adV**2 - self.adA**2))\
               * 4*self._gB1(z) + self._gt(z,y)

    def hisano_fbc(self, mchi):
        """ Hisano's two-loop function f_G^(b) and f_G^(c).
        
        Note that we multiply Hisano's loop function by 12pi/alphas 
        """
        y = self.mt**2/mchi**2
        w = self.MW**2/mchi**2
        z = self.MZ**2/mchi**2

        return np.real(3*self.alpha**2/self.sw**4 * ((self.dchi**2 - 1)/(8*self.MW**3) * self._gW(w,y)))

    def f_q_hisano(self, mchi):
        """The result is valid for all input values and gives (in principle) a real output."""
        w = self.MW**2/mchi**2
        def gH(x):
            bx = np.sqrt(1-x/4+0*1j)
            out = np.real_if_close(-2/bx * (2 + 2*x - x**2) * np.arctan(2*bx/np.sqrt(x))\
                                   + 2*np.sqrt(x) * (2 - x*np.log(x)))
            return out
        return (self.alpha)**2/(4*self.Mh**2*self.sw**4) * ((self.dchi**2 - 1)/(8*self.MW) * gH(w))

    def d_q_hisano(self, mchi):
        """The result is valid for all input values and gives (in principle) a real output."""
        w = self.MW**2/mchi**2
        def gAV(x):
            bx = np.sqrt(1-x/4+0*1j)
            out = np.real_if_close(1/(24*bx) * np.sqrt(x) * (8 - x - x**2) * np.arctan(2*bx/np.sqrt(x))\
                  - 1/24 * x * (2 - (3+x)*np.log(x)))
            return out
        return (self.alpha)**2/(self.MW**2*self.sw**4) * ((self.dchi**2 - 1)/8 * gAV(w))

    def g_q_1_hisano(self, mchi):
        """The result is valid for all input values and gives (in principle) a real output."""
        w = self.MW**2/mchi**2
        def gT1(x):
            bx = np.sqrt(1-x/4+0*1j)
            out = np.real_if_close(bx/3 * (2 + x**2) * np.arctan(2*bx/np.sqrt(x))\
                                   + np.sqrt(x)/12 * (1 - 2*x - x*(2-x)*np.log(x)))
            return out
        return (self.alpha)**2/(self.sw**4) * ((self.dchi**2 - 1)/(8*self.MW**3) * gT1(w))

    def g_q_2_hisano(self, mchi):
        """The result is valid for all input values and gives (in principle) a real output."""
        w = self.MW**2/mchi**2
        def gT2(x):
            bx = np.sqrt(1-x/4+0*1j)
            out = np.real_if_close(1/bx/4 * x * (2 - 4*x + x**2) * np.arctan(2*bx/np.sqrt(x))\
                                   - np.sqrt(x)/4 * (1 - 2*x - x*(2-x)*np.log(x)))
            return out
        return (self.alpha)**2/(self.sw**4) * ((self.dchi**2 - 1)/(8*self.MW**3) * gT2(w))

    def f_q_light(self, mchi):
        return -3*(self.alpha)**2/(8*self.MW**2*self.Mh**2*self.sw**4) * mchi * (self.dchi**2 - 1)

    def hisano_fa(self, mchi):
        """ Hisano's two-loop function f_G^(a).
        
        Note that we multiply Hisano's loop function by 12pi/alphas 
        """
        return - self.f_q_hisano(mchi)

