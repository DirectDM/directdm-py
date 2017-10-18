#!/usr/bin/env python3

import numpy as np
import scipy.integrate as spint
from directdm.num.num_input import Num_input



#-----------------------------------------------#
# The Higgs penguin function from Hisano et al. #
#-----------------------------------------------#

class Higgspenguin(object):
    """ Hisano's two-loop Higgs penguin function [arxiv:1104.0228] """
    def __init__(self, Ychi, Jchi):
        self.Ychi = Ychi
        self.Jchi = Jchi

        # Some input parameters:
        ip = Num_input()

        self.alpha = 1/ip.aMZinv
        self.Mh = ip.Mh
        self.MW = ip.Mw
        self.MZ = ip.Mz
        self.cw = self.MW/self.MZ
        self.sw = np.sqrt(1-self.cw**2)
        self.mt = ip.mt_pole

        # Hisano definitions
        self.auV = 1/4 - 2/3 * self.sw**2
        self.adV = - 1/4 + 1/3 * self.sw**2
        self.auA = - 1/4
        self.adA = 1/4
        self.nchi = 2*self.Jchi+1

    def __my_sqrt(self, z):
        """ sqrt for negative entries """
        if z >= 0:
            return np.sqrt(z)
        elif z < 0:
            return 1j*np.sqrt(-z)
        else:
            raise Exception("imaginary argument of square root")

    def __Gt1(self, z, y):
        return np.real_if_close( - (self.__my_sqrt(z)*(12*y**2 - z*y + z**2))/(3*(4*y-z)**2) \
                                 + (z**(3/2)*(48*y**3 - 20*z*y**2 + 12*z**2*y - z**3))/(6*(4*y-z)**3) * np.log(z) \
                                 + 2*z**(3/2)*y**2*(4*y-7*z)/(3*(4*y-z)**3) * np.log(4*y) \
                                 - (z**(3/2) * self.__my_sqrt(y) * (16*y**3 - 4*(2+7*z)*y**2 + 14*(2+z) + 5*z))/\
                                   (3*(4*y-z)**3*self.__my_sqrt(1-y)) * np.arctan(self.__my_sqrt((1-y)/y)) \
                                 - ((48*y**3 - z**3)*(z**2-2*z+4) - 4*z*(5*z**2 - 10*z + 44)*y**2 + 12*z**3*(z-2)*y)/\
                                   (3*(4*y-z)**3*self.__my_sqrt(4-z)) * np.arctan(self.__my_sqrt((4-z)/z)))

    def __Gt2(self, z, y):
        return np.real_if_close(   (self.__my_sqrt(z)*(2*y-z))/(4*y-z) - (z**(3/2)*(8*y**2 - 8*y*z +z**2))/(2*(4*y-z)**2) * np.log(z) \
                                 - 4*z**(3/2)*y**2/(4*y-z)**2 * np.log(4*y) \
                                 - 4*z**(3/2)*y**2/(4*y-z)**2 * np.arctan(self.__my_sqrt((1-y)/y)) \
                                 - (8*z*(z**2-2*z+1)*y - (z**2-2*z+4)*(8*y**2+z**2))/\
                                   ((4*y-z)**2*self.__my_sqrt(4-z)) * np.arctan(self.__my_sqrt((4-z)/z)))

    def __gtnolog(self, z, y):
        return self.auV**2 * self.__Gt1(z, y) + self.auA**2 * self.__Gt2(z, y)

    def __I1(self, y, z):
        def integrand(t):
            return ((self.__my_sqrt(t+4)-self.__my_sqrt(t)) * (np.log(self.__my_sqrt(t+4*y)+self.__my_sqrt(t))\
                     - np.log(self.__my_sqrt(t+4*y)-self.__my_sqrt(t))))/\
                   ((t+z)**2 * (t+4*y)**(5/2) * t)
        return spint.quad(integrand, 0, np.inf)[0]

    def __I2(self, y, z):
        def integrand(t):
            return ((t + 2 - self.__my_sqrt(t+4)*self.__my_sqrt(t)) * (np.log(self.__my_sqrt(t+4*y)+self.__my_sqrt(t))\
                     - np.log(self.__my_sqrt(t+4*y)-self.__my_sqrt(t))))/\
                   ((t+z)**2 * (t+4*y)**(5/2) * t**(1/2))/2
        return spint.quad(integrand, 0, np.inf)[0]

    def __I3(self, y, z):
        def integrand(t):
            return ((self.__my_sqrt(t+4)-self.__my_sqrt(t)) * (np.log(self.__my_sqrt(t+4*y)+self.__my_sqrt(t))\
                     - np.log(self.__my_sqrt(t+4*y)-self.__my_sqrt(t))))/\
                   ((t+z)**2 * (t+4*y)**(5/2))
        return spint.quad(integrand, 0, np.inf)[0]

    def __I4(self, y, z):
        def integrand(t):
            return ((t + 2 - self.__my_sqrt(t+4)*self.__my_sqrt(t)) * t**(1/2) * (np.log(self.__my_sqrt(t+4*y)+self.__my_sqrt(t))\
                     - np.log(self.__my_sqrt(t+4*y)-self.__my_sqrt(t))))/\
                   ((t+z)**2 * (t+4*y)**(5/2))/2
        return spint.quad(integrand, 0, np.inf)[0]

    def __gtlog(self, z, y):
        A1 = -2*self.auV**2 + 4*self.auA**2
        A2 = -self.auV**2 + self.auA**2
        return 4*z**(3/2)*y**2 * (A1 * y * (self.__I1(y,z) + self.__I2(y,z)) + A2 * (self.__I3(y,z) + self.__I4(y,z)))

    def __gt(self, z, y):
        return self.__gtnolog(z, y) + self.__gtlog(z, y)

    def __gB1(self, x):
        return np.real_if_close( -1/24 * self.__my_sqrt(x) * (x*np.log(x) - 2) + ((x**2-2*x+4) \
                                       * np.arctan(2*self.__my_sqrt(1-x/4)/self.__my_sqrt(x)))/(24*self.__my_sqrt(1-x/4)))

    def __gB31(self, x, y):
        return np.real_if_close( - x**(3/2)/(12*(y-x)) - x**(3/2)*y**2/(24*(y-x)**2) * np.log(y) - x**(5/2)*(x-2*y)/(24*(y-x)**2) * np.log(x) \
                                 - x**(3/2)*y**(1/2)*(y+2)*self.__my_sqrt(4-y)/(12*(y-x)**2) * np.arctan(self.__my_sqrt((4-y)/y)) \
                                 + x*(x**3 - 2*(y+1)*x**2 + 4*(y+1)*x + 4*y)/(12*(y-x)**2*self.__my_sqrt(4-x)) * np.arctan(self.__my_sqrt((4-x)/x))) 

    def __gB32(self, x, y):
        return np.real_if_close(  - x**(3/2)*y/(12*(y-x)**2) - x**(5/2)*y**2/(24*(y-x)**3) * np.log(y) - x**(5/2)*y**2/(24*(y-x)**3) * np.log(x) \
                                 + x**(3/2)*y**(1/2)*(-6*y+x*y**2-2*x*y-2*x)/(12*(y-x)**3*self.__my_sqrt(4-y)) * np.arctan(self.__my_sqrt((4-y)/y)) \
                                 - x*y*(x**2*y-2*x*y-6*x-2*y)/(12*(y-x)**3*self.__my_sqrt(4-x)) * np.arctan(self.__my_sqrt((4-x)/x)))

    def __gB3(self, x, y):
        return self.__gB31(x,y) + self.__gB32(x,y)

    def __gW(self, w, y):
        return 2*self.__gB1(w) + self.__gB3(w,y)

    def __gZ(self, z, y):
        return ( 2*(self.auV**2 + self.auA**2) + 3*(self.adV**2 + self.adA**2) - 2*(self.auV**2 - self.auA**2) - 2*(self.adV**2 - self.adA**2))\
               * 4*self.__gB1(z) + self.__gt(z,y)

    def hisano_fbc(self, mchi):
        """ Hisano's loop function. 
        
        We include only f_G^(b) and f_G^(c) as the other has an obvious typo.  
        
        Note that we multiply Hisano's loop function by 12pi/alphas 
        """
        y = self.mt**2/mchi**2
        w = self.MW**2/mchi**2
        z = self.MZ**2/mchi**2
        Y = self.Ychi/2

        return np.real(3*self.alpha**2/self.sw**4 * ((self.nchi**2 - (4*Y**2+1))/(8*self.MW**3) * self.__gW(w,y) + Y**2/(4*self.MZ**3*self.cw**4) * self.__gZ(z,y)))
        #return np.real_if_close(3*self.alpha**2/self.sw**2 * ((self.nchi**2 - (4*Y**2+1))/(8*self.MW**3) * self.__gW(w,y)\
        #                                                        + Y**2/(4*self.MZ**3*self.cw**4) * self.__gZ(z,y)))

    def oneloop_ew(self, mchi):
        """The result is valid for all input values and gives (in principle) a real output."""
        w = self.MW**2/mchi**2
        z = self.MZ**2/mchi**2
        Y = self.Ychi/2
        # def f(x):
        #     if x > 4:
        #         out = np.real_if_close(2*(x**2-2*x-2)/(np.sqrt(x-4)) * np.log(np.sqrt(x)/2 + np.sqrt(x/4-1)) + np.sqrt(x)*(2-x*np.log(x)))
        #     if x < 4:
        #         out = np.real_if_close(2*(x**2-2*x-2)/(np.sqrt(4-x)*1j) * np.log(np.sqrt(x)/2 + np.sqrt(1-x/4)*1j) + np.sqrt(x)*(2-x*np.log(x)))
        #     assert np.imag(out) == 0, "Imaginary part of Higgs Penguin should not appear"
        #     return out
        def gH(x):
            bx = np.sqrt(1-x/4+0*1j)
            out = np.real_if_close(-2/bx * (2 + 2*x - x**2) * np.arctan(2*bx/np.sqrt(x)) + 2*np.sqrt(x) * (2 - x*np.log(x)))
            return out
        # return (self.alpha)**2/(2*self.MW*self.Mh**2*self.sw**4) * ( (self.Jchi*(self.Jchi+1)-self.Ychi**2/4)*f(w) + self.Ychi**2/4/self.cw**3*f(z) )
        return (self.alpha)**2/(4*self.Mh**2*self.sw**4) * ((self.nchi**2 - (4*Y**2+1))/(8*self.MW) * gH(w) + Y**2/(4*self.MZ*self.cw**4) * gH(z))

    def oneloop_light(self, mchi):
        return -3*(self.alpha)**2/(8*self.MW**2*self.Mh**2*self.sw**4*self.cw**2) * mchi * \
                  ( self.Ychi**2 + self.cw**2 * (4*self.Jchi*(self.Jchi+1) - self.Ychi**2) )

    def twoloop_ew_fa(self, mchi):
        return - self.oneloop_ew(mchi)

