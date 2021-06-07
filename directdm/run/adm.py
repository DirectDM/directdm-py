#!/usr/bin/env python3

import numpy as np
import re
from pkg_resources import resource_filename
from ..num.num_input import Num_input
from directdm.run import rge

#-----------------------#
# Conventions and Basis #
#-----------------------#

# The basis of operators in the DM-SM sector below the weak scale (5-flavor EFT) is given by


# dim.5 (2 operators)
#
# 'C51', 'C52',


# dim.6 (32 operators)
#
# 'C61u', 'C61d', 'C61s', 'C61c', 'C61b', 'C61e', 'C61mu', 'C61tau', 
# 'C62u', 'C62d', 'C62s', 'C62c', 'C62b', 'C62e', 'C62mu', 'C62tau',
# 'C63u', 'C63d', 'C63s', 'C63c', 'C63b', 'C63e', 'C63mu', 'C63tau', 
# 'C64u', 'C64d', 'C64s', 'C64c', 'C64b', 'C64e', 'C64mu', 'C64tau',


# dim.7 (129 operators)
#
# 'C71', 'C72', 'C73', 'C74',
# 'C75u', 'C75d', 'C75s', 'C75c', 'C75b', 'C75e', 'C75mu', 'C75tau', 
# 'C76u', 'C76d', 'C76s', 'C76c', 'C76b', 'C76e', 'C76mu', 'C76tau',
# 'C77u', 'C77d', 'C77s', 'C77c', 'C77b', 'C77e', 'C77mu', 'C77tau', 
# 'C78u', 'C78d', 'C78s', 'C78c', 'C78b', 'C78e', 'C78mu', 'C78tau',
# 'C79u', 'C79d', 'C79s', 'C79c', 'C79b', 'C79e', 'C79mu', 'C79tau', 
# 'C710u', 'C710d', 'C710s', 'C710c', 'C710b', 'C710e', 'C710mu', 'C710tau',
# 'C711', 'C712', 'C713', 'C714',
# 'C715u', 'C715d', 'C715s', 'C715c', 'C715b', 'C715e', 'C715mu', 'C715tau', 
# 'C716u', 'C716d', 'C716s', 'C716c', 'C716b', 'C716e', 'C716mu', 'C716tau',
# 'C717u', 'C717d', 'C717s', 'C717c', 'C717b', 'C717e', 'C717mu', 'C717tau', 
# 'C718u', 'C718d', 'C718s', 'C718c', 'C718b', 'C718e', 'C718mu', 'C718tau',
# 'C719u', 'C719d', 'C719s', 'C719c', 'C719b', 'C719e', 'C719mu', 'C719tau', 
# 'C720u', 'C720d', 'C720s', 'C720c', 'C720b', 'C720e', 'C720mu', 'C720tau', 
# 'C721u', 'C721d', 'C721s', 'C721c', 'C721b', 'C721e', 'C721mu', 'C721tau', 
# 'C722u', 'C722d', 'C722s', 'C722c', 'C722b', 'C722e', 'C722mu', 'C722tau',
# 'C723u', 'C723d', 'C723s', 'C723c', 'C723b', 'C723e', 'C723mu', 'C723tau', 
# 'C725',


# dim.8 (12 operators)
#
# 'C81u', 'C81d', 'C81s', 'C82u', 'C82d', 'C82s'
# 'C83u', 'C83d', 'C83s', 'C84u', 'C84d', 'C84s'

# In total, we have 2+32+129+12=175 operators. 
# In total, we have 2+32+129=163 operators w/o dim.8. 


#-----------------------------#
# The QED anomalous dimension #
#-----------------------------#

def ADM_QED(nf):
    """ Return the QED anomalous dimension in the DM-SM sector for nf flavor EFT """
    Qu = 2/3
    Qd = -1/3
    Qe = -1
    nc = 3
    gamma_QED = np.array([[8/3*Qu*Qu*nc, 8/3*Qu*Qd*nc, 8/3*Qu*Qd*nc, 8/3*Qu*Qu*nc,\
                           8/3*Qu*Qd*nc, 8/3*Qu*Qe*nc, 8/3*Qu*Qe*nc, 8/3*Qu*Qe*nc],
                          [8/3*Qd*Qu*nc, 8/3*Qd*Qd*nc, 8/3*Qd*Qd*nc, 8/3*Qd*Qu*nc,\
                           8/3*Qd*Qd*nc, 8/3*Qd*Qe*nc, 8/3*Qd*Qe*nc, 8/3*Qd*Qe*nc],
                          [8/3*Qd*Qu*nc, 8/3*Qd*Qd*nc, 8/3*Qd*Qd*nc, 8/3*Qd*Qu*nc,\
                           8/3*Qd*Qd*nc, 8/3*Qd*Qe*nc, 8/3*Qd*Qe*nc, 8/3*Qd*Qe*nc],
                          [8/3*Qu*Qu*nc, 8/3*Qu*Qd*nc, 8/3*Qu*Qd*nc, 8/3*Qu*Qu*nc,\
                           8/3*Qu*Qd*nc, 8/3*Qu*Qe*nc, 8/3*Qu*Qe*nc, 8/3*Qu*Qe*nc],
                          [8/3*Qd*Qu*nc, 8/3*Qd*Qd*nc, 8/3*Qd*Qd*nc, 8/3*Qd*Qu*nc,\
                           8/3*Qd*Qd*nc, 8/3*Qd*Qe*nc, 8/3*Qd*Qe*nc, 8/3*Qd*Qe*nc],
                          [8/3*Qe*Qu,    8/3*Qe*Qd,    8/3*Qe*Qd,    8/3*Qe*Qu,\
                           8/3*Qe*Qd,    8/3*Qe*Qe,    8/3*Qe*Qe,    8/3*Qe*Qe],
                          [8/3*Qe*Qu,    8/3*Qe*Qd,    8/3*Qe*Qd,    8/3*Qe*Qu,\
                           8/3*Qe*Qd,    8/3*Qe*Qe,    8/3*Qe*Qe,    8/3*Qe*Qe],
                          [8/3*Qe*Qu,    8/3*Qe*Qd,    8/3*Qe*Qd,    8/3*Qe*Qu,\
                           8/3*Qe*Qd,    8/3*Qe*Qe,    8/3*Qe*Qe,    8/3*Qe*Qe]])
    gamma_QED_1 = np.zeros((2,163))
    gamma_QED_2 = np.hstack((np.zeros((8,2)),gamma_QED,np.zeros((8,153))))
    gamma_QED_3 = np.hstack((np.zeros((8,10)),gamma_QED,np.zeros((8,145))))
    gamma_QED_4 = np.zeros((145,163))
    gamma_QED = np.vstack((gamma_QED_1, gamma_QED_2, gamma_QED_3, gamma_QED_4))

    if nf == 5:
        return gamma_QED
    elif nf == 4:
        return np.delete(np.delete(gamma_QED, [6, 14, 22, 30, 42, 50, 58, 66, 74, 82, 94,\
                                               102, 110, 118, 126, 134, 142, 150, 158], 0)\
                                            , [6, 14, 22, 30, 42, 50, 58, 66, 74, 82, 94,\
                                               102, 110, 118, 126, 134, 142, 150, 158], 1)
    elif nf == 3:
        return np.delete(np.delete(gamma_QED, [5,6, 13,14, 21,22, 29,30, 41,42,\
                                               49,50, 57,58, 65,66, 73,74, 81,82,\
                                               93,94, 101,102, 109,110, 117,118,\
                                               125,126, 133,134, 141,142, 149,150, 158,159], 0)\
                                            , [5,6, 13,14, 21,22, 29,30, 41,42,\
                                               49,50, 57,58, 65,66, 73,74, 81,82,\
                                               93,94, 101,102, 109,110, 117,118,\
                                               125,126, 133,134, 141,142, 149,150, 158,159], 1)
    else:
        raise Exception("nf has to be 3, 4 or 5")


def ADM_QED2(nf):
    """ Return the QED anomalous dimension in the DM-SM sector for nf flavor EFT at alpha^2 """

    # Mixing of Q_{11}^(7) into Q_{5,f}^(7) and Q_{12}^(7) into Q_{6,f}^(7),
    # now correctly adapted from Hill et al. [1409.8290]. 
    Qu = 2/3
    Qd = -1/3
    Qe = -1
    gamma_QED2_gf = np.array([[8*Qu**2, 8*Qd**2, 8*Qd**2, 8*Qu**2,\
                              8*Qd**2, 8*Qe**2, 8*Qe**2, 8*Qe**2]])
    gamma_QED2_1 = np.zeros((86,163))
    gamma_QED2_2 = np.hstack((np.zeros((1,38)),gamma_QED2_gf,np.zeros((1,117))))
    gamma_QED2_3 = np.hstack((np.zeros((1,46)),gamma_QED2_gf,np.zeros((1,109))))
    gamma_QED2_4 = np.zeros((75,163))
    gamma_QED2 = np.vstack((gamma_QED2_1, gamma_QED2_2, gamma_QED2_3, gamma_QED2_4))

    if nf == 5:
        return gamma_QED2
    elif nf == 4:
        return np.delete(np.delete(gamma_QED2, [6, 14, 22, 30, 42, 50, 58, 66, 74, 82, 94,\
                                                102, 110, 118, 126, 134, 142, 150, 158], 0)\
                                             , [6, 14, 22, 30, 42, 50, 58, 66, 74, 82, 94,\
                                                102, 110, 118, 126, 134, 142, 150, 158], 1)
    elif nf == 3:
        return np.delete(np.delete(gamma_QED2, [5,6, 13,14, 21,22, 29,30, 41,42,\
                                                49,50, 57,58, 65,66, 73,74, 81,82,\
                                                93,94, 101,102, 109,110, 117,118,\
                                                125,126, 133,134, 141,142, 149,150, 158,159], 0)\
                                             , [5,6, 13,14, 21,22, 29,30, 41,42,\
                                                49,50, 57,58, 65,66, 73,74, 81,82,\
                                                93,94, 101,102, 109,110, 117,118,\
                                                125,126, 133,134, 141,142, 149,150, 158,159], 1)
    else:
        raise Exception("nf has to be 3, 4 or 5")


#------------------------------#
# The QCD anomalous dimensions #
#------------------------------#

def ADM_QCD(nf):
    """ Return the QCD anomalous dimension in the DM-SM sector for nf flavor EFT, when ADM starts at O(alphas) """
    gamma_QCD_T = 32/3 * np.eye(5)
    gt2qq = 64/9
    gt2qg = -4/3
    gt2gq = -64/9
    gt2gg = 4/3*nf
    gamma_twist2 = np.array([[gt2qq, 0,     0,     0,     0,     0,     0,     0,     gt2qg],
                             [0,     gt2qq, 0,     0,     0,     0,     0,     0,     gt2qg],
                             [0,     0,     gt2qq, 0,     0,     0,     0,     0,     gt2qg],
                             [0,     0,     0,     gt2qq, 0,     0,     0,     0,     gt2qg],
                             [0,     0,     0,     0,     gt2qq, 0,     0,     0,     gt2qg],
                             [0,     0,     0,     0,     0,     0,     0,     0,     0    ],
                             [0,     0,     0,     0,     0,     0,     0,     0,     0    ],
                             [0,     0,     0,     0,     0,     0,     0,     0,     0    ],
                             [gt2gq, gt2gq, gt2gq, gt2gq, gt2gq, 0,     0,     0,     gt2gg]])
    gamma_QCD_1 = np.zeros((70,163))
    gamma_QCD_2 = np.hstack((np.zeros((5,70)), gamma_QCD_T, np.zeros((5,88))))
    gamma_QCD_3 = np.zeros((3,163))
    gamma_QCD_4 = np.hstack((np.zeros((5,78)), gamma_QCD_T, np.zeros((5,80))))
    gamma_QCD_5 = np.zeros((71,163))
    gamma_QCD_6 = np.hstack((np.zeros((9,154)), gamma_twist2))
    gamma_QCD = [np.vstack((gamma_QCD_1, gamma_QCD_2, gamma_QCD_3,\
                            gamma_QCD_4, gamma_QCD_5, gamma_QCD_6))]

    if nf == 5:
        return gamma_QCD
    elif nf == 4:
        return np.delete(np.delete(gamma_QCD, [6, 14, 22, 30, 42, 50, 58, 66, 74, 82, 94,\
                                               102, 110, 118, 126, 134, 142, 150, 158], 1)\
                                            , [6, 14, 22, 30, 42, 50, 58, 66, 74, 82, 94,\
                                               102, 110, 118, 126, 134, 142, 150, 158], 2)
    elif nf == 3:
        return np.delete(np.delete(gamma_QCD, [5,6, 13,14, 21,22, 29,30, 41,42,\
                                               49,50, 57,58, 65,66, 73,74, 81,82,\
                                               93,94, 101,102, 109,110, 117,118,\
                                               125,126, 133,134, 141,142, 149,150, 158,159], 1)\
                                            , [5,6, 13,14, 21,22, 29,30, 41,42,\
                                               49,50, 57,58, 65,66, 73,74, 81,82,\
                                               93,94, 101,102, 109,110, 117,118,\
                                               125,126, 133,134, 141,142, 149,150, 158,159], 2)
    else:
        raise Exception("nf has to be 3, 4 or 5")


def ADM_QCD2(nf):

    # CHECK ADM #

    """ Return the QCD anomalous dimension in the DM-SM sector for nf flavor EFT, when ADM starts at O(alphas^2) """
    # Mixing of Q_1^(7) into Q_{5,q}^(7) and Q_2^(7) into Q_{6,q}^(7), from Hill et al. [1409.8290].
    # Note that we have different prefactors and signs. 
    cf = 4/3
    gamma_gq = 8*cf # changed 2019-08-29, double check with RG solution
    # Mixing of Q_3^(7) into Q_{7,q}^(7) and Q_4^(7) into Q_{8,q}^(7), from Hill et al. [1409.8290].
    # Note that we have different prefactors and signs. 
    gamma_5gq = -8 # changed 2019-08-29, double check with RG solution
    gamma_QCD2_gq = np.array([5*[gamma_gq]])
    gamma_QCD2_5gq = np.array([5*[gamma_5gq]])
    gamma_QCD2_1 = np.zeros((34,163))
    gamma_QCD2_2 = np.hstack((np.zeros((1,38)),gamma_QCD2_gq,np.zeros((1,120))))
    gamma_QCD2_3 = np.hstack((np.zeros((1,46)),gamma_QCD2_gq,np.zeros((1,112))))
    gamma_QCD2_4 = np.hstack((np.zeros((1,54)),gamma_QCD2_5gq,np.zeros((1,104))))
    gamma_QCD2_5 = np.hstack((np.zeros((1,62)),gamma_QCD2_5gq,np.zeros((1,96))))
    gamma_QCD2_6 = np.zeros((125,163))
    gamma_QCD2 = [np.vstack((gamma_QCD2_1, gamma_QCD2_2, gamma_QCD2_3,\
                             gamma_QCD2_4, gamma_QCD2_5, gamma_QCD2_6))]

    if nf == 5:
        return gamma_QCD2
    elif nf == 4:
        return np.delete(np.delete(gamma_QCD2, [6, 14, 22, 30, 42, 50, 58, 66, 74, 82, 94,\
                                                102, 110, 118, 126, 134, 142, 150, 158], 1)\
                                             , [6, 14, 22, 30, 42, 50, 58, 66, 74, 82, 94,\
                                                102, 110, 118, 126, 134, 142, 150, 158], 2)
    elif nf == 3:
        return np.delete(np.delete(gamma_QCD2, [5,6, 13,14, 21,22, 29,30, 41,42,\
                                                49,50, 57,58, 65,66, 73,74, 81,82,\
                                                93,94, 101,102, 109,110, 117,118,\
                                                125,126, 133,134, 141,142, 149,150, 158,159], 1)\
                                             , [5,6, 13,14, 21,22, 29,30, 41,42,\
                                                49,50, 57,58, 65,66, 73,74, 81,82,\
                                                93,94, 101,102, 109,110, 117,118,\
                                                125,126, 133,134, 141,142, 149,150, 158,159], 2)
    else:
        raise Exception("nf has to be 3, 4 or 5")



def ADM5(Ychi, dchi):
    """ The dimension-five anomalous dimension
    
    Return a numpy array with the anomalous dimension matrices for g1, g2, g3, and yt 
    The Higgs self coupling lambda is currently ignored. 

    Variables
    ---------

    Ychi: The DM hypercharge, defined via the Gell-Mann - Nishijima relation Q = I_W^3 + Ychi/2. 

    dchi: The dimension of the electroweak SU(2) representation furnished by the DM multiplet. 
    """
    jj1 = (dchi**2-1)/4

    # The beta functions for one multiplet
    b1 = - 41/6 - Ychi**2 * dchi/3
    b2 = 19/6 - 4*jj1*dchi/9
    adm5_g1 = np.array([[5/2*Ychi**2-2*b1, 0, -6*Ychi, 0, 0, 0, 0, 0],
                        [-4*Ychi*jj1, Ychi**2/2, 0, 12*Ychi, 0, 0, 0, 0],
                        [0, 0, -3/2*(1+Ychi**2), 0, 0, 0, 0, 0],
                        [0, 0, 0, -3/2*(1+Ychi**2), 0, 0, 0, 0],
                        [0, 0, 0, 0, 5/2*Ychi**2-2*b1, 0, -6*Ychi, 0],
                        [0, 0, 0, 0, -4*Ychi*jj1, Ychi**2/2, 0, 12*Ychi],
                        [0, 0, 0, 0, 0, 0, -3/2*(1+Ychi**2), 0],
                        [0, 0, 0, 0, 0, 0, 0, -3/2*(1+Ychi**2)]])
    adm5_g2 = np.array([[2*jj1, -4*Ychi, 0, -24, 0, 0, 0, 0],
                        [0, (10*jj1-8)-2*b2, 12*jj1, 0, 0, 0, 0, 0],
                        [0, 0, (-9/2-6*jj1), 0, 0, 0, 0, 0],
                        [0, 0, 0, (3/2-6*jj1), 0, 0, 0, 0],
                        [0, 0, 0, 0, 2*jj1, -4*Ychi, 0, -24],
                        [0, 0, 0, 0, 0, (10*jj1-8)-2*b2, 12*jj1, 0],
                        [0, 0, 0, 0, 0, 0, (-9/2-6*jj1), 0],
                        [0, 0, 0, 0, 0, 0, 0, (3/2-6*jj1)]])
    adm5_g3   = np.zeros((8,8))
    adm5_yc   = np.diag([0,0,6,6,0,0,6,6])
    adm5_ytau = np.diag([0,0,2,2,0,0,2,2])
    adm5_yb   = np.diag([0,0,6,6,0,0,6,6])
    adm5_yt   = np.diag([0,0,6,6,0,0,6,6])
    adm5_lam  = np.diag([0,0,3,1,0,0,3,1])
    full_adm  = np.array([adm5_g1, adm5_g2, adm5_g3, adm5_yc, adm5_ytau, adm5_yb, adm5_yt, adm5_lam])
    if dchi == 1:
        return np.delete(np.delete(full_adm, [1,3,5,7], 1), [1,3,5,7], 2)
    else:
        return full_adm



def ADM6(Ychi, dchi):
    """ The dimension-five anomalous dimension
    
    Return a numpy array with the anomalous dimension matrices for g1, g2, g3, ytau, yb, and yt 
    The running due to the Higgs self coupling lambda is currently ignored. 

    The operator basis is Q1-Q14 1st, 2nd, 3rd gen.; S1-S17 (mixing of gen: 1-1, 2-2, 3-3, 1-2, 1-3, 2-3), 
                          S18-S24 1st, 2nd, 3rd gen., S25; D1-D4. 

    The explicit ordering of the operators, including flavor indices, is contained in the file 
    "directdm/run/operator_ordering.txt"

    Variables
    ---------

    Ychi: The DM hypercharge, defined via the Gell-Mann - Nishijima relation Q = I_W^3 + Ychi/2. 

    dchi: The dimension of the electroweak SU(2) representation furnished by the DM multiplet. 
    """

    scope = locals()

    def load_adm(admfile):
        with open(admfile, "r") as f:
            adm = []
            for line in f:
                line = re.sub("\n", "", line)
                line = line.split(",")
                adm.append(list(map(lambda x: eval(x, scope), line)))
            return adm

    admg1    = load_adm(resource_filename("directdm", "run/full_adm_g1.py"))
    admg2    = load_adm(resource_filename("directdm", "run/full_adm_g2.py"))
    admg3    = np.zeros((207,207))
    admyc    = load_adm(resource_filename("directdm", "run/full_adm_yc.py"))
    admytau  = load_adm(resource_filename("directdm", "run/full_adm_ytau.py"))
    admyb    = load_adm(resource_filename("directdm", "run/full_adm_yb.py"))
    admyt    = load_adm(resource_filename("directdm", "run/full_adm_yt.py"))
    admlam   = np.zeros((207,207))

    full_adm = np.array([np.array(admg1), np.array(admg2), admg3,\
                         np.array(admyc), np.array(admytau), np.array(admyb),\
                         np.array(admyt), np.array(admlam)])
    if dchi == 1:
        return np.delete(np.delete(full_adm, [0, 4, 8, 11, 14, 18, 22, 25, 28, 32, 36, 39,\
                                              42, 44, 205, 206], 1),\
                                             [0, 4, 8, 11, 14, 18, 22, 25, 28, 32, 36, 39,\
                                              42, 44, 205, 206], 2)
    else:
        return full_adm



def ADM_QCD_dim8(nf):
    """ Return the QCD anomalous dimension in the DM-SM sector at dim.8, for nf flavor EFT """

    beta0 = rge.QCD_beta(nf, 1).trad()
    gammam0 = rge.QCD_gamma(nf, 1).trad()

    ADM8 = 2*(gammam0 - beta0) * np.eye(12)

    return ADM8



def ADM_SM_QCD(nf):
    """ Return the QCD anomalous dimension in the SM-SM sector for nf flavor EFT, for a subset of SM dim.6 operators 

    The basis is spanned by a subset of 10*8 + 5*4 = 100 SM operators, with Wilson coefficients 

    ['P61ud', 'P62ud', 'P63ud', 'P63du', 'P64ud', 'P65ud', 'P66ud', 'P66du', 
     'P61us', 'P62us', 'P63us', 'P63su', 'P64us', 'P65us', 'P66us', 'P66su', 
     'P61uc', 'P62uc', 'P63uc', 'P63cu', 'P64uc', 'P65uc', 'P66uc', 'P66cu', 
     'P61ub', 'P62ub', 'P63ub', 'P63bu', 'P64ub', 'P65ub', 'P66ub', 'P66bu', 
     'P61ds', 'P62ds', 'P63ds', 'P63sd', 'P64ds', 'P65ds', 'P66ds', 'P66sd', 
     'P61dc', 'P62dc', 'P63dc', 'P63cd', 'P64dc', 'P65dc', 'P66dc', 'P66cd', 
     'P61db', 'P62db', 'P63db', 'P63bd', 'P64db', 'P65db', 'P66db', 'P66bd', 
     'P61sc', 'P62sc', 'P63sc', 'P63cs', 'P64sc', 'P65sc', 'P66sc', 'P66cs', 
     'P61sb', 'P62sb', 'P63sb', 'P63bs', 'P64sb', 'P65sb', 'P66sb', 'P66bs', 
     'P61cb', 'P62cb', 'P63cb', 'P63bc', 'P64cb', 'P65cb', 'P66cb', 'P66bc',
     'P61u', 'P62u', 'P63u', 'P64u', 
     'P61d', 'P62d', 'P63d', 'P64d', 
     'P61s', 'P62s', 'P63s', 'P64s', 
     'P61c', 'P62c', 'P63c', 'P64c', 
     'P61b', 'P62b', 'P63b', 'P64b']
    """

    adm_qqp_qqp = np.array([[0, 0, 0, 0, 0, 12, 0, 0],
                            [0, 0, 0, 0, 12, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 12],
                            [0, 0, 0, 0, 0, 0, 12, 0],
                            [0, 8/3, 0, 0, - 19/3, 5, 0, 0],
                            [8/3, 0, 0, 0, 5, - 9, 0, 0],
                            [0, 0, 0, 8/3, 0, 0, - 23/3, 5],
                            [0, 0, 8/3, 0, 0, 0, 5, - 23/3]])

    adm_qqp_qqpp = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 4/3, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 4/3, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0]])

    adm_qpq_qppq = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 4/3, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 4/3]])

    adm_qqp_qppq = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 4/3, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 4/3],
                             [0, 0, 0, 0, 0, 0, 0, 0]])

    adm_qpq_qqpp = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 4/3, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 4/3, 0]])

    adm_q_q = np.array([[4, 4, 0, - 28/3],
                        [0, 0, 0, 44/3],
                        [0, 0, 44/9, 0],
                        [5/3, 13/3, 0, - 106/9]])

    adm_qqp_q = np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 4/3],
                          [0, 0, 0, 0],
                          [0, 0, 4/9, 0],
                          [0, 0, 0, 0]])


    adm_qpq_q = np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 4/3],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 4/9, 0]])

    adm_q_qqp = np.array([[0, 0, 0, 0, 8/3, 0, 0, 0],
                          [0, 0, 0, 0, 8/3, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 8/3, 0],
                          [0, 0, 0, 0, 20/9, 0, 0, 0]])

    adm_q_qpq = np.array([[0, 0, 0, 0, 8/3, 0, 0, 0],
                          [0, 0, 0, 0, 8/3, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 8/3],
                          [0, 0, 0, 0, 20/9, 0, 0, 0]])

    adm_ud = np.hstack((adm_qqp_qqp, adm_qqp_qqpp, adm_qqp_qqpp,\
                        adm_qqp_qqpp, adm_qpq_qqpp, adm_qpq_qqpp,\
                        adm_qpq_qqpp, np.zeros((8, 24)), adm_qqp_q,\
                        adm_qpq_q, np.zeros((8,12))))

    adm_us = np.hstack((adm_qqp_qqpp, adm_qqp_qqp, adm_qqp_qqpp,\
                        adm_qqp_qqpp, adm_qpq_qppq, np.zeros((8,16)),\
                        adm_qpq_qqpp, adm_qpq_qqpp, np.zeros((8, 8)),\
                        adm_qqp_q, np.zeros((8,4)), adm_qpq_q, np.zeros((8,8))))

    adm_uc = np.hstack((adm_qqp_qqpp, adm_qqp_qqpp, adm_qqp_qqp,\
                        adm_qqp_qqpp, np.zeros((8,8)), adm_qpq_qppq,\
                        np.zeros((8,8)), adm_qpq_qppq, np.zeros((8, 8)),\
                        adm_qpq_qqpp, adm_qqp_q, np.zeros((8,8)),\
                        adm_qpq_q, np.zeros((8,4))))

    adm_ub = np.hstack((adm_qqp_qqpp, adm_qqp_qqpp, adm_qqp_qqpp,\
                        adm_qqp_qqp, np.zeros((8,16)), adm_qpq_qppq,\
                        np.zeros((8,8)), adm_qpq_qppq, adm_qpq_qppq,\
                        adm_qqp_q, np.zeros((8,12)), adm_qpq_q))

    adm_ds = np.hstack((adm_qqp_qppq, adm_qpq_qppq, np.zeros((8,16)),\
                        adm_qqp_qqp, adm_qqp_qqpp, adm_qqp_qqpp,\
                        adm_qpq_qqpp, adm_qpq_qqpp, np.zeros((8,8)),\
                        np.zeros((8,4)), adm_qqp_q, adm_qpq_q, np.zeros((8,8))))

    adm_dc = np.hstack((adm_qqp_qppq, np.zeros((8,8)), adm_qpq_qppq,\
                        np.zeros((8,8)), adm_qqp_qqpp, adm_qqp_qqp, adm_qqp_qqpp,\
                        adm_qpq_qppq, np.zeros((8,8)), adm_qpq_qqpp,\
                        np.zeros((8,4)), adm_qqp_q, np.zeros((8,4)),\
                        adm_qpq_q, np.zeros((8,4))))

    adm_db = np.hstack((adm_qqp_qppq, np.zeros((8,16)), adm_qpq_qppq,\
                        adm_qqp_qqpp, adm_qqp_qqpp, adm_qqp_qqp,\
                        np.zeros((8,8)), adm_qpq_qppq, adm_qpq_qppq,\
                        np.zeros((8,4)), adm_qqp_q, np.zeros((8,8)), adm_qpq_q))

    adm_sc = np.hstack((np.zeros((8,8)), adm_qqp_qppq, adm_qpq_qppq,\
                        np.zeros((8,8)), adm_qqp_qppq, adm_qpq_qppq, np.zeros((8,8)),\
                        adm_qqp_qqp, adm_qqp_qqpp, adm_qpq_qqpp, np.zeros((8,8)),\
                        adm_qqp_q, adm_qpq_q, np.zeros((8,4))))

    adm_sb = np.hstack((np.zeros((8,8)), adm_qqp_qppq, np.zeros((8,8)),\
                        adm_qpq_qppq, adm_qqp_qppq, np.zeros((8,8)), adm_qpq_qppq,\
                        adm_qqp_qqpp, adm_qqp_qqp, adm_qpq_qppq, np.zeros((8,8)),\
                        adm_qqp_q, np.zeros((8,4)), adm_qpq_q))

    adm_cb = np.hstack((np.zeros((8,16)), adm_qqp_qppq, adm_qpq_qppq,\
                        np.zeros((8,8)), adm_qqp_qppq, adm_qpq_qppq,\
                        adm_qqp_qppq, adm_qpq_qppq, adm_qqp_qqp,\
                        np.zeros((8,12)), adm_qqp_q, adm_qpq_q))

    adm_u = np.hstack((adm_q_qqp, adm_q_qqp, adm_q_qqp, adm_q_qqp,\
                       np.zeros((4,48)), adm_q_q, np.zeros((4,16))))

    adm_d = np.hstack((adm_q_qpq, np.zeros((4,24)), adm_q_qqp, adm_q_qqp,\
                       adm_q_qqp, np.zeros((4,24)), np.zeros((4,4)),\
                       adm_q_q, np.zeros((4,12))))

    adm_s = np.hstack((np.zeros((4,8)), adm_q_qpq, np.zeros((4,16)),\
                       adm_q_qpq, np.zeros((4,16)), adm_q_qqp,\
                       adm_q_qqp, np.zeros((4,8)),\
                       np.zeros((4,8)), adm_q_q, np.zeros((4,8))))

    adm_c = np.hstack((np.zeros((4,16)), adm_q_qpq, np.zeros((4,16)),\
                       adm_q_qpq, np.zeros((4,8)),\
                       adm_q_qpq, np.zeros((4,8)), adm_q_qqp,\
                       np.zeros((4,12)), adm_q_q, np.zeros((4,4))))

    adm_b = np.hstack((np.zeros((4,24)), adm_q_qpq, np.zeros((4,16)),\
                       adm_q_qpq, np.zeros((4,8)), adm_q_qpq,\
                       adm_q_qpq, np.zeros((4,16)), adm_q_q))


    adm = np.vstack((adm_ud, adm_us, adm_uc, adm_ub, adm_ds,\
                     adm_dc, adm_db, adm_sc, adm_sb, adm_cb,\
                     adm_u, adm_d, adm_s, adm_c, adm_b))

    if nf == 5:
        return adm
    elif nf == 4:
        return np.delete(np.delete(adm, np.r_[np.s_[24:32], np.s_[48:56],\
                                              np.s_[64:80], np.s_[96:100]], 0),\
                                        np.r_[np.s_[24:32], np.s_[48:56],\
                                              np.s_[64:80], np.s_[96:100]], 1)
    else:
        raise Exception("nf has to be 4 or 5")




def ADT_QCD(nf, input_dict=None):
    """ Return the QCD anomalous dimension tensor for nf flavor EFT,
        for double insertions of DM-SM and SM-SM operators 

    Our basis of operators below the electroweak scale includes a set of 12 dimension-eight operators, 
    with Wilson coefficients for Dirac DM
 
    ['C81u', 'C81d', 'C81s', 'C82u', 'C82d', 'C82s', 'C83u', 'C83d', 'C83s', 'C84u', 'C84d', 'C84s']

    and by a subset of 10*8 = 80 SM operators, with Wilson coefficients 

    ['P61ud', 'P62ud', 'P63ud', 'P63du', 'P64ud', 'P65ud', 'P66ud', 'P66du', 
     'P61us', 'P62us', 'P63us', 'P63su', 'P64us', 'P65us', 'P66us', 'P66su', 
     'P61uc', 'P62uc', 'P63uc', 'P63cu', 'P64uc', 'P65uc', 'P66uc', 'P66cu', 
     'P61ub', 'P62ub', 'P63ub', 'P63bu', 'P64ub', 'P65ub', 'P66ub', 'P66bu', 
     'P61ds', 'P62ds', 'P63ds', 'P63sd', 'P64ds', 'P65ds', 'P66ds', 'P66sd', 
     'P61dc', 'P62dc', 'P63dc', 'P63cd', 'P64dc', 'P65dc', 'P66dc', 'P66cd', 
     'P61db', 'P62db', 'P63db', 'P63bd', 'P64db', 'P65db', 'P66db', 'P66bd', 
     'P61sc', 'P62sc', 'P63sc', 'P63cs', 'P64sc', 'P65sc', 'P66sc', 'P66cs', 
     'P61sb', 'P62sb', 'P63sb', 'P63bs', 'P64sb', 'P65sb', 'P66sb', 'P66bs', 
     'P61cb', 'P62cb', 'P63cb', 'P63bc', 'P64cb', 'P65cb', 'P66cb', 'P66bc']

    The anomalous dimension tensor defined below uses the following subset of the dim.6 DM-SM basis,

    ['C63u', 'C63d', 'C63s', 'C63c', 'C63b', 'C64u', 'C64d', 'C64s', 'C64c', 'C64b']

    and the basis above.

    Arguments
    ---------

    nf -- the number of active flavors

    input_dict (optional) -- a dictionary of hadronic input parameters
                            (default is Num_input().input_parameters)
    """

    if input_dict is None:
        ip = Num_input().input_parameters
        # One should include a warning in case the dictionary
        # does not contain all necessary keys
    else:
        ip = input_dict

    mb = ip['mb_at_MZ']
    mc = ip['mc_at_MZ']
    ms = ip['ms_at_MZ']
    md = ip['md_at_MZ']
    mu = ip['mu_at_MZ']

    
    # Create the ADT:

    gamma_hat_P63cu_Q81u = np.hstack((np.zeros(3), -48 * mc**2/mu**2, np.zeros(6)))
    gamma_hat_P63bu_Q81u = np.hstack((np.zeros(4), -48 * mb**2/mu**2, np.zeros(5)))

    gamma_hat_P63cd_Q81d = np.hstack((np.zeros(3), -48 * mc**2/md**2, np.zeros(6)))
    gamma_hat_P63bd_Q81d = np.hstack((np.zeros(4), -48 * mb**2/md**2, np.zeros(5)))

    gamma_hat_P63cs_Q81s = np.hstack((np.zeros(3), -48 * mc**2/ms**2, np.zeros(6)))
    gamma_hat_P63bs_Q81s = np.hstack((np.zeros(4), -48 * mb**2/ms**2, np.zeros(5)))



    gamma_hat_P63cu_Q82u = np.hstack((np.zeros(8), -48 * mc**2/mu**2, np.zeros(1)))
    gamma_hat_P63bu_Q82u = np.hstack((np.zeros(9), -48 * mb**2/mu**2))

    gamma_hat_P63cd_Q82d = np.hstack((np.zeros(8), -48 * mc**2/md**2, np.zeros(1)))
    gamma_hat_P63bd_Q82d = np.hstack((np.zeros(9), -48 * mb**2/md**2))

    gamma_hat_P63cs_Q82s = np.hstack((np.zeros(8), -48 * mc**2/ms**2, np.zeros(1)))
    gamma_hat_P63bs_Q82s = np.hstack((np.zeros(9), -48 * mb**2/ms**2))



    gamma_hat_P62uc_Q83u = np.hstack((np.zeros(3), -48 * mc**2/mu**2, np.zeros(6)))
    gamma_hat_P62ub_Q83u = np.hstack((np.zeros(4), -48 * mb**2/mu**2, np.zeros(5)))

    gamma_hat_P62dc_Q83d = np.hstack((np.zeros(3), -48 * mc**2/md**2, np.zeros(6)))
    gamma_hat_P62db_Q83d = np.hstack((np.zeros(4), -48 * mb**2/md**2, np.zeros(5)))

    gamma_hat_P62sc_Q83s = np.hstack((np.zeros(3), -48 * mc**2/ms**2, np.zeros(6)))
    gamma_hat_P62sb_Q83s = np.hstack((np.zeros(4), -48 * mb**2/ms**2, np.zeros(5)))



    gamma_hat_P62uc_Q84u = np.hstack((np.zeros(8), -48 * mc**2/mu**2, np.zeros(1)))
    gamma_hat_P62ub_Q84u = np.hstack((np.zeros(9), -48 * mb**2/mu**2))

    gamma_hat_P62dc_Q84d = np.hstack((np.zeros(8), -48 * mc**2/md**2, np.zeros(1)))
    gamma_hat_P62db_Q84d = np.hstack((np.zeros(9), -48 * mb**2/md**2))

    gamma_hat_P62sc_Q84s = np.hstack((np.zeros(8), -48 * mc**2/ms**2, np.zeros(1)))
    gamma_hat_P62sb_Q84s = np.hstack((np.zeros(9), -48 * mb**2/ms**2))



    gamma_hat_Q81u = np.vstack((np.zeros((19,10)), gamma_hat_P63cu_Q81u,\
                                np.zeros((7,10)), gamma_hat_P63bu_Q81u, np.zeros((52,10))))
    gamma_hat_Q81d = np.vstack((np.zeros((43,10)), gamma_hat_P63cd_Q81d,\
                                np.zeros((7,10)), gamma_hat_P63bd_Q81d, np.zeros((28,10))))
    gamma_hat_Q81s = np.vstack((np.zeros((59,10)), gamma_hat_P63cs_Q81s,\
                                np.zeros((7,10)), gamma_hat_P63bs_Q81s, np.zeros((12,10))))

    gamma_hat_Q82u = np.vstack((np.zeros((19,10)), gamma_hat_P63cu_Q82u,\
                                np.zeros((7,10)), gamma_hat_P63bu_Q82u, np.zeros((52,10))))
    gamma_hat_Q82d = np.vstack((np.zeros((43,10)), gamma_hat_P63cd_Q82d,\
                                np.zeros((7,10)), gamma_hat_P63bd_Q82d, np.zeros((28,10))))
    gamma_hat_Q82s = np.vstack((np.zeros((59,10)), gamma_hat_P63cs_Q82s,\
                                np.zeros((7,10)), gamma_hat_P63bs_Q82s, np.zeros((12,10))))

    gamma_hat_Q83u = np.vstack((np.zeros((17,10)), gamma_hat_P62uc_Q83u,\
                                np.zeros((7,10)), gamma_hat_P62ub_Q83u, np.zeros((54,10))))
    gamma_hat_Q83d = np.vstack((np.zeros((41,10)), gamma_hat_P62dc_Q83d,\
                                np.zeros((7,10)), gamma_hat_P62db_Q83d, np.zeros((30,10))))
    gamma_hat_Q83s = np.vstack((np.zeros((57,10)), gamma_hat_P62sc_Q83s,\
                                np.zeros((7,10)), gamma_hat_P62sb_Q83s, np.zeros((14,10))))

    gamma_hat_Q84u = np.vstack((np.zeros((17,10)), gamma_hat_P62uc_Q84u,\
                                np.zeros((7,10)), gamma_hat_P62ub_Q84u, np.zeros((54,10))))
    gamma_hat_Q84d = np.vstack((np.zeros((41,10)), gamma_hat_P62dc_Q84d,\
                                np.zeros((7,10)), gamma_hat_P62db_Q84d, np.zeros((30,10))))
    gamma_hat_Q84s = np.vstack((np.zeros((57,10)), gamma_hat_P62sc_Q84s,\
                                np.zeros((7,10)), gamma_hat_P62sb_Q84s, np.zeros((14,10))))



    gamma_hat = np.array([gamma_hat_Q81u, gamma_hat_Q81d, gamma_hat_Q81s,\
                          gamma_hat_Q82u, gamma_hat_Q82d, gamma_hat_Q82s,\
                          gamma_hat_Q83u, gamma_hat_Q83d, gamma_hat_Q83s,\
                          gamma_hat_Q84u, gamma_hat_Q84d, gamma_hat_Q84s])


    # Return the tensor for given number of active quark flavors

    # tensor, zeile, spalte

    if nf == 5:
        return gamma_hat
    elif nf == 4:
        return np.delete(np.delete(gamma_hat, np.r_[np.s_[24:32], np.s_[48:56],\
                                                    np.s_[64:80]], 1), [4, 9], 2)
    else:
        raise Exception("nf has to be 4 or 5")


