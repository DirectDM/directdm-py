#!/usr/bin/env python3

import numpy as np


#-----------------------------#
# The QED anomalous dimension #
#-----------------------------#


def ADM_QED(nf):
    """ Returns the QED anomalous dimension for nf flavor EFT """
    Qu = 2/3
    Qd = -1/3
    Qe = -1
    nc = 3
    gamma_QED = np.array([[8/3*Qu*Qu*nc, 8/3*Qu*Qd*nc, 8/3*Qu*Qd*nc, 8/3*Qu*Qu*nc, 8/3*Qu*Qd*nc, 8/3*Qu*Qe*nc, 8/3*Qu*Qe*nc, 8/3*Qu*Qe*nc],
                          [8/3*Qd*Qu*nc, 8/3*Qd*Qd*nc, 8/3*Qd*Qd*nc, 8/3*Qd*Qu*nc, 8/3*Qd*Qd*nc, 8/3*Qd*Qe*nc, 8/3*Qd*Qe*nc, 8/3*Qd*Qe*nc],
                          [8/3*Qd*Qu*nc, 8/3*Qd*Qd*nc, 8/3*Qd*Qd*nc, 8/3*Qd*Qu*nc, 8/3*Qd*Qd*nc, 8/3*Qd*Qe*nc, 8/3*Qd*Qe*nc, 8/3*Qd*Qe*nc],
                          [8/3*Qu*Qu*nc, 8/3*Qu*Qd*nc, 8/3*Qu*Qd*nc, 8/3*Qu*Qu*nc, 8/3*Qu*Qd*nc, 8/3*Qu*Qe*nc, 8/3*Qu*Qe*nc, 8/3*Qu*Qe*nc],
                          [8/3*Qd*Qu*nc, 8/3*Qd*Qd*nc, 8/3*Qd*Qd*nc, 8/3*Qd*Qu*nc, 8/3*Qd*Qd*nc, 8/3*Qd*Qe*nc, 8/3*Qd*Qe*nc, 8/3*Qd*Qe*nc],
                          [8/3*Qe*Qu,    8/3*Qe*Qd,    8/3*Qe*Qd,    8/3*Qe*Qu,    8/3*Qe*Qd,    8/3*Qe*Qe,    8/3*Qe*Qe,    8/3*Qe*Qe],
                          [8/3*Qe*Qu,    8/3*Qe*Qd,    8/3*Qe*Qd,    8/3*Qe*Qu,    8/3*Qe*Qd,    8/3*Qe*Qe,    8/3*Qe*Qe,    8/3*Qe*Qe],
                          [8/3*Qe*Qu,    8/3*Qe*Qd,    8/3*Qe*Qd,    8/3*Qe*Qu,    8/3*Qe*Qd,    8/3*Qe*Qe,    8/3*Qe*Qe,    8/3*Qe*Qe]])
    gamma_QED_1 = np.zeros((2,86))
    gamma_QED_2 = np.hstack((np.zeros((8,2)),gamma_QED,np.zeros((8,76))))
    gamma_QED_3 = np.hstack((np.zeros((8,10)),gamma_QED,np.zeros((8,68))))
    gamma_QED_4 = np.zeros((68,86))
    pre_gamma_QED = np.vstack((gamma_QED_1, gamma_QED_2, gamma_QED_3, gamma_QED_4))
    if nf == 5:
        return pre_gamma_QED
    elif nf == 4:
        return np.delete(np.delete(pre_gamma_QED, [6, 14, 22, 30, 42, 50, 58, 66, 74, 82], 0), [6, 14, 22, 30, 42, 50, 58, 66, 74, 82], 1)
    elif nf == 3:
        return np.delete(np.delete(pre_gamma_QED, [5,6, 13,14, 21,22, 29,30, 41,42, 49,50, 57,58, 65,66, 73,74, 81,82], 0)\
                                                , [5,6, 13,14, 21,22, 29,30, 41,42, 49,50, 57,58, 65,66, 73,74, 81,82], 1)
    else:
        raise Exception("nf has to be 3, 4 or 5")


#------------------------------#
# The QCD anomalous dimensions #
#------------------------------#

def ADM_QCD(nf):
    """ Returns the QCD anomalous dimension for nf flavor EFT, when ADM starts at O(alphas) """
    gamma_QCD_T = 32/3 * np.eye(5)
    gamma_QCD_1 = np.zeros((70,86))
    gamma_QCD_2 = np.hstack((np.zeros((5,70)),gamma_QCD_T,np.zeros((5,11))))
    gamma_QCD_3 = np.zeros((3,86))
    gamma_QCD_4 = np.hstack((np.zeros((5,78)),gamma_QCD_T,np.zeros((5,3))))
    gamma_QCD_5 = np.zeros((3,86))
    gamma_QCD = [np.vstack((gamma_QCD_1, gamma_QCD_2, gamma_QCD_3, gamma_QCD_4, gamma_QCD_5))]
    if nf == 5:
        return gamma_QCD
    elif nf == 4:
        return np.delete(np.delete(gamma_QCD, [6, 14, 22, 30, 42, 50, 58, 66, 74, 82], 1), [6, 14, 22, 30, 42, 50, 58, 66, 74, 82], 2)
    elif nf == 3:
        return np.delete(np.delete(gamma_QCD, [5,6, 13,14, 21,22, 29,30, 41,42, 49,50, 57,58, 65,66, 73,74, 81,82], 1)\
                                            , [5,6, 13,14, 21,22, 29,30, 41,42, 49,50, 57,58, 65,66, 73,74, 81,82], 2)
    else:
        raise Exception("nf has to be 3, 4 or 5")


def ADM_QCD2(nf):
    """ Returns the QCD anomalous dimension for nf flavor EFT, when ADM starts at O(alphas^2) """
    # Mixing of Q_1^(7) into Q_{5,q}^(7) and Q_2^(7) into Q_{6,q}^(7), from Hill et al. 1409.8290. Note that we have different prefactors and signs. 
    gamma_gq = -32/3
    # Mixing of Q_3^(7) into Q_{7,q}^(7) and Q_4^(7) into Q_{8,q}^(7), from Hill et al. 1409.8290. Note that we have different prefactors and signs. 
    gamma_5gq = 8
    gamma_QCD2_gq = np.array([5*[gamma_gq]])
    gamma_QCD2_5gq = np.array([5*[gamma_5gq]])
    gamma_QCD2_1 = np.zeros((34,86))
    gamma_QCD2_2 = np.hstack((np.zeros((1,38)),gamma_QCD2_gq,np.zeros((1,43))))
    gamma_QCD2_3 = np.hstack((np.zeros((1,46)),gamma_QCD2_gq,np.zeros((1,35))))
    gamma_QCD2_4 = np.hstack((np.zeros((1,54)),gamma_QCD2_5gq,np.zeros((1,27))))
    gamma_QCD2_5 = np.hstack((np.zeros((1,62)),gamma_QCD2_5gq,np.zeros((1,19))))
    gamma_QCD2_6 = np.zeros((48,86))
    gamma_QCD2 = [np.vstack((gamma_QCD2_1, gamma_QCD2_2, gamma_QCD2_3, gamma_QCD2_4, gamma_QCD2_5, gamma_QCD2_6))]
    if nf == 5:
        return gamma_QCD2
    elif nf == 4:
        return np.delete(np.delete(gamma_QCD2, [6, 14, 22, 30, 42, 50, 58, 66, 74, 82], 1), [6, 14, 22, 30, 42, 50, 58, 66, 74, 82], 2)
    elif nf == 3:
        return np.delete(np.delete(gamma_QCD2, [5,6, 13,14, 21,22, 29,30, 41,42, 49,50, 57,58, 65,66, 73,74, 81,82], 1)\
                                             , [5,6, 13,14, 21,22, 29,30, 41,42, 49,50, 57,58, 65,66, 73,74, 81,82], 2)
    else:
        raise Exception("nf has to be 3, 4 or 5")

