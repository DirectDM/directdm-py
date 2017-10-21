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
    gamma_QED_1 = np.zeros((2,154))
    gamma_QED_2 = np.hstack((np.zeros((8,2)),gamma_QED,np.zeros((8,144))))
    gamma_QED_3 = np.hstack((np.zeros((8,10)),gamma_QED,np.zeros((8,136))))
    gamma_QED_4 = np.zeros((136,154))
    gamma_QED = np.vstack((gamma_QED_1, gamma_QED_2, gamma_QED_3, gamma_QED_4))

    if nf == 5:
        return gamma_QED
    elif nf == 4:
        return np.delete(np.delete(gamma_QED, [6, 14, 22, 30, 42, 50, 58, 66, 74, 82, 94, 102, 110, 118, 126, 134, 142, 150], 0)\
                                            , [6, 14, 22, 30, 42, 50, 58, 66, 74, 82, 94, 102, 110, 118, 126, 134, 142, 150], 1)
    elif nf == 3:
        return np.delete(np.delete(gamma_QED, [5,6, 13,14, 21,22, 29,30, 41,42, 49,50, 57,58, 65,66, 73,74, 81,82,\
                                               93,94, 101,102, 109,110, 117,118, 125,126, 133,134, 141,142, 149,150], 0)\
                                            , [5,6, 13,14, 21,22, 29,30, 41,42, 49,50, 57,58, 65,66, 73,74, 81,82,\
                                               93,94, 101,102, 109,110, 117,118, 125,126, 133,134, 141,142, 149,150], 1)
    else:
        raise Exception("nf has to be 3, 4 or 5")


def ADM_QED2(nf):
    """ Returns the QED anomalous dimension for nf flavor EFT at alpha^2 """

    # Mixing of Q_{11}^(7) into Q_{5,f}^(7) and Q_{12}^(7) into Q_{6,f}^(7), adapted from Hill et al. [1409.8290]. 
    gamma_gf = -8
    gamma_QED2_gf = np.array([5*[gamma_gf]])
    gamma_QED2_1 = np.zeros((86,154))
    gamma_QED2_2 = np.hstack((np.zeros((1,38)),gamma_QED2_gf,np.zeros((1,111))))
    gamma_QED2_3 = np.hstack((np.zeros((1,46)),gamma_QED2_gf,np.zeros((1,103))))
    gamma_QED2_4 = np.zeros((66,154))
    gamma_QED2 = np.vstack((gamma_QED2_1, gamma_QED2_2, gamma_QED2_3, gamma_QED2_4))

    if nf == 5:
        return gamma_QED2
    elif nf == 4:
        return np.delete(np.delete(gamma_QED2, [6, 14, 22, 30, 42, 50, 58, 66, 74, 82, 94, 102, 110, 118, 126, 134, 142, 150], 0)\
                                             , [6, 14, 22, 30, 42, 50, 58, 66, 74, 82, 94, 102, 110, 118, 126, 134, 142, 150], 1)
    elif nf == 3:
        return np.delete(np.delete(gamma_QED2, [5,6, 13,14, 21,22, 29,30, 41,42, 49,50, 57,58, 65,66, 73,74, 81,82,\
                                                93,94, 101,102, 109,110, 117,118, 125,126, 133,134, 141,142, 149,150], 0)\
                                             , [5,6, 13,14, 21,22, 29,30, 41,42, 49,50, 57,58, 65,66, 73,74, 81,82,\
                                                93,94, 101,102, 109,110, 117,118, 125,126, 133,134, 141,142, 149,150], 1)
    else:
        raise Exception("nf has to be 3, 4 or 5")


#------------------------------#
# The QCD anomalous dimensions #
#------------------------------#

def ADM_QCD(nf):
    """ Returns the QCD anomalous dimension for nf flavor EFT, when ADM starts at O(alphas) """
    gamma_QCD_T = 32/3 * np.eye(5)
    gamma_QCD_1 = np.zeros((70,154))
    gamma_QCD_2 = np.hstack((np.zeros((5,70)),gamma_QCD_T,np.zeros((5,79))))
    gamma_QCD_3 = np.zeros((3,154))
    gamma_QCD_4 = np.hstack((np.zeros((5,78)),gamma_QCD_T,np.zeros((5,71))))
    gamma_QCD_5 = np.zeros((71,154))
    gamma_QCD = [np.vstack((gamma_QCD_1, gamma_QCD_2, gamma_QCD_3, gamma_QCD_4, gamma_QCD_5))]

    if nf == 5:
        return gamma_QCD
    elif nf == 4:
        return np.delete(np.delete(gamma_QCD, [6, 14, 22, 30, 42, 50, 58, 66, 74, 82, 94, 102, 110, 118, 126, 134, 142, 150], 1)\
                                            , [6, 14, 22, 30, 42, 50, 58, 66, 74, 82, 94, 102, 110, 118, 126, 134, 142, 150], 2)
    elif nf == 3:
        return np.delete(np.delete(gamma_QCD, [5,6, 13,14, 21,22, 29,30, 41,42, 49,50, 57,58, 65,66, 73,74, 81,82,\
                                               93,94, 101,102, 109,110, 117,118, 125,126, 133,134, 141,142, 149,150], 1)\
                                            , [5,6, 13,14, 21,22, 29,30, 41,42, 49,50, 57,58, 65,66, 73,74, 81,82,\
                                               93,94, 101,102, 109,110, 117,118, 125,126, 133,134, 141,142, 149,150], 2)
    else:
        raise Exception("nf has to be 3, 4 or 5")


def ADM_QCD2(nf):
    """ Returns the QCD anomalous dimension for nf flavor EFT, when ADM starts at O(alphas^2) """
    # Mixing of Q_1^(7) into Q_{5,q}^(7) and Q_2^(7) into Q_{6,q}^(7), from Hill et al. [1409.8290]. Note that we have different prefactors and signs. 
    gamma_gq = -32/3
    # Mixing of Q_3^(7) into Q_{7,q}^(7) and Q_4^(7) into Q_{8,q}^(7), from Hill et al. [1409.8290]. Note that we have different prefactors and signs. 
    gamma_5gq = 8
    gamma_QCD2_gq = np.array([5*[gamma_gq]])
    gamma_QCD2_5gq = np.array([5*[gamma_5gq]])
    gamma_QCD2_1 = np.zeros((34,154))
    gamma_QCD2_2 = np.hstack((np.zeros((1,38)),gamma_QCD2_gq,np.zeros((1,111))))
    gamma_QCD2_3 = np.hstack((np.zeros((1,46)),gamma_QCD2_gq,np.zeros((1,103))))
    gamma_QCD2_4 = np.hstack((np.zeros((1,54)),gamma_QCD2_5gq,np.zeros((1,95))))
    gamma_QCD2_5 = np.hstack((np.zeros((1,62)),gamma_QCD2_5gq,np.zeros((1,87))))
    gamma_QCD2_6 = np.zeros((116,154))
    gamma_QCD2 = [np.vstack((gamma_QCD2_1, gamma_QCD2_2, gamma_QCD2_3, gamma_QCD2_4, gamma_QCD2_5, gamma_QCD2_6))]

    if nf == 5:
        return gamma_QCD2
    elif nf == 4:
        return np.delete(np.delete(gamma_QCD2, [6, 14, 22, 30, 42, 50, 58, 66, 74, 82, 94, 102, 110, 118, 126, 134, 142, 150], 1)\
                                             , [6, 14, 22, 30, 42, 50, 58, 66, 74, 82, 94, 102, 110, 118, 126, 134, 142, 150], 2)
    elif nf == 3:
        return np.delete(np.delete(gamma_QCD2, [5,6, 13,14, 21,22, 29,30, 41,42, 49,50, 57,58, 65,66, 73,74, 81,82,\
                                                93,94, 101,102, 109,110, 117,118, 125,126, 133,134, 141,142, 149,150], 1)\
                                             , [5,6, 13,14, 21,22, 29,30, 41,42, 49,50, 57,58, 65,66, 73,74, 81,82,\
                                                93,94, 101,102, 109,110, 117,118, 125,126, 133,134, 141,142, 149,150], 2)
    else:
        raise Exception("nf has to be 3, 4 or 5")

