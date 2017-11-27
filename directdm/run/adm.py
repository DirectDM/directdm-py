#!/usr/bin/env python3

import numpy as np
from ..num.num_input import Num_input


#-----------------------------#
# The QED anomalous dimension #
#-----------------------------#


def ADM_QED(nf):
    """ Return the QED anomalous dimension for nf flavor EFT """
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
    """ Return the QED anomalous dimension for nf flavor EFT at alpha^2 """

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
    """ Return the QCD anomalous dimension for nf flavor EFT, when ADM starts at O(alphas) """
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
    """ Return the QCD anomalous dimension for nf flavor EFT, when ADM starts at O(alphas^2) """
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



def ADM_SM_QCD(nf):
    """ Return the QCD anomalous dimension for nf flavor EFT, for a subset of SM dim.6 operators 

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
                            [0, 8/3, 0, 0, - 9 + 4/3, 5 + 4/3, 0, 0],
                            [8/3, 0, 0, 0,5, - 9, 0, 0],
                            [0, 0, 0, 8/3, 0, 0, - 9 + 4/3, 5],
                            [0, 0, 16/9, 0, 0, 0, 5, - 9 + 4/3]])

    adm_qqp_qqpp = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 4/3, 4/3, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 4/3, 0],
                             [0, 0, 0, 0, 0, 0, 0, 4/3]])

    adm_q_q = np.array([[4, 4, 0, - 12 + 8/3],
                        [0, 0, 0, 12 + 8/3],
                        [0, 0, 4 + 8/9, 0],
                        [5/3, 13/3, 0, - 14 - 4/9 + 8/3]])

    adm_qqp_q = np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 4/3],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 4/9, 0]])

    adm_q_qqp = np.array([[0, 0, 0, 0, 0, 8/3, 0, 0],
                          [0, 0, 0, 0, 0, 8/3, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 8/3],
                          [0, 0, 0, 0, 0, - 4/9 + 8/3, 0, 0]])

    adm_ud = np.hstack((adm_qqp_qqp, adm_qqp_qqpp, adm_qqp_qqpp, adm_qqp_qqpp, adm_qqp_qqpp, adm_qqp_qqpp,\
                        adm_qqp_qqpp, np.zeros((8, 24)), adm_qqp_q, adm_qqp_q, np.zeros((8,12))))

    adm_us = np.hstack((adm_qqp_qqpp, adm_qqp_qqp, adm_qqp_qqpp, adm_qqp_qqpp, adm_qqp_qqpp, np.zeros((8,16)),\
                        adm_qqp_qqpp, adm_qqp_qqpp, np.zeros((8, 8)), adm_qqp_q, np.zeros((8,4)), adm_qqp_q, np.zeros((8,8))))

    adm_uc = np.hstack((adm_qqp_qqpp, adm_qqp_qqpp, adm_qqp_qqp, adm_qqp_qqpp, np.zeros((8,8)), adm_qqp_qqpp,\
                        np.zeros((8,8)), adm_qqp_qqpp, np.zeros((8, 8)), adm_qqp_qqpp, adm_qqp_q, np.zeros((8,8)), adm_qqp_q, np.zeros((8,4))))

    adm_ub = np.hstack((adm_qqp_qqpp, adm_qqp_qqpp, adm_qqp_qqpp, adm_qqp_qqp, np.zeros((8,16)), adm_qqp_qqpp,\
                        np.zeros((8,8)), adm_qqp_qqpp, adm_qqp_qqpp, adm_qqp_q, np.zeros((8,12)), adm_qqp_q))

    adm_ds = np.hstack((adm_qqp_qqpp, adm_qqp_qqpp, np.zeros((8,16)), adm_qqp_qqp, adm_qqp_qqpp, adm_qqp_qqpp,\
                        adm_qqp_qqpp, adm_qqp_qqpp, np.zeros((8,8)), np.zeros((8,4)), adm_qqp_q, adm_qqp_q, np.zeros((8,8))))

    adm_dc = np.hstack((adm_qqp_qqpp, np.zeros((8,8)), adm_qqp_qqpp, np.zeros((8,8)), adm_qqp_qqpp, adm_qqp_qqp, adm_qqp_qqpp,\
                        adm_qqp_qqpp, np.zeros((8,8)), adm_qqp_qqpp, np.zeros((8,4)), adm_qqp_q, np.zeros((8,4)), adm_qqp_q, np.zeros((8,4))))

    adm_db = np.hstack((adm_qqp_qqpp, np.zeros((8,16)), adm_qqp_qqpp, adm_qqp_qqpp, adm_qqp_qqpp, adm_qqp_qqp,\
                        np.zeros((8,8)), adm_qqp_qqpp, adm_qqp_qqpp, np.zeros((8,4)), adm_qqp_q, np.zeros((8,8)), adm_qqp_q))

    adm_sc = np.hstack((np.zeros((8,8)), adm_qqp_qqpp, adm_qqp_qqpp, np.zeros((8,8)), adm_qqp_qqpp, adm_qqp_qqpp, np.zeros((8,8)),\
                        adm_qqp_qqp, adm_qqp_qqpp, adm_qqp_qqpp, np.zeros((8,8)), adm_qqp_q, adm_qqp_q, np.zeros((8,4))))

    adm_sb = np.hstack((np.zeros((8,8)), adm_qqp_qqpp, np.zeros((8,8)), adm_qqp_qqpp, adm_qqp_qqpp, np.zeros((8,8)), adm_qqp_qqpp,\
                        adm_qqp_qqpp, adm_qqp_qqp, adm_qqp_qqpp, np.zeros((8,8)), adm_qqp_q, np.zeros((8,4)), adm_qqp_q))

    adm_cb = np.hstack((np.zeros((8,16)), adm_qqp_qqpp, adm_qqp_qqpp, np.zeros((8,8)), adm_qqp_qqpp, adm_qqp_qqpp,\
                        adm_qqp_qqpp, adm_qqp_qqpp, adm_qqp_qqp, np.zeros((8,12)), adm_qqp_q, adm_qqp_q))

    adm_u = np.hstack((adm_q_qqp, adm_q_qqp, adm_q_qqp, adm_q_qqp, np.zeros((4,48)), adm_q_q, np.zeros((4,16))))

    adm_d = np.hstack((adm_q_qqp, np.zeros((4,24)), adm_q_qqp, adm_q_qqp, adm_q_qqp, np.zeros((4,24)), np.zeros((4,4)), adm_q_q, np.zeros((4,12))))

    adm_s = np.hstack((np.zeros((4,8)), adm_q_qqp, np.zeros((4,16)), adm_q_qqp, np.zeros((4,16)), adm_q_qqp, adm_q_qqp, np.zeros((4,8)),\
                       np.zeros((4,8)), adm_q_q, np.zeros((4,8))))

    adm_c = np.hstack((np.zeros((4,16)), adm_q_qqp, np.zeros((4,16)), adm_q_qqp, np.zeros((4,8)), adm_q_qqp, np.zeros((4,8)), adm_q_qqp,\
                       np.zeros((4,12)), adm_q_q, np.zeros((4,4))))

    adm_b = np.hstack((np.zeros((4,24)), adm_q_qqp, np.zeros((4,16)), adm_q_qqp, np.zeros((4,8)), adm_q_qqp, adm_q_qqp, np.zeros((4,16)), adm_q_q))


    adm = np.vstack((adm_ud, adm_us, adm_uc, adm_ub, adm_ds, adm_dc, adm_db, adm_sc, adm_sb, adm_cb, adm_u, adm_d, adm_s, adm_c, adm_b))

    if nf == 5:
        return adm
    elif nf == 4:
        return np.delete(np.delete(adm, np.r_[np.s_[24:32], np.s_[48:56], np.s_[64:80], np.s_[96:100]], 1),\
                                        np.r_[np.s_[24:32], np.s_[48:56], np.s_[64:80], np.s_[96:100]], 2)
    else:
        raise Exception("nf has to be 4 or 5")




def ADT_QCD(nf):
    """ Return the QCD anomalous dimension tensor for nf flavor EFT, for double insertions of DMDM and DMSM operators 

    Just for including this effect, we extend our basis of operators below the electroweak scale by a set of 6 dimension-eight operators, 
    with Wilson coefficients for Dirac DM
 
    ['C83u', 'C83d', 'C83s', 'C84u', 'C84d', 'C84s']

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
    """

    # As input for the quark-mass ratios, we use the quark masses at MZ
    mu = 1.4e-3
    md = 3.1e-3
    ms = 63e-3
    mc = 0.78
    mb = 3.1

    # Create the ADT:

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



    gamma_hat_Q83u = np.vstack((np.zeros((17,10)), gamma_hat_P62uc_Q83u, np.zeros((7,10)), gamma_hat_P62ub_Q83u, np.zeros((54,10))))
    gamma_hat_Q83d = np.vstack((np.zeros((41,10)), gamma_hat_P62dc_Q83d, np.zeros((7,10)), gamma_hat_P62db_Q83d, np.zeros((30,10))))
    gamma_hat_Q83s = np.vstack((np.zeros((57,10)), gamma_hat_P62sc_Q83s, np.zeros((7,10)), gamma_hat_P62sb_Q83s, np.zeros((14,10))))

    gamma_hat_Q84u = np.vstack((np.zeros((17,10)), gamma_hat_P62uc_Q84u, np.zeros((7,10)), gamma_hat_P62ub_Q84u, np.zeros((54,10))))
    gamma_hat_Q84d = np.vstack((np.zeros((41,10)), gamma_hat_P62dc_Q84d, np.zeros((7,10)), gamma_hat_P62db_Q84d, np.zeros((30,10))))
    gamma_hat_Q84s = np.vstack((np.zeros((57,10)), gamma_hat_P62sc_Q84s, np.zeros((7,10)), gamma_hat_P62sb_Q84s, np.zeros((14,10))))



    gamma_hat = np.array([gamma_hat_Q83u, gamma_hat_Q83d, gamma_hat_Q83s, gamma_hat_Q84u, gamma_hat_Q84d, gamma_hat_Q84s])


    # Return the tensor for given number of active quark flavors

    # tensor, zeile, spalte

    if nf == 5:
        return gamma_hat
    elif nf == 4:
        return np.delete(np.delete(gamma_hat, np.r_[np.s_[24:32], np.s_[48:56], np.s_[64:80]], 1), [4, 9], 2)
    else:
        raise Exception("nf has to be 4 or 5")





