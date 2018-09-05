#!/usr/bin/env python3

import sys
import numpy as np


class Num_input(object):
    def __init__(self):
    # numerical input. All masses in GeV. Updated from PDG 2018.

        # couplings etc.
        self.asMZ = 0.1181
        self.dasMZ = 0.0011
        self.GF = 1.1663787*10**(-5)
        self.dGF = 0.0000006*10**(-5)
        self.aMZinv = 127.955
        self.daMZinv = 0.01
        self.amtauinv = 133.476
        self.damtauinv = 0.007
        self.alowinv = 137.035999139
        self.dalowinv = 0.000000031
        self.sw2_MSbar = 0.23122
        self.dsw2_MSbar = 0.00004

        # Boson masses
        self.Mz = 91.1876
        self.dMz = 0.0021
        self.Mh = 125.18
        self.dMh = 0.16
        self.Mw = 80.379
        self.dMw = 0.012

        # Lepton masses
        self.mtau = 1.77686
        self.mmu = 105.6583715e-3
        self.me = 0.000510998928

        # Baryon masses
        self.mproton = 938.272081e-3
        self.dmproton = 0.000006e-3
        self.mneutron = 939.565413e-3
        self.dmneutron = 0.000006e-3
        self.mN = (self.mproton+self.mneutron)/2

        # Meson masses
        self.mpi0 = 134.98e-3
        self.dmpi0 = 0
        self.meta = 547.862e-3
        self.dmeta = 0.017e-3

        # PDG
        self.mt_at_mt_QCD = 160. #GeV
        self.dmt_at_mt_QCD = 5 #GeV
        self.mb_at_mb = 4.18 #GeV
        self.dmb_at_mb = 0.04 #GeV
        self.mt_pole = 173.0 #GeV
        self.dmt_pole = 0.4 #GeV
        self.mc_at_mc = 1.275 #GeV
        self.dmc_at_mc = 0.03 #GeV

        # Light-quark masses from PDG. MSbar scheme at 2 GeV
        self.ms_at_2GeV = 0.096 #GeV
        self.md_at_2GeV = 0.0047 #GeV
        self.mu_at_2GeV = 0.0022 #GeV

        # mc(2 GeV) (1 loop):
        self.mc_at_2GeV = 1.18 # GeV

        # mc(3 GeV) [0907.2110]:
        self.mc_at_3GeV = 0.986 # GeV
        self.dmc_at_3GeV = 0.013 # GeV

        # Quark masses at MZ (at 1-loop )
        self.mu_at_MZ = 1.4480486828689913e-3
        self.md_at_MZ = 3.093558685557306e-3
        self.ms_at_MZ = 63.187583673361705e-3
        self.mc_at_MZ = 0.775
        self.mb_at_MZ = 3.08
        self.mt_at_MZ = 182 # This is converted to MSbar QCD and EW, and run to MZ at 1-loop QCD


        # MSbar input at mu=MZ:
        self.g2_at_MZ   = np.sqrt(4*np.pi/self.aMZinv/self.sw2_MSbar)
        self.g1_at_MZ   = np.sqrt(self.g2_at_MZ**2/(1/self.sw2_MSbar - 1))
        self.g3_at_MZ   = np.sqrt(4*np.pi*self.asMZ)
        self.yc_at_MZ   = np.sqrt(np.sqrt(2)*self.GF)*np.sqrt(2) * self.mc_at_MZ
        self.yb_at_MZ   = np.sqrt(np.sqrt(2)*self.GF)*np.sqrt(2) * self.mb_at_MZ
        self.ytau_at_MZ = np.sqrt(np.sqrt(2)*self.GF)*np.sqrt(2) * self.mtau
        self.yt_at_MZ   = np.sqrt(np.sqrt(2)*self.GF)*np.sqrt(2) * self.mt_at_MZ
        self.lam_at_MZ  = 2*np.sqrt(2) * self.GF * self.Mh**2


        # Further low-energy input for pionless EFT
        self.gA = 1.2723
        self.dgA = 0.0023
        self.mG = 0.848
        self.dmG = 0.014

        self.sigmaup = 17e-3
        self.sigmadp = 32e-3
        self.sigmaun = 15e-3
        self.sigmadn = 36e-3
        self.sigmas = 41.3e-3

        self.dsigmaup = 5e-3
        self.dsigmadp = 10e-3
        self.dsigmaun = 5e-3
        self.dsigmadn = 10e-3
        self.dsigmas = 7.7e-3

        self.Deltaup = 0.897
        self.Deltadp = -0.376
        self.Deltas = -0.031
        self.Deltaun = self.Deltadp
        self.Deltadn = self.Deltaup

        self.dDeltaup = 0.027
        self.dDeltadp = 0.027
        self.dDeltas = 0.005

        self.B0mu = 6.1e-3
        self.dB0mu = 0.5e-3
        self.B0md = 13.3e-3
        self.dB0md = 0.5e-3
        self.B0ms = 0.268
        self.dB0ms = 0.003

        # nuclear dipole moments
        self.mup = 2.793
        self.mun = -1.913

        self.muup = 1.8045
        self.mudp = -1.097
        self.mudn = self.muup
        self.muun = self.mudp
        self.mus = -0.064

        self.ap = 1.793
        self.an = -1.913
        self.F2sp = -0.064

        # nuclear tensor charges (at 2 GeV)
        self.gTu = 0.794
        self.gTd = -0.204
        self.gTs = 3.2e-4

        self.dgTu = 0.015
        self.dgTd = 0.008
        self.dgTs = 8.6e-4

        self.BT10up = 3.0
        self.BT10dp = 0.24
        self.BT10un = self.BT10dp
        self.BT10dn = self.BT10up
        self.BT10s = 0

        self.dBT10up = self.BT10up/2
        self.dBT10dp = self.BT10dp/2
        self.dBT10s = 0.2

        # dependent variables
        self.mtilde = 1/(1/self.mu_at_2GeV + 1/self.md_at_2GeV + 1/self.ms_at_2GeV)


        # "Astrophysical" input

        # velocity of light [km/s]:
        self.clight = 299792.458 

        # DM energy density [GeV / cm^3]:
        self.rho0 = 0.47 # PDG 2016
        # mean DM velocity [km/s]:
        self.vmean = 240 # PDG 2016
        # galactic escape velocity [km/s]:
        self.vescape = 544
        # velocity of earth [km/s]:
        self.vearth = 244


