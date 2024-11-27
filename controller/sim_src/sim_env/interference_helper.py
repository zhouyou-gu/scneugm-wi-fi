import math

import numpy as np
import scipy
class InterferenceHelper():
    C = 299792458.0
    PI = 3.14159265358979323846
    SIMPLE_CONSTANT_NOISE_FLOOR_DBM = -94.
    BOLTZMANN = 1.3803e-23
    NOISEFIGURE = 5    
    @staticmethod
    def bandwidth_hz_to_noise_dbm(B_hz, constant_noise = False):
        # we assume constant noise floor for simplicity.
        if not constant_noise:
            nf = InterferenceHelper.db_to_dec(InterferenceHelper.NOISEFIGURE);
            Nt = InterferenceHelper.BOLTZMANN * 290 * B_hz;
            noiseFloor = nf * Nt
            return InterferenceHelper.watts_to_dbm(noiseFloor)
        else:
            return InterferenceHelper.SIMPLE_CONSTANT_NOISE_FLOOR_DBM

    @staticmethod
    def fre_dis_to_loss_dB(fre_Hz, dis, model_name="log"):
        if model_name == "log":
            return InterferenceHelper.fre_dis_to_loss_dB_logdis_model(fre_Hz, dis)
        if model_name == "friis":
            return InterferenceHelper.fre_dis_to_loss_dB_friis_model(fre_Hz, dis)
        return 0.

    @staticmethod
    def fre_dis_to_loss_dB_logdis_model(fre_Hz, dis):
        """
        We use log-distance path loss model, assuming a factory or office scenario.
        @article{series2017propagation,
        title={Propagation data and prediction methods for the planning of short-range outdoor radiocommunication systems and radio local area networks in the frequency range 300 MHz to 100 GHz},
        author={Series, P},
        journal={ITU recommmendations},
        pages={1411--9},
        year={2017}}
        """
        L = 20. * math.log10(fre_Hz/1e6) + 16 - 28
        loss = L + 28 * math.log10(dis+1) # at least one-meter distance
        return loss
    
    @staticmethod
    def fre_dis_to_loss_dB_friis_model(fre_Hz, dis):
        """
        We use log-distance path loss model, assuming a factory or office scenario.
        @article{friis1946note,
        title={A note on a simple transmission formula},
        author={Friis, Harald T},
        journal={Proceedings of the IRE},
        volume={34},
        number={5},
        pages={254--256},
        year={1946},
        publisher={IEEE}}
        """
        m_lambda = InterferenceHelper.C / fre_Hz
        numerator = m_lambda * m_lambda
        denominator = 16 * InterferenceHelper.PI * InterferenceHelper.PI * dis * dis
        lossDb = -10 * math.log10 (numerator / denominator)
        return lossDb
    
    @staticmethod
    def db_to_dec(snr_db):
        return 10.**(snr_db/10.)

    @staticmethod
    def dec_to_db(snr_dec):
        return 10.* math.log10(snr_dec)

    @staticmethod
    def watts_to_dbm(w):
        return 10. * math.log10(w) + 30.

    @staticmethod
    def dbm_to_watts(p):
        return 10.**((p-30.)/10.)
    
    @staticmethod
    def polyanskiy_model(snr_dec, L, B, T):
        # we use polyanskiy model to compute the error rate
        nu = - L * math.log(2.) + B * T * math.log(1+snr_dec)
        do = math.sqrt(B * T * (1. - 1./((1.+snr_dec)**2)))
        return scipy.stats.norm.sf(nu/do)

    @staticmethod
    def err(x, L, B, T, max_err):
        snr = InterferenceHelper.db_to_dec(x)
        return InterferenceHelper.polyanskiy_model(snr, L, B, T)/max_err - 1.

    @staticmethod
    def bisection_method_min_snr_dec(L, B, T, max_err=1e-5, a=-5., b=30., tol=0.1):
        # find minimum snr for given channel setup and error rate.
        if InterferenceHelper.err(a, L, B, T, max_err) * InterferenceHelper.err(b, L, B, T, max_err) >= 0:
            print("Bisection method fails.")
            return None

        while (InterferenceHelper.err(a, L, B, T, max_err) - InterferenceHelper.err(b, L, B, T, max_err)) > tol:
            midpoint = (a + b) / 2
            if InterferenceHelper.err(midpoint, L, B, T, max_err) == 0:
                return midpoint
            elif InterferenceHelper.err(a, L, B, T, max_err) * InterferenceHelper.err(midpoint, L, B, T, max_err) < 0:
                b = midpoint
            else:
                a = midpoint

        return (a + b) / 2
    
    
if __name__ == "__main__":
    i = InterferenceHelper()
    print("noise power is ",InterferenceHelper.bandwidth_hz_to_noise_dbm(20e6))
    print("log_distance loss",InterferenceHelper.fre_dis_to_loss_dB_logdis_model(5.8e9,10*math.sqrt(2)))
    print("friis loss",InterferenceHelper.fre_dis_to_loss_dB_friis_model(5.8e9,1))
    print("error", InterferenceHelper.polyanskiy_model(InterferenceHelper.db_to_dec(-3),100*8,20e6,140e-6))