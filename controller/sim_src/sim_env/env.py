import math

import numpy as np
import scipy

from sim_src.sim_env.interference_helper import InterferenceHelper

class WiFiNet(InterferenceHelper):
    
    """
    this class implements an ap-sta network to setup wi-fi network simulation.
    
    """
    STA_AP_LOSS_THREHOLD = 100
    def __init__(self, cell_edge = 20., cell_size = 5, sta_density_per_1m2 = 10e-3, fre_Hz = 5.8e9, txp_dbm_hi = 5., min_s_n_ratio = 0.1, packet_bit = 800, bandwidth = 20e6, max_err = 1e-5, seed=1):
        """
        Initializes the simulation environment with the given parameters.

        Parameters:
        -----------
        cell_edge : float
            The edge length of a single cell in meters.
        cell_size : int
            The number of cells along one dimension of the grid.
        sta_density_per_1m2 : float
            The density of stations (STAs) per square meter.
        fre_Hz : float
            The frequency of the signal in Hertz.
        txp_dbm_hi : float
            The transmission power of the access points (APs) in dBm.
        txp_offset : float
            The offset to be applied to the transmission power.
        min_s_n_ratio : float
            The minimum signal-to-noise ratio required.
        packet_bit : int
            The size of each packet in bits.
        bandwidth : float
            The bandwidth of the signal in Hertz.
        max_err : float
            The maximum allowable error rate.
        seed : int
            Seed for random number generation.
        """
        
        self.rand_gen_loc = np.random.default_rng(seed)
        self.rand_gen_fad = np.random.default_rng(seed)
        self.rand_gen_mob = np.random.default_rng(seed)

        self.cell_edge = cell_edge
        self.cell_size = cell_size

        self.grid_edge = self.cell_edge * self.cell_size

        self.n_ap = int(self.cell_size ** 2)
        self.ap_offset = self.cell_edge / 2.

        self.sta_density_per_1m2 = sta_density_per_1m2
        self.sta_density_per_grid = self.sta_density_per_1m2 * self.cell_edge ** 2
        self.n_sta = int(self.cell_size**2 * self.sta_density_per_grid)

        self.fre_Hz = fre_Hz
        self.txp_dbm_hi = txp_dbm_hi
        self.min_s_n_ratio = min_s_n_ratio
        self.packet_bit = packet_bit
        self.bandwidth = bandwidth
        self.max_err = max_err

        self.ap_locs = None
        self.sta_locs = None
        self.sta_dirs = None

        self.min_sinr = None
        self.loss = None

        self._config_ap_locs()
        self._config_sta_locs()
        self._config_sta_dirs()

    def get_loss_ap_ap(self):
        ret = np.zeros((self.n_ap,self.n_ap))
        for i in range(self.n_ap):
            for j in range(self.n_ap):
                if i == j:
                    continue
                ret[i,j] = self._get_loss_between_locs(self.ap_locs[i],self.ap_locs[j])
        return ret

    def get_loss_sta_ap(self):
        ret = np.ones((self.n_sta,self.n_ap))*np.inf
        for i in range(self.n_sta):
            for j in range(self.n_ap):
                ret[i,j] = self._get_loss_between_locs(self.sta_locs[i],self.ap_locs[j])
            #TODO: double check the losses of each STA so each STA has at least one AP can detect its signal.
        return ret

    def get_loss_sta_sta(self):
        ret = np.zeros((self.n_sta,self.n_sta))
        for i in range(self.n_sta):
            for j in range(i,self.n_sta):
                if i == j:
                    continue
                ret[i,j] = self._get_loss_between_locs(self.sta_locs[i],self.sta_locs[j])
                ret[j,i] = ret[i,j]
        return ret

    def _get_loss_between_locs(self, a, b):
        dis = np.linalg.norm(np.array(a)-np.array(b),ord=2)
        return self._get_loss_distance(dis)

    def _get_loss_distance(self, dis):
        #shadowing is disabled
        return InterferenceHelper.fre_dis_to_loss_dB(self.fre_Hz,dis)

    def get_sta_states(self):
        state_list = []
        for k in range(self.n_sta):
            tmp_list = []
            for a in range(self.n_ap):
                if self._get_loss_between_locs(self.sta_locs[k],self.ap_locs[a]) <= self.get_loss_sta_ap_threhold():
                    t = (a,self.ap_locs[a][0],self.ap_locs[a][1],self._get_loss_between_locs(self.sta_locs[k],self.ap_locs[a]))
                    tmp_list.append(t)
            sorted(tmp_list, key=lambda x: x[3])
            state_list.append(tmp_list)        
        return state_list
    
    def get_interfering_node_matrix(self):
        ret = np.zeros((self.n_sta,self.n_sta))
        loss_sta_ap = self.get_loss_sta_ap()
        asso = np.argmin(loss_sta_ap,axis=1)
        S_gain = loss_sta_ap[:, asso]
        ret[S_gain<=self.get_loss_sta_ap_threhold()] = 1   
        np.fill_diagonal(ret,0)
        return ret
    
    def get_contending_node_matrix(self):
        ret = np.zeros((self.n_sta,self.n_sta))
        loss_sta_sta = self.get_loss_sta_sta()
        ret[loss_sta_sta<=self.get_loss_sta_sta_threhold()] = 1   
        np.fill_diagonal(ret,0)
        return ret
    
    def get_hidden_node_matrix(self):
        I = self.get_interfering_node_matrix()
        C = self.get_contending_node_matrix()
        ret = np.logical_and(I.astype(bool) , np.logical_not(C.astype(bool))).astype(float)
        np.fill_diagonal(ret,0)
        return ret
    
    def get_loss_sta_ap_threhold(self):
        #TODO: change it according to the rx sensitivity
        return self.STA_AP_LOSS_THREHOLD
    
    def get_loss_sta_sta_threhold(self):
        #TODO: change it according to the rx sensitivity
        return self.STA_AP_LOSS_THREHOLD
    
    def convert_loss_sta_ap_threshold(self, loss):
        ret = np.copy(loss)
        ret[ret>self.get_loss_sta_ap_threhold()] = self.HIDDEN_LOSS
        return ret

    def _config_ap_locs(self):
        x=np.linspace(0 + self.ap_offset, self.grid_edge - self.ap_offset, self.cell_size)
        y=np.linspace(0 + self.ap_offset, self.grid_edge - self.ap_offset, self.cell_size)
        xx,yy=np.meshgrid(x,y)
        self.ap_locs = np.array((xx.ravel(), yy.ravel())).T

    def _config_sta_locs(self):
        self.sta_locs = self.rand_gen_loc.uniform(low=0.,high=self.grid_edge,size=(self.n_sta,2))

    def _config_sta_dirs(self):
        dd = self.rand_gen_mob.standard_normal(size=(self.n_sta,2))
        self.sta_dirs = dd/np.linalg.norm(dd,axis=1,keepdims=True)

    def _get_random_dir(self):
        dd = self.rand_gen_mob.standard_normal(2)
        return dd/np.linalg.norm(dd)

    def rand_user_mobility(self, mobility_in_meter_s = 0., t_us = 0, resolution_us = 1.):

        if mobility_in_meter_s == 0. or t_us == 0.:
            return
        n_step = math.ceil(t_us/resolution_us)
        for n in range(n_step):
            for i in range(self.n_sta):
                dd = self.sta_dirs[i] * mobility_in_meter_s * resolution_us/1e6
                x = self.sta_locs[i][0] + dd[0]
                y = self.sta_locs[i][1] + dd[1]
                if 0 <= x <= self.grid_edge and 0 <= y <= self.grid_edge:
                    self.sta_locs[i] = np.array([x,y])
                else:
                    self.sta_dirs[i] = self._get_random_dir()

    def check_cell_edge_snr(self):
        l = InterferenceHelper.fre_dis_to_loss_dB(self.fre_Hz,self.cell_edge/2*math.sqrt(2))
        s_db = self.txp_dbm_hi - l - self.bandwidth_txpr_to_noise_dBm(self.bandwidth)
        s_dec = InterferenceHelper.db_to_dec(s_db)
        print("snr_db", s_db, "snr_dec", s_dec)
        return
    
    def check_max_detectable_range(self):
        min_snr_db = InterferenceHelper.dec_to_db(self.min_s_n_ratio)
        l = self.txp_dbm_hi - min_snr_db - self.bandwidth_txpr_to_noise_dBm(self.bandwidth)
        tol = 0.001
        a = 0.01
        b = self.cell_edge*self.cell_size
        while abs(InterferenceHelper.fre_dis_to_loss_dB(self.fre_Hz,b) - InterferenceHelper.fre_dis_to_loss_dB(self.fre_Hz,a)) > tol:
            midpoint = (a + b) / 2
            if InterferenceHelper.fre_dis_to_loss_dB(self.fre_Hz,midpoint) == l:
                break
            elif (InterferenceHelper.fre_dis_to_loss_dB(self.fre_Hz,a)-l) * (InterferenceHelper.fre_dis_to_loss_dB(self.fre_Hz,midpoint)-l) < 0:
                b = midpoint
            else:
                a = midpoint

        print("maximum detectable range", (a + b) / 2)
        return 


if __name__ == "__main__":
    test_obj = WiFiNet()
    test_obj.check_cell_edge_snr()   
    test_obj.check_max_detectable_range()
    print(test_obj.get_sta_states()[0:5])
    for k in test_obj.get_sta_states():
        print(k.__len__())
        print(k)

    print(test_obj.n_ap,test_obj.n_sta)
    print(test_obj._get_loss_distance(10*1.4))
    print("get_contending_node_matrix",test_obj.get_contending_node_matrix().sum()/test_obj.n_sta)
    print("get_hidden_node_matrix",test_obj.get_hidden_node_matrix().sum()/test_obj.n_sta)
    print("get_interfering_node_matrix",test_obj.get_interfering_node_matrix().sum()/test_obj.n_sta)
    print(np.logical_and(np.logical_not(test_obj.get_interfering_node_matrix().astype(bool)),test_obj.get_contending_node_matrix().astype(bool).astype(int)).sum()/test_obj.n_sta)
    print(np.logical_and(np.logical_not(test_obj.get_contending_node_matrix().astype(bool)),test_obj.get_interfering_node_matrix().astype(bool).astype(int)).sum()/test_obj.n_sta)
    print(test_obj.get_contending_node_matrix().diagonal().sum())
    print(test_obj.get_hidden_node_matrix().diagonal().sum())
    print(test_obj.get_interfering_node_matrix().diagonal().sum())
    