import math

import numpy as np

from sim_src.sim_env.interference_helper import InterferenceHelper

class WiFiNet(InterferenceHelper):
    
    """
    this class implements an ap-sta network to setup wi-fi network simulation.
    
    """
    HIDDEN_LOSS = 200.
    TARGET_PACKET_LOSS = 1e-2

    N_PACKETS = 100
    #The energy (dBm) of a received signal should be higher than this threshold to allow the PHY layer to detect the signal.
    RxSensitivity = -95
    #Preamble is successfully detection if the SNR is at or above this value (expressed in dB).
    PreambleDetectionThreshold = 0.
    def __init__(self, cell_edge = 10., cell_size = 10, n_sta = 100, fre_Hz = 5.8e9, txp_dbm_hi = 0., packet_bit = 800, bandwidth_hz = 20e6, max_err = 1e-5, seed=1):
        """
        Initializes the simulation environment with the given parameters.

        Parameters:
        -----------
        cell_edge : float
            The edge length of a single cell in meters.
        cell_size : int
            The number of cells along one dimension of the grid.
        n_sta : float
            The number of STAs.
        fre_Hz : float
            The frequency of the signal in Hertz.
        txp_dbm_hi : float
            The transmission power of the access points (APs) in dBm.
        packet_bit : int
            The size of each packet in bits.
        bandwidth : float
            The bandwidth of the signal in Hertz.
        max_err : float
            The maximum allowable error rate.
        seed : int
            Seed for random number generation.
        """
        self.seed = seed
        self.rand_gen_loc = np.random.default_rng(seed)
        self.rand_gen_fad = np.random.default_rng(seed)
        self.rand_gen_mob = np.random.default_rng(seed)

        self.cell_edge = cell_edge
        self.cell_size = cell_size

        self.grid_edge = self.cell_edge * self.cell_size

        self.n_ap = int(self.cell_size ** 2)
        self.ap_offset = self.cell_edge / 2.

        self.n_sta = int(n_sta)

        self.fre_Hz = fre_Hz
        self.txp_dbm_hi = txp_dbm_hi
        self.packet_bit = packet_bit
        self.bandwidth_hz = bandwidth_hz
        self.max_err = max_err

        self.ap_locs = None
        self.sta_locs = None
        self.sta_dirs = None

        self.min_sinr = None
        self.loss = None

        self._config_ap_locs()
        self._config_sta_locs()
        self._config_sta_dirs()

    def apply_sta_filter(self, filter):
        self.n_sta = filter.sum()
        self.sta_locs = self.sta_locs[filter==1]
        self.sta_dirs = self.sta_dirs[filter==1]
        
    def get_loss_ap_ap(self):
        # Compute pairwise differences between STAs
        diffs = self.ap_locs[:, np.newaxis, :] - self.ap_locs[np.newaxis, :, :]  # Shape: (n_sta, n_sta, 2)
        
        # Compute Euclidean distances between STAs
        distances = np.linalg.norm(diffs, axis=2)  # Shape: (n_sta, n_sta)
        
        # Compute path loss based on distances
        losses = self._get_loss_distance(distances)  # Shape: (n_sta, n_sta)
        
        # Fill the diagonal with zeros (or np.inf if appropriate)
        np.fill_diagonal(losses, 0)
        
        return losses

    def get_loss_sta_ap(self):
        # Compute differences in coordinates
        sta_locs_expanded = self.sta_locs[:, np.newaxis, :]  # Shape: (n_sta, 1, 2)
        ap_locs_expanded = self.ap_locs[np.newaxis, :, :]     # Shape: (1, n_ap, 2)

        # Compute Euclidean distances between STAs and APs
        diff = sta_locs_expanded - ap_locs_expanded           # Shape: (n_sta, n_ap, 2)
        distances = np.linalg.norm(diff, axis=2)              # Shape: (n_sta, n_ap)

        # Compute path loss based on distances
        losses = self._get_loss_distance(distances)           # Shape: (n_sta, n_ap)

        return losses

    def get_loss_sta_sta(self):
        # Compute pairwise differences between STAs
        diffs = self.sta_locs[:, np.newaxis, :] - self.sta_locs[np.newaxis, :, :]  # Shape: (n_sta, n_sta, 2)
        
        # Compute Euclidean distances between STAs
        distances = np.linalg.norm(diffs, axis=2)  # Shape: (n_sta, n_sta)
        
        # Compute path loss based on distances
        losses = self._get_loss_distance(distances)  # Shape: (n_sta, n_sta)
        
        # Fill the diagonal with zeros (or np.inf if appropriate)
        np.fill_diagonal(losses, 0)
        
        return losses

    def _get_loss_between_locs(self, a, b):
        dis = np.linalg.norm(np.array(a)-np.array(b),ord=2)
        return self._get_loss_distance(dis)

    def _get_loss_distance(self, dis):
        #shadowing is disabled
        return InterferenceHelper.fre_dis_to_loss_dB(self.fre_Hz,dis)

    def get_sta_states(self):
        # Step 1: Compute the path loss matrix between all STAs and all APs
        loss_sta_ap = self.get_loss_sta_ap()  # Shape: (n_sta, n_ap)
        
        # Step 2: Determine which APs are within the threshold for each STA
        threshold = self.get_loss_sta_ap_threhold()
        within_threshold = loss_sta_ap <= threshold  # Shape: (n_sta, n_ap)
        
        # Step 3: For each STA, get the indices of APs within the threshold
        state_list = []
        for k in range(self.n_sta):
            # Get indices of APs within threshold for STA k
            ap_indices = np.where(within_threshold[k])[0]
            
            if ap_indices.size == 0:
                # If no APs are within threshold, append an empty array
                tmp_list = np.empty((0, 3))
            else:
                # Extract AP coordinates and losses
                ap_coords = self.ap_locs[ap_indices]  # Shape: (num_aps_within_threshold, 2)
                losses = loss_sta_ap[k, ap_indices]   # Shape: (num_aps_within_threshold,)
                
                # Combine coordinates and losses into a single array
                t = np.hstack((ap_coords, losses[:, np.newaxis]))  # Shape: (num_aps_within_threshold, 3)
                
                # Normalize the array
                t_normalized = self._normalize_sta_tuples(t)  # We'll modify this function to handle arrays
                
                # Sort the array based on the normalized loss (last column)
                sorted_indices = np.argsort(t_normalized[:, -1])
                tmp_list = t_normalized[sorted_indices]
            
            # Append the result to the state list
            state_list.append(tmp_list)
        
        return state_list
    
    def get_sta_to_associated_ap_loss(self):
        loss_sta_ap = self.get_loss_sta_ap()
        asso = np.argmin(loss_sta_ap,axis=1)
        S_loss = loss_sta_ap[:, asso]
        S_loss = -(S_loss - self.get_loss_sta_ap_threhold())/10
        S_loss[S_loss<=0] = 0   
        np.fill_diagonal(S_loss,0)
        A_loss = np.min(loss_sta_ap,axis=1)
        A_loss = -(A_loss - self.get_loss_sta_ap_threhold())/10
        return S_loss, A_loss
        
    def _normalize_sta_tuples(self, t_array):
        # t_array is of shape (num_samples, 3)
        t_array = np.array(t_array, dtype=float)
        t_array[:, 0] = (t_array[:, 0] - self.grid_edge / 2.0) / self.grid_edge
        t_array[:, 1] = (t_array[:, 1] - self.grid_edge / 2.0) / self.grid_edge
        t_array[:, 2] = -(t_array[:, 2] - self.get_loss_sta_ap_threhold()) / self.get_loss_sta_ap_threhold()
        return t_array
    
    def get_interfering_node_matrix(self):
        ret = np.zeros((self.n_sta,self.n_sta))
        loss_sta_ap = self.get_loss_sta_ap()
        asso = np.argmin(loss_sta_ap,axis=1)
        S_loss = loss_sta_ap[:, asso]
        ret[S_loss<=self.get_loss_sta_ap_threhold()] = 1   
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
    
    def get_CH_matrix(self):
        C = self.get_contending_node_matrix()
        H = self.get_hidden_node_matrix()
        ret = np.logical_or(C.astype(bool) , H.astype(bool)).astype(float)
        np.fill_diagonal(ret,0)
        return ret
    
    def get_loss_sta_ap_threhold(self):
        return self.txp_dbm_hi-self.RxSensitivity
    
    def get_loss_sta_sta_threhold(self):
        return self.txp_dbm_hi-self.RxSensitivity
    
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

    
    def check_snr_at_dis(self,dis):
        l = InterferenceHelper.fre_dis_to_loss_dB(self.fre_Hz,dis)
        s_db = self.txp_dbm_hi - l - self.bandwidth_hz_to_noise_dbm(self.bandwidth_hz)
        s_dec = InterferenceHelper.db_to_dec(s_db)
        print("snr_db", s_db, "snr_dec", s_dec, "distance", dis, "loss", l )
        return

    def check_cell_edge_snr(self):
        self.check_snr_at_dis(self.cell_edge/2*math.sqrt(2))
        self.check_snr_at_dis(0)
        return
    
    def check_detection_range(self):
        if self.RxSensitivity - self.bandwidth_hz_to_noise_dbm(self.bandwidth_hz) < self.PreambleDetectionThreshold:
            print(f"RxSensitivity {self.RxSensitivity} is too low to detect Preamble, snr {self.RxSensitivity - self.bandwidth_hz_to_noise_dbm(self.bandwidth_hz)}, snr_threshold{self.PreambleDetectionThreshold}" )
            return
        l = self.txp_dbm_hi - self.RxSensitivity
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

        print("maximum detectable range", (a + b) / 2, "tx power", self.txp_dbm_hi, "RxSensitivity",self.RxSensitivity, "loss", InterferenceHelper.fre_dis_to_loss_dB(self.fre_Hz,a))
        return

    def get_config(self):
        CMD_CONFIGS = {}
        CMD_CONFIGS["packetSize"] = int(self.packet_bit/8)
        CMD_CONFIGS["numPackets"] = self.N_PACKETS
        CMD_CONFIGS["n_ap"] = self.n_ap
        CMD_CONFIGS["n_sta"] = self.n_sta
        CMD_CONFIGS["TxPower"] = self.txp_dbm_hi
        CMD_CONFIGS["RxNoiseFigure"] = self.NOISEFIGURE
        CMD_CONFIGS["CcaEdThreshold"] = self.RxSensitivity 
        CMD_CONFIGS["RxSensitivity"] = self.RxSensitivity 
        CMD_CONFIGS["PreambleDetectionThresholdMinimumRssi"] = self.RxSensitivity 
        return CMD_CONFIGS
        
    def get_state(self):
        state = {}
        state["loss_ap_ap"] = self.get_loss_ap_ap()
        state["loss_sta_ap"] = self.get_loss_sta_ap()
        state["loss_sta_sta"] = self.get_loss_sta_sta()
        return state
    
    @staticmethod
    def evaluate_qos(ret):
        packet_loss_rate = WiFiNet.evaluate_bler(ret)
        qos_fail = (packet_loss_rate >= WiFiNet.TARGET_PACKET_LOSS)
        return qos_fail
        
    @staticmethod
    def evaluate_bler(ret):
        packet_loss_rate = 1. - ret/WiFiNet.N_PACKETS
        return packet_loss_rate
        


if __name__ == "__main__":
    # given a grid of K*K cells, assign a color to each cell such that no adjacent cells have the same color.
    # using 3 colors to color the grid.
    K=10
    cell_color = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            cell_color[i][j] = (i+j)%3
    print(cell_color)
    exit(0)
    test_obj = WiFiNet()
    
    # filter out all STAs in cell_color with color 0
    cell_color_flat = cell_color.ravel()
    filter = np.zeros(test_obj.n_sta)
    for i in range(test_obj.n_sta):
        sta_loc = test_obj.sta_locs[i]
        cell_x = int(sta_loc[0] // test_obj.cell_edge)
        cell_y = int(sta_loc[1] // test_obj.cell_edge)
        cell_index = cell_x * test_obj.cell_size + cell_y
        if cell_color_flat[cell_index] != 0:
            filter[i] = 1
    
    test_obj.apply_sta_filter(np.random.randint(0, 2, size=test_obj.n_sta))
    exit(0)

    test_obj.check_cell_edge_snr()   
    test_obj.check_detection_range()
    # test_obj.check_snr_at_dis(10)   
    # test_obj.check_snr_at_dis(100)   
    print(test_obj.get_sta_states()[0:5])
    for k in test_obj.get_sta_states():
        print(k.__len__())
        print(k)

    print(test_obj.n_ap,test_obj.n_sta)
    print(test_obj._get_loss_distance(10*1.4))
    print("get_contending_node_matrix",test_obj.get_contending_node_matrix().sum()/test_obj.n_sta)
    print("get_hidden_node_matrix",test_obj.get_hidden_node_matrix().sum()/test_obj.n_sta)
    print("get_interfering_node_matrix",test_obj.get_interfering_node_matrix().sum()/test_obj.n_sta)
    print("get_CH_matrix",test_obj.get_CH_matrix().sum()/test_obj.n_sta)
    print(np.logical_and(np.logical_not(test_obj.get_interfering_node_matrix().astype(bool)),test_obj.get_contending_node_matrix().astype(bool).astype(int)).sum()/test_obj.n_sta)
    print(np.logical_and(np.logical_not(test_obj.get_contending_node_matrix().astype(bool)),test_obj.get_interfering_node_matrix().astype(bool).astype(int)).sum()/test_obj.n_sta)
    print(test_obj.get_contending_node_matrix().diagonal().sum())
    print(test_obj.get_hidden_node_matrix().diagonal().sum())
    print(test_obj.get_interfering_node_matrix().diagonal().sum())
    
    print(test_obj.get_sta_states()[0].shape)
    