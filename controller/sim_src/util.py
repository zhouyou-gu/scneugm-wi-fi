import csv
import math
import os
from datetime import datetime
import pprint
from time import time


import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

class GlobalSeedGen():
    SEED = 0
    def __init__(self):
        GlobalSeedGen.SEED +=1 

def GetSeed():
    GlobalSeedGen()
    return GlobalSeedGen.SEED
    
def p_true(probability_of_true):
    return np.random.choice([True, False], p=[probability_of_true, 1 - probability_of_true])

def binarize_vector(prob_vector):
    binary_vector = np.random.binomial(1, prob_vector)
    return binary_vector
    

def DbToRatio(a):
    return 10.0**(0.1 * a)

def RatioToDb(a):
    return 10.0 * math.log10(a)

CUDA_AVAILABLE = torch.cuda.is_available()
USE_CUDA = CUDA_AVAILABLE
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LONG_FLOAT = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
INTEGER = torch.cuda.IntTensor if USE_CUDA else torch.IntTensor
LONG_INTEGER = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, requires_grad=False, dtype=FLOAT):
    t = torch.from_numpy(np.array(ndarray))
    t.requires_grad_(requires_grad)
    if USE_CUDA:
        return t.type(dtype).to(torch.cuda.current_device())
    else:
        return t.type(dtype)

def to_device(var):
    if USE_CUDA:
        return var.to(torch.cuda.current_device())
    return var

def pad_tensor_sequence(list_of_tensor_sequence, batch_first = True):
    train_lengths = [seq.size(0) for seq in list_of_tensor_sequence]
    padded_train_sequences = pad_sequence(list_of_tensor_sequence, batch_first=batch_first)
    return to_device(padded_train_sequences), train_lengths

def pad_numpy_sequence(list_of_sequences, padding_value=0, batch_first=True):
    """
    Pads a list of sequences (list of list of np.ndarray) with a specified padding value.

    Args:
        list_of_sequences (List[List[np.ndarray]]): 
            A list where each element is a sequence (list) of NumPy array vectors.
        padding_value (float or int, optional): 
            The value used for padding shorter sequences. Defaults to 0.
        batch_first (bool, optional): 
            If True, the output array shape will be (batch_size, max_seq_len, vector_dim).
            If False, the shape will be (max_seq_len, batch_size, vector_dim). Defaults to True.

    Returns:
        padded_sequences (np.ndarray): 
            A NumPy array containing all sequences padded to the same length.
        sequence_lengths (List[int]): 
            A list containing the original lengths of each sequence before padding.
    """
    # Calculate the length of each sequence
    sequence_lengths = [len(seq) for seq in list_of_sequences]
    
    # Determine the maximum sequence length
    max_seq_len = max(sequence_lengths)
    
    if max_seq_len == 0:
        raise ValueError("All sequences are empty. Cannot perform padding.")
    
    # Assume all vectors have the same shape
    # Retrieve the shape of a single vector from the first non-empty sequence
    for seq in list_of_sequences:
        if len(seq) > 0:
            vector_shape = seq[0].shape
            break
    else:
        raise ValueError("All sequences are empty. Cannot determine vector shape.")
    
    # Determine the shape of the padded array
    if batch_first:
        padded_shape = (len(list_of_sequences), max_seq_len) + vector_shape
    else:
        padded_shape = (max_seq_len, len(list_of_sequences)) + vector_shape
    
    # Initialize the padded array with the padding_value
    padded_sequences = np.full(padded_shape, padding_value, dtype=list_of_sequences[0][0].dtype)
    
    # Iterate over each sequence and pad accordingly
    for i, seq in enumerate(list_of_sequences):
        if len(seq) == 0:
            continue  # Skip empty sequences
        
        # Stack the vectors in the current sequence into a NumPy array
        stacked_seq = np.stack(seq)
        
        if batch_first:
            padded_sequences[i, :sequence_lengths[i]] = stacked_seq
        else:
            padded_sequences[:sequence_lengths[i], i] = stacked_seq
    
    return padded_sequences, sequence_lengths


def cat_str_dot_txt(sl):
    ret = ""
    for s in sl:
        ret += s
        ret += "."
    ret += "txt"

    return ret


def soft_update_inplace(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update_inplace(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def add_param_noise_inplace(target, std=0.01):
    for target_param in list(target.parameters()):
        d = np.random.randn(1)
        d = d * std
        d = to_tensor(d, requires_grad=False)
        target_param.data.add_(d)

def get_current_time_str():
    return datetime.now().strftime("%Y-%B-%d-%H-%M-%S")

def counted(f):
    def wrapped(self, *args, **kwargs):
        self.N_STEP += 1
        return f(self, *args, **kwargs)

    return wrapped

def timed(f):
    def wrapped(self, *args):
        ts = time()
        result = f(self, *args)
        te = time()
        print('%s func:%r took: %2.4f sec' % (self, f.__name__, te - ts))
        return result

    return wrapped

LOGGED_NP_DATA_HEADER_SIZE = 3


class ParameterConfig(dict):
    CONFIG_NAME = "sim_config"
    def save(self, path: str, postfix: str):
        try:
            os.mkdir(path)
        except:
            pass
        data_name = "%s.%s.txt" % (self.CONFIG_NAME,postfix)
        data_path = os.path.join(path,data_name)
        with open(data_path, 'w') as f:
            for key, value in self.items():
                f.write('%15s, %s\n' % (key, value))

class STATS_OBJECT:
    N_STEP = 0
    DISABLE_ALL_DEBUG = False
    DEBUG_STEP = 100
    DEBUG = False

    MOVING_AVERAGE_TIME_WINDOW = 100

    INIT_MOVING_AVERAGE = False
    INIT_LOGGED_NP_DATA = False
    INIT_TIMEING_OBJECT = False

    MOVING_AVERAGE_DICT = {}
    MOVING_AVERAGE_DICT_N_STEP = {}
    LOGGED_NP_DATA = {}

    LOGGED_CLASS_NAME = None

    PRINT_DIM = 5
    def save(self, path: str, postfix: str):
        pass

    def save_np(self, path: str, postfix: str):
        try:
            os.mkdir(path)
        except:
            pass
        for key in self.LOGGED_NP_DATA:
            if self.LOGGED_CLASS_NAME:
                data_name = "%s.%s.%s.txt" % (self.LOGGED_CLASS_NAME,key,postfix)
            else:
                data_name = "%s.%s.%s.txt" % (self.__class__.__name__,key,postfix)
            data_path = os.path.join(path,data_name)
            np.savetxt(data_path, self.LOGGED_NP_DATA[key] , delimiter=',')

    def _add_np_log(self, key, step, float_row_data, g_step=0):
        if not self.INIT_LOGGED_NP_DATA:
            self.LOGGED_NP_DATA = {}
            self.INIT_LOGGED_NP_DATA = True

        float_row_data = np.squeeze(float_row_data)
        assert isinstance(float_row_data, np.ndarray)
        assert float_row_data.ndim == 1 or float_row_data.ndim == 0
        if not (key in self.LOGGED_NP_DATA):
            self.LOGGED_NP_DATA[key] = np.zeros((0,float_row_data.size+LOGGED_NP_DATA_HEADER_SIZE))
        assert float_row_data.size + LOGGED_NP_DATA_HEADER_SIZE == self.LOGGED_NP_DATA[key].shape[1]
        s_t = np.array([g_step,step,time()])
        data = np.hstack((s_t,float_row_data))
        self.LOGGED_NP_DATA[key] = np.vstack((self.LOGGED_NP_DATA[key],data))

    def status(self):
        if self.DEBUG:
            pprint.pprint(vars(self))

    def _print(self, *args, **kwargs):
        if self.DEBUG and not STATS_OBJECT.DISABLE_ALL_DEBUG and (
                self.N_STEP % self.DEBUG_STEP == 0 or self.N_STEP % self.DEBUG_STEP == 1 or self.N_STEP % self.DEBUG_STEP == 2):
            print(("%10s\t" % self.__class__.__name__) + ("STEP:%5d, " % self.N_STEP) +  " ".join(map(str, args)), **kwargs)

    def _printalltime(self, *args, **kwargs):
        print(("%10s\t" % self.__class__.__name__) + ("STEP:%5d, " % self.N_STEP) +  " ".join(map(str, args)), **kwargs)

    def _moving_average(self, key, new_value):
        if not self.INIT_MOVING_AVERAGE:
            self.MOVING_AVERAGE_DICT = {}
            self.MOVING_AVERAGE_DICT_N_STEP = {}
            self.INIT_MOVING_AVERAGE = True

        if not (key in self.MOVING_AVERAGE_DICT):
            self.MOVING_AVERAGE_DICT[key] = 0.
            self.MOVING_AVERAGE_DICT_N_STEP[key] = 0.

        if key in self.MOVING_AVERAGE_DICT and key in self.MOVING_AVERAGE_DICT_N_STEP:
            step = self.MOVING_AVERAGE_DICT_N_STEP[key] + 1
            step = step if step < self.MOVING_AVERAGE_TIME_WINDOW else self.MOVING_AVERAGE_TIME_WINDOW

            self.MOVING_AVERAGE_DICT[key] = self.MOVING_AVERAGE_DICT[key] * (1.-1./step) + 1./step * new_value
            self.MOVING_AVERAGE_DICT_N_STEP[key] += 1

            return self.MOVING_AVERAGE_DICT[key]
        else:
            return 0.

    def _debug(self, ENABLE ,debug_step=100):
        self.DEBUG = ENABLE
        self.DEBUG_STEP = debug_step

    def _get_tic(self):
        if not self.INIT_TIMEING_OBJECT:
            self.timers = []
            self.ntimer = 0
            self.INIT_TIMEING_OBJECT = True

        self.ntimer += 1
        self.timers.append((self.ntimer,time()))
        return self.ntimer

    def _get_tim(self,tic_id):
        for t in self.timers:
            if t[0] == tic_id:
                tim = t[1]
                self.timers.remove(t)
                return (time()-tim)*1e6
        raise Exception("no timer is found.")



class CSV_WRITER_OBJECT:
    def __init__(self, path=None):
        self.path = path
        try:
            os.mkdir(self.path)
        except:
            pass
        self.files = {}
        self.writers = {}

    def log_one_scalar(self, data_name, iteration, value, g_iteration = 0):
        if self.path is None:
            return

        if data_name not in self.files.keys():
            self.files[data_name] = open(os.path.join(self.path,data_name), 'w', newline='')
            self.writers[data_name] = csv.writer(self.files[data_name])

        self.files[data_name].writerow([g_iteration, iteration, value])
        self.writers[data_name].flush()

    def log_mul_scalar(self, data_name, iteration, values, g_iteration = 0):
        if self.path is None:
            return

        if data_name not in self.files.keys():
            self.files[data_name] = open(os.path.join(self.path,data_name), 'w', newline='')
            self.writers[data_name] = csv.writer(self.files[data_name])

        self.writers[data_name].writerow([g_iteration, iteration]+ [v for v in values])
        self.files[data_name].flush()
    def close(self):
        for file in self.files.values():
            file.close()


def GET_LOG_PATH_FOR_SIM_SCRIPT(sim_script_path):
    OUT_ALL_SIM_FOLDER = "log-"+os.path.splitext(os.path.basename(sim_script_path))[0]
    OUT_ALL_SIM_FOLDER = os.path.join(os.path.dirname(os.path.realpath(sim_script_path)), OUT_ALL_SIM_FOLDER)
    try:
        os.mkdir(OUT_ALL_SIM_FOLDER)
    except:
        pass
    SIM_NAME_TIME = os.path.splitext(os.path.basename(sim_script_path))[0] + "-" + get_current_time_str() + "-ail"
    OUT_PER_SIM_FOLDER = os.path.join(OUT_ALL_SIM_FOLDER, SIM_NAME_TIME)
    return OUT_PER_SIM_FOLDER

def GET_FILE_NAME_FOR_SIM_SCRIPT(file):
    FILE_NAME = os.path.splitext(os.path.basename(file))[0]
    return FILE_NAME


def print_histogram_counts_one_line(arr, bins='auto', range=None):
    """
    Prints the counts of numbers in each bin (like a histogram), in one line.
    Automatically adjusts the number of bins based on the data.
    
    Parameters:
    arr (array-like): Input array of any shape (scalar, vector, or multidimensional matrix).
    bins (int, str, or sequence): Number of histogram bins, bin edges, or binning strategy.
                                  Default is 'auto' for automatic binning.
    range (tuple): Lower and upper range of the bins (optional).
    """
    arr = np.asarray(arr).flatten()
    
    # Handle the case where the array has no elements
    if arr.size == 0:
        print("The array is empty. No histogram to display.")
        return
    
    # Compute histogram data with automatic binning
    hist, bin_edges = np.histogram(arr, bins=bins, range=range)
    
    # Format counts per bin in one line, including bin ranges
    counts_str = ', '.join(f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): {count}" 
                           for i, count in enumerate(hist))
    print(f"Histogram Counts: {counts_str}")