from sim_src.util import STATS_OBJECT, counted


class curr(STATS_OBJECT):
    MAX_N_STA = 1000
    WINDOW_SIZE = 10
    def __init__(self):
        self.n_sta = 20        
        self.reward_list = []
    
    @counted
    def update(self, indicator):
        self.reward_list.append(indicator)
        if len(self.reward_list) > self.WINDOW_SIZE:
            self.reward_list.pop(0)
            print(self.reward_list, sum(self.reward_list)/self.WINDOW_SIZE)
            ratio = sum(self.reward_list)/self.WINDOW_SIZE
            if ratio >= 0.9:
                self.n_sta += 50
                self.reward_list = []
        
    def get_n(self):
        return min(self.n_sta,self.MAX_N_STA)
    
    