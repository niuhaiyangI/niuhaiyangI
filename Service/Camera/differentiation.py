import torch
from torchvision import transforms
import numpy as np
class SLOT:
    def __init__(self, slot_size, s_list, red_average):
        self.slot_size=slot_size
        self.slot_list=s_list
        self.red_average=red_average
        # print(len(self.slot_list))
        # print(self.red_average.shape)
        self.max_frame=s_list[self.red_average.argmax()]
        self.max_average=self.red_average.max()
        self.min_frame=s_list[self.red_average.argmin()]
        self.min_average=self.red_average.min()
        self.bin_size = 3
        self.bin = [x for x in range(256) if x % int(256 / self.bin_size) == 0]
        self.diff=self.max_frame-self.min_frame
        self.score=self.cal_score()

    def cal_score(self):
        score=0
        for i in range(self.bin_size):
            score=score+i*i*(((self.diff>=self.bin[i])&(self.diff<self.bin[i+1])).sum()/(self.diff.shape[0]*self.diff.shape[1]))
        print(score)
        return score
