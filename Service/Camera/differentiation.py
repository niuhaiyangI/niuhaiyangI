import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
class SLOT:
    def __init__(self, slot_size, s_list, red_average,fps):
        self.gama=15
        self.fps=fps
        self.slot_size=slot_size
        self.Hz=1.0/(slot_size/self.fps)
        # print(1.0/self.Hz)
        self.slot_list=s_list
        self.red_average=red_average
        self.max_frame=s_list[self.red_average.argmax()]
        self.max_average=self.red_average.max()
        self.min_frame=s_list[self.red_average.argmin()]
        self.min_average=self.red_average.min()
        self.bin_size = 3
        self.bin = [x for x in range(256) if x % int(256 / self.bin_size) == 0]
        self.diff=self.max_frame-self.min_frame
        self.M=(self.diff>self.gama)
        self.score=self._cal_score()
        self.W_c=self._cal_W()
        self.DN_index=0

    def _cal_score(self):
        score=0
        for i in range(self.bin_size):
            score=score+i*i*(((self.diff>=self.bin[i])&(self.diff<self.bin[i+1])).sum()/(self.diff.shape[0]*self.diff.shape[1]))
        # print(score)
        return score

    def _cal_W(self):
        W_list=[]
        for img in self.slot_list:
            temp=torch.tensor(img*self.M).cuda().sum(dim=(0,1))/torch.tensor(self.M).cuda().sum(dim=(0,1))
            W_list.append(temp.tolist())
        W_list=torch.tensor(W_list)
        W_list.cuda()
        return W_list

    def show(self):
        plt.close()
        x=range(self.slot_size)
        plt.plot(x, self.red_average, color='r', label='red_average')
        # plt.plot(x,self.W_c[:,2], color='r', label='red_average')
        # plt.plot(x, self.W_c[:, 1], color='g', label='green_average')
        # plt.plot(x, self.W_c[:, 0], color='b', label='blue_average')
        plt.show()

    def _change(self,x):
        return (x)/(self.slot_size-1)

    def get_Systolic_DiastolicFeature(self):
        x=(torch.tensor(range(self.slot_size)))/(self.slot_size-1)
        red_channel=self.W_c[:,2]
        green_channel=self.W_c[:,1]
        blue_channel = self.W_c[:,0]
        red_channel=(red_channel-red_channel.min())/(red_channel.max()-red_channel.min())
        green_channel = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
        blue_channel = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
        sum=self.W_c.sum(dim=1)/3
        sum=(sum - sum.min()) / (sum.max() - sum.min())
        plt.close()
        plt.plot(x, red_channel, color='r', label='red_average')
        plt.plot(x, green_channel, color='g', label='green_average')
        plt.plot(x, blue_channel, color='b', label='blue_average')
        plt.plot(x, sum, color='black', label='blue_average')
        plt.show()
        red_list=[]
        green_list=[]
        blue_list=[]
        index_r=red_channel.argmax()
        index_g=green_channel.argmax()
        index_b=blue_channel.argmax()


    # def get_Non_fiducialFeature(self,slot):


