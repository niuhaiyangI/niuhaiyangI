import pandas as pd
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
        self.heart_time=slot_size/self.fps
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
        self.rolling_window=4

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
        red_channel=W_list[:,2]
        green_channel=W_list[:,1]
        blue_channel = W_list[:,0]
        b,a = signal.butter(10,0.3, btype='highpass',fs=self.slot_size)
        # red_channel=signal.filtfilt(b,a,red_channel.tolist(),axis=0,padlen=self.slot_size-1)
        # green_channel = signal.filtfilt(b, a, green_channel.tolist(),axis=0,padlen=self.slot_size-1)
        # blue_channel = signal.filtfilt(b,a,blue_channel.tolist(),axis=0,padlen=self.slot_size-1)
        red_channel=signal.lfilter(b,a,red_channel)
        green_channel = signal.lfilter(b, a, green_channel)
        blue_channel = signal.lfilter(b,a,blue_channel)
        # sos= signal.butter(10,0.3, btype='highpass',fs=self.slot_size,output='sos')
        # zi=signal.sosfilt_zi(sos)
        # red_channel,_=signal.sosfilt(sos,red_channel,zi=zi)
        # green_channel,_ = signal.sosfilt(sos, green_channel,zi=zi)
        # blue_channel,_ = signal.sosfilt(sos,blue_channel,zi=zi)
        W_list[:,2]=torch.tensor(red_channel.copy())
        W_list[:,1]=torch.tensor(green_channel.copy())
        W_list[:,0]=torch.tensor(blue_channel.copy())
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
        print(self.Hz)
        x=(torch.tensor(range(self.slot_size)))/(self.slot_size-1)
        red_channel=self.W_c[:,2]
        green_channel=self.W_c[:,1]
        blue_channel = self.W_c[:,0]
        red_channel=(red_channel-red_channel.min())/(red_channel.max()-red_channel.min())
        green_channel = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
        blue_channel = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
        plt.close()
        plt.plot(x, red_channel, color='r', label='red_average')
        plt.plot(x, green_channel, color='g', label='green_average')
        plt.plot(x, blue_channel, color='b', label='blue_average')
        plt.show()


    def get_Non_fiducialFeature(self):
        print(self.heart_time)
        x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        red_channel = self.W_c[:, 2]
        green_channel = self.W_c[:, 1]
        blue_channel = self.W_c[:, 0]
        # red_channel = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min())
        # green_channel = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
        # blue_channel = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
        b1,a1=signal.butter(10,1,btype='high',fs=self.slot_size)
        b2,a2=signal.butter(10,2,btype='high',fs=self.slot_size)
        sf_r1=signal.lfilter(b1,a1,red_channel)
        sf_r2 = signal.lfilter(b2, a2, red_channel)
        sf_r1_rolling=pd.Series(sf_r1).rolling(window=self.rolling_window).mean()
        sf_r2_rolling = pd.Series(sf_r2).rolling(window=self.rolling_window).mean()
        peak_r1,_=signal.find_peaks(-sf_r1_rolling)
        peak_r2, _ = signal.find_peaks(-sf_r2_rolling)
        # print(sf_r1)
        plt.close()
        # plt.plot(peak_r1/(self.slot_size-1), sf_r1_rolling[peak_r1], 'x', color='black')
        # plt.plot(peak_r2/(self.slot_size-1), sf_r2_rolling[peak_r2], 'x', color='b')
        # plt.plot(x, sf_r1_rolling, color='r', label='green_average')
        # plt.plot(x, sf_r2_rolling, color='b', label='green_average')
        plt.plot(x, sf_r1, color='r', label='green_average')
        plt.plot(x, sf_r2, color='b', label='green_average')
        plt.show()




