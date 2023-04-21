import pandas as pd
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft,ifft
class SLOT:
    def __init__(self, slot_size, s_list, red_average,fps):
        self.gama=0
        self.fps=fps
        self.slot_size=slot_size
        self.W_c1=torch.zeros([self.slot_size,3])
        self.W_c2 = torch.zeros([self.slot_size, 3])
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
            score=score+i*i*(((self.diff[:,:,2]>=self.bin[i])&(self.diff[:,:,2]<self.bin[i+1])).sum()/(self.diff.shape[0]*self.diff.shape[1]))
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
        x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        red_channel = self.W_c1[:, 2]
        green_channel = self.W_c1[:, 1]
        blue_channel = self.W_c1[:, 0]
        red_channel = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min())
        green_channel = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
        blue_channel = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
        sum = (red_channel + green_channel + blue_channel) / 3
        check=sum
        Series = pd.Series(check.tolist())
        rol = Series.rolling(window=3).mean()
        peak, _ = signal.find_peaks(check.tolist())
        peak2, _ = signal.find_peaks((-check).tolist())



    def get_Non_fiducialFeature1(self):
        x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        red_channel = self.W_c1[:, 2]
        green_channel = self.W_c1[:, 1]
        blue_channel = self.W_c1[:, 0]
        red_channel = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min())
        green_channel = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
        blue_channel = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
        sum = (red_channel + green_channel + blue_channel) / 3
        check=sum
        Series = pd.Series(check.tolist())
        rol = Series.rolling(window=3).mean()
        peak, _ = signal.find_peaks(check.tolist())
        peak2, _ = signal.find_peaks((-check).tolist())



    def get_Non_fiducialFeature2(self):
        x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        red_channel = self.W_c1[:, 2]
        green_channel = self.W_c1[:, 1]
        blue_channel = self.W_c1[:, 0]
        red_channel = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min())
        green_channel = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
        blue_channel = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
        sum = (red_channel + green_channel + blue_channel) / 3
        check=sum
        Series = pd.Series(check.tolist())
        rol = Series.rolling(window=3).mean()
        peak, _ = signal.find_peaks(check.tolist())
        peak2, _ = signal.find_peaks((-check).tolist())

    def show_Wc(self):
        print('show start')
        x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        red_channel = self.W_c[:, 2]
        green_channel = self.W_c[:, 1]
        blue_channel = self.W_c[:, 0]
        red_channel = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min())
        green_channel = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
        blue_channel = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
        # sum=(red_channel+green_channel+blue_channel)/3
        sum=(red_channel+green_channel)/2
        plt.close()
        # plt.plot(x, self.W_c[:,2],color='pink')
        plt.plot(x, red_channel.tolist(), color='r', label='red_average')
        plt.plot(x, green_channel.tolist(), color='g', label='green_average')
        plt.plot(x, blue_channel.tolist(), color='b', label='blue_average')
        check=sum
        # check=red_channel
        plt.plot(x, check.tolist(), color='black', label='blue_average')
        Series = pd.Series(check.tolist())
        rol = Series.rolling(window=3).mean()
        peak, _ = signal.find_peaks(check.tolist())
        peak2, _ = signal.find_peaks((-check).tolist())
        plt.plot(x, rol, color='pink', label='blue_average')
        plt.plot(x[peak], check[peak].tolist(), "x", color='black',label='blue_average')
        plt.plot(x[peak2], check[peak2].tolist(), "x", color='pink', label='blue_average')
        plt.show()

    def show_Wc1(self):
        print('show1 start')
        x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        red_channel = self.W_c1[:, 2]
        green_channel = self.W_c1[:, 1]
        blue_channel = self.W_c1[:, 0]
        red_channel = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min())
        green_channel = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
        blue_channel = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
        sum=(red_channel+green_channel+blue_channel)/3
        plt.close()
        # plt.plot(x, self.W_c[:,2],color='pink')
        plt.plot(x, red_channel.tolist(), color='r', label='red_average')
        plt.plot(x, green_channel.tolist(), color='g', label='green_average')
        plt.plot(x, blue_channel.tolist(), color='b', label='blue_average')
        check=sum
        # check=red_channel
        plt.plot(x, check.tolist(), color='black', label='blue_average')
        Series = pd.Series(check.tolist())
        rol = Series.rolling(window=3).mean()
        peak, _ = signal.find_peaks(check.tolist())
        peak2, _ = signal.find_peaks((-check).tolist())
        plt.plot(x, rol, color='pink', label='blue_average')
        plt.plot(x[peak], check[peak].tolist(), "x", color='black',label='blue_average')
        plt.plot(x[peak2], check[peak2].tolist(), "x", color='pink', label='blue_average')
        plt.show()

    def show_Wc2(self):
        print('show2 start')
        x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        red_channel = self.W_c2[:, 2]
        green_channel = self.W_c2[:, 1]
        blue_channel = self.W_c2[:, 0]
        red_channel = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min())
        green_channel = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
        blue_channel = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
        sum=(red_channel+green_channel+blue_channel)/3
        plt.close()
        # plt.plot(x, self.W_c[:,2],color='pink')
        plt.plot(x, red_channel.tolist(), color='r', label='red_average')
        plt.plot(x, green_channel.tolist(), color='g', label='green_average')
        plt.plot(x, blue_channel.tolist(), color='b', label='blue_average')
        check=sum
        # check=red_channel
        plt.plot(x, check.tolist(), color='black', label='blue_average')
        Series = pd.Series(check.tolist())
        rol = Series.rolling(window=3).mean()
        peak, _ = signal.find_peaks(check.tolist())
        peak2, _ = signal.find_peaks((-check).tolist())
        plt.plot(x, rol, color='pink', label='blue_average')
        plt.plot(x[peak], check[peak].tolist(), "x", color='black',label='blue_average')
        plt.plot(x[peak2], check[peak2].tolist(), "x", color='pink', label='blue_average')
        plt.show()


    def S_feature_get(self,input):
        x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        band1=[0.1,0.3]
        default1=0.2
        band2=[0.2,0.45]
        default2 = 0.35
        band3=[0.6,1.0]
        default3 = 0.8
        check = input
        Series = pd.Series(check.tolist())
        rol = Series.rolling(window=3).mean()
        peak, _ = signal.find_peaks(check.tolist())
        peak2, _ = signal.find_peaks((-check).tolist())
        DP_x=self._get_default(default1)
        DP_t=list([x for x in peak if band1[0] <= x/(self.slot_size - 1) <= band1[1]])
        if len(DP_t):
            DP_x=DP_t[0]


        DN_x = self._get_default(default2)
        DN_t = list([x for x in peak if band2[0] <= x/(self.slot_size - 1) <= band2[1]])
        if len(DN_t):
            DN_x = DN_t[0]

        SP_x = self._get_default(default3)
        SP_t = list([x for x in peak if band3[0] <= x / (self.slot_size - 1) <= band3[1]])
        if len(SP_t):
            SP_x = SP_t[0]

        h1=input[DP_x]
        h2=input[DN_x]
        t1=DP_x/ (self.slot_size - 1)
        t2=(DN_x/ (self.slot_size - 1))-(DP_x/ (self.slot_size - 1))
        t3=(SP_x/ (self.slot_size - 1))-(DN_x/ (self.slot_size - 1))
        t4=1.0-(SP_x/ (self.slot_size - 1))

        s1=abs(h1/t1)
        s2 = abs(h2 / t2)
        s3 = abs(1.0 / t1)
        s4 = abs(input[self.slot_size-1] / t1)

        return h1,h2,t1,t2,t3,t4,s1,s2,s3,s4


    def _get_default(self,default):
        x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        temp=0
        min=self.fps*100
        for i in range(len(x)):
            dis=abs(x[i]-default)
            if dis<min:
                min=dis
                temp=i
        return temp
