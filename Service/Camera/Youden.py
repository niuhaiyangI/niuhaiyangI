import cv2
import torch

from Camera.finger_cover import video
from Camera.divide import divide
import numpy as np
import math

import os
class youdenJ:
    def __init__(self):
        self.profilepath = '../Profile/User'
        self.path='D:\\毕业设计\\niuhaiyangI\\实验素材\\test'
        self.pos_list=[]
        self.pos_filenames=[]
        self.neg_list=[]
        self.neg_filenames=[]
        self.total=[]
        self.eta=6.0
        self.getPosList()
        self.getNgeList()
        print("pos_len:{}".format(len(self.pos_list)))
        print("neg_len:{}".format(len(self.neg_list)))


    def get_dist(self,path):
        camera = cv2.VideoCapture(path)
        if camera.isOpened():
            print("开始")
        else:
            camera.open(video)
            if camera.isOpened():
                print("成功")
        div = divide(camera)
        l,_,_,_,_=div.getDistList()
        return l


    def getPosList(self):
        path=os.path.join(self.path,'pos')
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                l=self.get_dist(filepath)
                for i in l:
                    self.pos_list.append(i)
                    self.total.append(i)
                    self.pos_filenames.append(filename)



    def getNgeList(self):
        path = os.path.join(self.path, 'neg')
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                l = self.get_dist(filepath)
                for i in l:
                    self.neg_list.append(i)
                    self.total.append(i)
                    self.neg_filenames.append(filename)

    def getSens(self,eta):
        pos_d=torch.tensor(self.pos_list)
        return (pos_d<=eta).sum()/pos_d.shape[0]

    def getSpec(self,eta):
        neg_d=torch.tensor(self.neg_list)
        return (neg_d>eta).sum()/neg_d.shape[0]

    def getJ(self,eta,neg_max=False):
        if neg_max:
            return self.getSpec(eta)
        else:
            return self.getSens(eta) + self.getSpec(eta) - 1


    def getMaxeEta(self,neg_max=False):
        result_eta = torch.tensor(self.pos_list).max()
        try:
            load_dict = np.load(os.path.join(self.profilepath,'eta_dic.npy'), allow_pickle=True).item()
            result_eta=load_dict['eta']
            sens = load_dict['sens']
            spec = load_dict['spec']
            pos = load_dict['pos']
            neg = load_dict['neg']
            negMax=load_dict['negMax']
            if (np.array(pos) == np.array(self.pos_filenames)).all() and (np.array(neg) == np.array(self.neg_filenames)).all() and negMax==neg_max:
                # print("exist:")
                # print("eta")
                # print(result_eta)
                # print("sens")
                # print(sens)
                # print("spec")
                # print(spec)
                return result_eta, sens, spec
            else:
                print("not exist")
                max = -10.0
                for eta in self.total:
                    temp_J = self.getJ(eta,neg_max=neg_max)
                    if temp_J > max:
                        max = temp_J
                        result_eta = eta
                sens = self.getSens(result_eta)
                spec = self.getSpec(result_eta)
                eta_dic = {'eta': result_eta, 'sens': sens, 'spec': spec, 'pos': self.pos_filenames,
                           'neg': self.neg_filenames,'negMax':neg_max}
                np.save(os.path.join(self.profilepath, 'eta_dic.npy'), eta_dic, allow_pickle=True)
                # print("eta")
                # print(result_eta)
                # print("sens")
                # print(sens)
                # print("spec")
                # print(spec)
                return result_eta, sens, spec
        except:
            max=-10.0
            for eta in self.total:
                temp_J=self.getJ(eta,neg_max=neg_max)
                if temp_J>max:
                    max=temp_J
                    result_eta=eta
            sens=self.getSens(result_eta)
            spec=self.getSpec(result_eta)
            eta_dic={'eta':result_eta,'sens':sens,'spec':spec,'pos':self.pos_filenames,'neg':self.neg_filenames,'negMax':neg_max}
            np.save(os.path.join(self.profilepath,'eta_dic.npy'),eta_dic,allow_pickle=True)
            # print("except:")
            # print("eta")
            # print(result_eta)
            # print("sens")
            # print(sens)
            # print("spec")
            # print(spec)
            return result_eta,sens,spec


    def getAccurate(self,n):
        # print("验证拒绝成功概率")
        eta, sens, spec = self.getMaxeEta()
        thsis=1.0-math.pow((1.0-spec),n)
        contTrue=0
        cont=0
        flag=[False]*n
        # print(len(flag))
        for i in range(len(self.neg_list)):
            flag[i%n]=(self.neg_list[i]>eta)
            if (i+1)%n==0:
                if  np.array(flag).any():
                    contTrue = contTrue + 1
                cont=cont+1
        # print('周期次数:{}'.format(n))
        # print('理论概率:{}'.format(thsis))
        # print('实际概率:{}'.format(contTrue/cont))
        return thsis,contTrue/cont

    def getFalserate(self, n):
        # print("验证接受成功概率")
        eta, sens, spec = self.getMaxeEta()
        thsis = math.pow(sens,n)
        contTrue = 0
        cont = 0
        flag = [False] * n
        # print(len(flag))
        for i in range(len(self.pos_list)):
            flag[i % n] = (self.pos_list[i] <= eta)
            if (i + 1) % n == 0:
                if np.array(flag).all():
                    contTrue = contTrue + 1
                cont = cont + 1
        # print('周期次数:{}'.format(n))
        # print('理论概率:{}'.format(thsis))
        # print('实际概率:{}'.format(contTrue / cont))
        return thsis, contTrue / cont



if __name__ == '__main__':
    ydj=youdenJ()
    max=-10.0
    max_i=0
    spec_max=0
    sens_max=0
    specMaxt = 0
    sensMaxt = 0
    for i in range(1,11):
        spec_t,spec_real=ydj.getAccurate(i)
        sens_t,sens_real=ydj.getFalserate(i)
        TJ=spec_real+sens_real-1.0
        print('\hline')
        print('{}&{.4f}&{.4f}&{.4f}&{.4f}&{.4f}&{.4f}&{.4f}&{.4f}&{.4f}\ \ '.format(i,sens_real,spec_real,1-spec_real,1-sens_real,TJ,sens_t,spec_t,1-spec_t,1-sens_t))
        if TJ>max:
            max=TJ
            max_i=i
            spec_max=spec_real
            sens_max=sens_real
            specMaxt=spec_t
            sensMaxt=sens_t
    print('sensT:{}'.format(sensMaxt))
    print('specT:{}'.format(specMaxt))
    print('sens:{}'.format(sens_max))
    print('spec:{}'.format(spec_max))
    print('最佳周期数:{}'.format(max_i))
    # eta, sens, spec = ydj.getMaxeEta()
    # eta,sens,spec=ydj.getMaxeEta(neg_max=True)




