import cupy as cp
import numpy as np
import torch
import ThrustRTC as trtc
import CURandRTC as rndrtc


class ttt_havetrtc:
    def __init__(self,a,b=None,c=None):
        self.a = np.array(a)
        self.b = cp.array(a)
        self.c = trtc.DVCupyVector(self.b)
        # self.c = torch.tensor(c)

class ttt_havenotrtc:
    def __init__(self,a,b=None,c=None):
        self.a = np.array(a)
        self.b = cp.array(a)
        # self.c = trtc.DVCupyVector(self.b)
        # self.c = torch.tensor(c)

class ttt_haverndrtc:
    def __init__(self,a,b=None,c=None):
        self.a = np.array(a)
        self.b = cp.array(a)
        self.c = rndrtc.DVRNG()
        # self.c = torch.tensor(c)
