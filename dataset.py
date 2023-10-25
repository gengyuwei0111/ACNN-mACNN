import torch
torch.set_default_tensor_type(torch.DoubleTensor)

class Trainset_random_generated(object):
    def __init__(self, batch, strating_snap, height, width):
        self.batch = batch
        self.strating_snap = strating_snap
        self.height = height
        self.width = width

    def __call__(self):
        torch.manual_seed(88)
        x=torch.rand(self.batch, self.strating_snap, self.height, self.width)
        self.snapTrain= ((-1-1) * x+1)*0.9
        return self.snapTrain


class Trainset_PDE_generated(object):#For merging bubble
    def __init__(self,strating_snap, height, width, epsilon,domain):
        self.strating_snap = strating_snap
        self.height = height
        self.width = width

        self.lbx = domain[0]
        self.ubx = domain[1]
        self.lby = domain[2]
        self.uby = domain[3]
        self.epsilon = epsilon

        self.xx = torch.linspace(self.lbx,self.ubx,steps = self.width+1)
        self.yy = torch.linspace(self.lby, self.uby, steps = self.height+1)
        ##
    def __call__(self):
        X, Y = torch.meshgrid(self.xx, self.yy)
        Z = torch.max(torch.tanh(
            (0.2 - torch.sqrt((X - 0.14) * (X - 0.14) + Y * Y)) / (self.epsilon)),
            torch.tanh((0.2 - torch.sqrt((X + 0.14) * (X + 0.14) + Y * Y)) / (
                self.epsilon)))
        self.snapTrain = torch.unsqueeze(torch.unsqueeze(Z, 0), 0)[:, :, 1:, 0:-1]

        return self.snapTrain


class Testset_random_generated(object):
    def __init__(self, batch, strating_snap, height, width):
        self.batch = batch
        self.strating_snap = strating_snap
        self.height = height
        self.width = width

    def __call__(self, seed=88):
        torch.manual_seed(seed)
        x=torch.rand(self.batch, self.strating_snap, self.height, self.width)
        self.snapTrain= ((-1-1) * x+1)*0.9
        return self.snapTrain
