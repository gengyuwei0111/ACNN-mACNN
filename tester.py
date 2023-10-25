from model import CNN_CN
from dataset import Trainset_PDE_generated,Testset_random_generated
import os
import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class Tester():
    def __init__(self,args):
        self.device = args.device
        self.args = args
        self.height = args.height
        self.width = args.width
        self.intial_snap = args.intial_snap
        self.mid_channels = args.mid_channels
        self.deltaT = args.deltaT
        self.trainingBatch = args.trainingBatch
        self.testBatch = args.testBatch
        self.domain=args.domain
        self.epsilon = args.epsilon
        self.TotalTrainingSnaps = args.TotalTrainingSnaps
        self.model_type = args.model_type

        self.device = 'cuda'

        self.CNN = CNN_CN(self.mid_channels)

        ######Random initial
        # self.Snap_Testing =Testset_random_generated(self.testBatch,self.intial_snap, self.height, self.width)
        #####Merging Bubbles case
        self.Snap_Testing = Trainset_PDE_generated(self.intial_snap, self.height, self.width, self.epsilon, self.domain)
        print(self.CNN)

        if self.device == 'cuda':
            self.CNN = self.CNN.to(self.device)
            self.Snap_TestingLB = self.Snap_Testing().to(self.device)


        self.model_name = f'NS_{self.model_type}_deltaT{self.deltaT}_{self.trainingBatch}_{self.width}_{self.height}_{self.TotalTrainingSnaps}_{self.mid_channels}Channels'
    def full(self,miss):

        full_left = torch.clone(miss[:, :, :, [0]])
        full = torch.cat((miss, full_left), 3)
        full_bottom = torch.clone(full[:, :, [-1], :])
        full = torch.cat((full_bottom, full), 2)

        return full

    def test(self,T):
        self.CNN.eval()
        best_model = torch.load(f'checkpoints/{self.model_name}/checkpoint.pth.tar')
        self.CNN.load_state_dict(best_model['state_dict'])


        Snap_init = torch.clone(self.Snap_TestingLB)

        TimeSteps = int(T/self.deltaT)
        n = TimeSteps
        with torch.no_grad():
            Snap_Test_Next = Snap_init
            for i in range(n):
                Snap_Test_Next = self.CNN(Snap_Test_Next, self.model_type)
                torch.cuda.empty_cache()



        Predict = self.full(Snap_Test_Next)[0][0].cpu().detach().numpy()
        initial = self.full(Snap_init)[0][0].cpu().detach().numpy()


        fig, ax = plt.subplots()
        im = ax.imshow(initial, interpolation='bilinear', cmap=cm.RdYlGn)
        plt.title(f'Initial snap')
        plt.colorbar(im)
        plt.show()

        fig, ax1 = plt.subplots()
        im1 = ax1.imshow(Predict, interpolation='bilinear', cmap=cm.RdYlGn)
        plt.title(f'NN at T={n * self.deltaT}')
        plt.colorbar(im1)
        plt.show()




if __name__ == '__main__':
    from options import Options

    args = Options().parse()
    tester = Tester(args)
    tester.test(args.TestingEndingTime)