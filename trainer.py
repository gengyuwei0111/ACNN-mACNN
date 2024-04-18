from model import CNN_CN
from dataset import Trainset_PDE_generated,Trainset_random_generated,Testset_random_generated

import time
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from loss import loss_be
from utils import save_checkpoints
import matplotlib.cm as cm
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class Trainer():
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

        #####Training set
        self.Snap_Training = Trainset_random_generated(self.trainingBatch,self.intial_snap, self.height, self.width)


        #####Testing set
        self.Snap_Testing =Testset_random_generated(self.testBatch,self.intial_snap, self.height, self.width)



        print(self.CNN)

        if self.device =='cuda':
            self.CNN = self.CNN.to(self.device)

            self.Snap_TrainingLB = self.Snap_Training().to(self.device)
            self.Snap_TestingLB = self.Snap_Testing().to(self.device)

            Dataset = self.Snap_TrainingLB
            self.dataloader = torch.utils.data.DataLoader(Dataset, batch_size=4, shuffle=True)

        self.model_name = f'NS_{self.model_type}_deltaT{self.deltaT}_{self.trainingBatch}_{self.width}_{self.height}_{self.TotalTrainingSnaps}_{self.mid_channels}Channels'

    def full(self,miss):

        full_left = torch.clone(miss[:, :, :, [0]])
        full = torch.cat((miss, full_left), 3)
        full_bottom = torch.clone(full[:, :, [-1], :])
        full = torch.cat((full_bottom, full), 2)

        return full

    def train(self):
        self.optimizer_Adam = optim.Adam(self.CNN.parameters(), lr=0.001)
        self.lr_scheduler = StepLR(self.optimizer_Adam, step_size=1, gamma=0.6)

        best_loss = 1e-10
        tt = time.time()
        self.CNN.train()
        print('Training Start...')

        for k in range(2):
            epoch = k
            for idx, (Snap_pre) in enumerate(self.dataloader):

                self.optimizer_Adam.zero_grad()
                m = 500

                for a in range(self.TotalTrainingSnaps):
                    kk = a
                    self.optimizer_Adam.zero_grad()
                    Snap_pre1 = torch.clone(Snap_pre)
                    for i in range(m):
                        self.optimizer_Adam.zero_grad()
                        Snap_Next = self.CNN(Snap_pre1,self.model_type)
                        residual = loss_be(Snap_pre1, Snap_Next, self.deltaT, self.domain,
                                           self.epsilon, self.model_type)
                        Loss = torch.nn.MSELoss()(residual, torch.zeros_like(residual))
                        Loss.backward()
                        self.optimizer_Adam.step()
                    if m <= 100:
                        m = 100
                    elif m > 100:
                        m=m-10
                    Snap_pre = torch.clone(Snap_Next.detach().cpu()).to('cuda')

                    if (self.TotalTrainingSnaps*epoch+ kk + 1) % 1 == 0:
                        Valid_Loss = Loss * 2
                        print(
                            f'#{self.TotalTrainingSnaps*epoch+ kk + 1:5d}: TrainingLoss={Loss.item():.2e}, lr={self.lr_scheduler.get_last_lr()[0]:.2e}, time={time.time() - tt:.2f}s')
                        is_best = Valid_Loss < best_loss
                        state = {
                            'epoch': epoch,
                            'state_dict': self.CNN.state_dict(),
                            'best_loss': best_loss
                        }
                        save_checkpoints(state, is_best, save_dir=f'{self.model_name}')
                        if is_best:
                            best_loss = Valid_Loss
                        tt = time.time()


                self.lr_scheduler.step()





        print('Training Finished!')
        self.CNN.eval()

        Snap_init0 = torch.clone(self.Snap_TestingLB)

        TimeSteps = 1000
        n=TimeSteps
        with torch.no_grad():
            Snap_Test_Next=Snap_init0
            for i in range(n):
                Snap_Test_Next = self.CNN(Snap_Test_Next,self.model_type)
                torch.cuda.empty_cache()

        pre = self.full(Snap_Test_Next)[0][0].cpu().detach().numpy()


        fig, ax = plt.subplots()
        im = ax.imshow(pre, interpolation='bilinear', cmap=cm.RdYlGn)
        plt.title(f'NN at T={n * self.deltaT}')
        plt.colorbar(im)
        plt.show()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
