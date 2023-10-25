import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
"""
Consider your input u_n and u_next as [BatchSize,1,Height,Width]

"""



def loss_be(u_n, u_next, deltaT, domain,epsilon,type='ACNN'):
    # case2
    if type=='ACNN':
        residual = (u_next - u_n) - deltaT * (epsilon * epsilon * (FiniteDiff_LB(u_n, domain) + FiniteDiff_LB(u_next, domain)) / 2 + ( FF(u_n) + FF(u_next)) / 2)
    elif type=='mACNN':
        residual = u_next - u_n - deltaT * (epsilon * epsilon * (FiniteDiff_LB(u_n, domain) + FiniteDiff_LB(u_next, domain)) / 2 +(FF(u_next) + FF(u_n)) / 2 - (integral(FF(u_next)) + integral(FF(u_n))) / 2)

    return residual

def FF(U):
    ####Double-well potential function
    return U * (1 - U * U)
    ###Flory-Huggins potential function
    # return 0.4*(torch.log(1-U) -torch.log(1+U))+1.6*U


def FiniteDiff_LB(x, domain):
    # consider a left bottom 4X4 matrix as input given a 5X5 image. Periodic boundary condition is applied
    # x = x[:,:,1:,0:-1]
    #Note this works for LeftBottom Part only!
    h_x = x.size()[2]
    w_x = x.size()[3]
    lx = domain[1] - domain[0]
    ly = domain[3] - domain[2]
    dx = lx / (w_x)
    dy = ly / (h_x)

    FD_right = torch.zeros_like(x)
    FD_left = torch.zeros_like(x)
    FD_top = torch.zeros_like(x)
    FD_bot = torch.zeros_like(x)
    FD_right[:,:,:,w_x-1] = x[:,:,:,0]
    FD_right[:,:,:,0:w_x-1] = x[:,:,:,1:w_x]
    FD_left[:,:,:,0] = x[:,:,:,w_x-1]
    FD_left[:,:,:,1:w_x] = x[:,:,:,0:w_x-1]
    FD_top[:,:,0,:] = x[:,:,h_x-1,:]
    FD_top[:,:,1:h_x,:] = x[:,:,0:h_x-1,:]
    FD_bot[:,:,h_x-1,:] = x[:,:,0,:]
    FD_bot[:,:,0:h_x-1,:] = x[:,:,1:h_x,:]

    FD = (FD_right + FD_left - 2*x)/(dx**2) + (FD_top + FD_bot - 2*x)/(dy**2)
    return FD


def integral(x):
    h_x = x.size()[2]
    w_x = x.size()[3]
    a = x.mean([2, 3]).unsqueeze(2).unsqueeze(3)
    b = a.expand(-1, -1, h_x, w_x)
    return b

