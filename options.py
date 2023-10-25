import argparse
import torch


class Options(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--no_cuda',
                            action='store_true',
                            default=False,
                            help='Disables CUDA training.')
        parser.add_argument('--cuda_index',
                            type=int,
                            default=0,
                            help='Cuda index you want to choose.')
        parser.add_argument('--height',
                            type=int,
                            default=256,
                            help='number of points in one dimension')
        parser.add_argument('--width',
                            type=int,
                            default=256,
                            help='number of points in one dimension')
        parser.add_argument('--intial_snap',
                            type=int,
                            default=1,
                            help='initial_snap')
        parser.add_argument('--mid_channels',
                            type=int,
                            default=16,
                            help='number of mid_channel')
        parser.add_argument('--deltaT',
                            type=float,
                            default=0.1,
                            help='deltaT')
        parser.add_argument('--trainingBatch',
                            type=int,
                            default=20,
                            help='number of training input images')
        parser.add_argument('--testBatch',
                            type=int,
                            default=1,
                            help='number of test input images')
        parser.add_argument('--domain',
                            nargs='+',
                            type=float,
                            default=[-0.5,0.5,-0.5,0.5],
                            help='domain')
        parser.add_argument('--model_type',
                            type=str,
                            default='mACNN',
                            help='type of training model')
        parser.add_argument('--epsilon',
                            type=float,
                            default=0.01,
                            help='epsilon')
        parser.add_argument('--TotalTrainingSnaps',
                            type=int,
                            default=10,
                            help='number of time steps')
        parser.add_argument('--TestingEndingTime',
                            type=int,
                            default=50,
                            help='Testing End Time')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        if args.cuda:
            args.device = 'cuda'
        else:
            args.device = 'cpu'
        return args
if __name__ == '__main__':
    args = Options().parse()
    print(args)