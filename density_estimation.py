import os
import argparse
import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.datasets import make_gaussian_quantiles
import torch
import torch.nn as nn
import torch.optim as optim
import random
from model.layers import *

torch.manual_seed(27)
random.seed(27)
np.random.seed(27)

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=5000)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--n_step', type=int, default=1)
parser.add_argument('--n_depth', type=int, default=8)
parser.add_argument('--width', type=int, default=24)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
# input_size, n_step, n_depth, width




class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def eight_gaussian(num_samples):
    z = np.random.randn(num_samples, 2)
    scale = 4
    sq2 = 1/np.sqrt(2)
    centers = [(1,0), (-1,0), (0,1), (0,-1),
                (sq2,sq2), (-sq2,sq2), (sq2,-sq2), (-sq2,-sq2)]
    centers = np.array([(scale * x, scale * y) for x,y in centers])
    x = sq2 * (0.5 * z + centers[np.random.randint(len(centers), 
                                                    size=(num_samples,))])
    print(x.shape)
    return x


def get_batch(num_samples):
    # points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.5)
    z = torch.randn(num_samples, 2)
    scale = 4
    sq2 = 1/np.sqrt(2)
    centers = [(1,0), (-1,0), (0,1), (0,-1),
                (sq2,sq2), (-sq2,sq2), (sq2,-sq2), (-sq2,-sq2)]
    centers = torch.tensor([(scale * x, scale * y) for x,y in centers])
    x = sq2 * (0.5 * z + centers[torch.randint(len(centers), 
                                                    size=(num_samples,))])
    
    x = x.type(torch.float32).to(device)
    # x = torch.tensor(points).type(torch.float32).to(device)


    return x


if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    # model
    # Define the prior distribution, usually diagonal Gaussian, this distribution takes cpu tensors as inputs and outputs
    p_z0 = DiagGaussian(torch.tensor([0.0, 0.0]), torch.tensor([[1., 0.0], [0.0, 1.]])) 
    # initialize a KRnet, a normalizing flow model
    flow = KRnet(p_z0,args.input_size,args.n_step,args.n_depth,args.width,2,device=device).to(device=device)
    # rand_in = torch.rand(100, 2, device=device)
    # z,log_det1 = flow.forward(rand_in)
    # rec_in,log_det2 = flow.inverse(z)
    # print(torch.square(rec_in-rand_in).sum())
    # print(log_det1, log_det2)
    optimizer = optim.AdamW(flow.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

    loss_meter = RunningAverageMeter()


    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()

        x = get_batch(args.num_samples)
        log_px = flow.log_prob(x)
        loss= -log_px.mean()  
        loss.backward()
        # nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0, norm_type=2)
        optimizer.step()

        loss_meter.update(loss.item())
        if itr%100==0:
            lr_scheduler.step()
            viz_samples = 30000
            viz_timesteps = 41
            target_sample = get_batch(viz_samples)
            # target_sample = eight_gaussian(viz_samples)
            with torch.no_grad():
                # Generate evolution of samples
                z_t_samples = flow.sample(viz_samples)
                print(z_t_samples.shape)

                # Generate evolution of density
                x = np.linspace(-4.5, 4.5, 100)
                y = np.linspace(-4.5, 4.5, 100)
                ranges = [-4.5,4.5]
                points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

                z_t1 = torch.tensor(points).type(torch.float32).to(device)

                z_density = torch.exp(flow.log_prob(z_t1))
                print('estimate integral pdf:', 8.1e-3*np.sum(z_density.cpu().numpy()))
                fig = plt.figure(figsize=(12, 4), dpi=200)
                # plt.tight_layout()
                # plt.axis('off')
                # plt.margins(0, 0)
                # fig.suptitle(f'{t:.2f}s')

                ax1 = fig.add_subplot(1, 3, 1)
                ax1.set_title('Target')
                ax1.get_xaxis().set_ticks([])
                ax1.get_yaxis().set_ticks([])
                ax2 = fig.add_subplot(1, 3, 2)
                ax2.set_title('Samples')
                ax2.get_xaxis().set_ticks([])
                ax2.get_yaxis().set_ticks([])
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.set_title('Probability')
                ax3.get_xaxis().set_ticks([])
                ax3.get_yaxis().set_ticks([])

                ax1.hist2d(*target_sample.detach().cpu().numpy().T, bins=300, density=True,
                range=[ranges, ranges])
                # ,range=[[-1.5, 1.5], [-1.5, 1.5]]

                ax2.hist2d(*z_t_samples.detach().cpu().numpy().T, bins=300, density=True,range=[ranges, ranges])
                #,range=[[-1.5, 1.5], [-1.5, 1.5]]

                ax3.contourf(x, y, z_density.cpu().numpy().reshape(100,100), levels=50, origin='lower')

                # ax3.tricontourf(*z_t1.detach().cpu().numpy().T,
                #                 z_density.detach().cpu().numpy(), 200)
                print('save fig')
                # plt.savefig("./flow_sample.pdf",
                #             pad_inches=0.2, bbox_inches='tight')
                plt.savefig("./flow_sample.png")
                plt.close()

        print('Iter: {}, running avg loss: {:.4f}'.format(itr, loss_meter.avg))

    print('Training complete after {} iters.'.format(itr))

