import torch
from torch import nn
from torch.nn import functional as F
from .layers import Encoder, warp, image_demo


class Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        w, t = c.get('width',32), c.get('samples',5)
        self.encoder = Encoder(3,w)
        self.trajectory = nn.Linear(w,2*t)
        self.jacobian = nn.Conv2d(w,4,1)
        self.free = nn.Conv2d(w,2*t,1)
        self.uncertainty = nn.Conv2d(w,1,1)
        self.proximal = nn.Sequential(nn.Conv2d(6,w,3,padding=1),nn.SiLU(),nn.Conv2d(w,3,3,padding=1))

    def forward(self,b):
        observed = b['x'].float()
        h = self.encoder(observed)
        n,_,height,width = h.shape
        t = self.c.get('samples',5)
        trajectory = self.trajectory(h.mean((2,3))).reshape(n,t,2)
        trajectory = trajectory - trajectory.mean(1,keepdim=True)
        matrix = .1*self.jacobian(h).reshape(n,2,2,height,width)
        eye = torch.eye(2,device=h.device)[None,:,:,None,None]
        flow = torch.einsum('nijhw,ntj->ntihw',matrix+eye,trajectory)
        if self.c['variant']=='no_integrability':
            flow = self.free(h).reshape(n,t,2,height,width)
        gate = torch.exp(-F.softplus(self.uncertainty(h)))
        if self.c['variant']=='no_gate': gate = torch.ones_like(gate)
        latent = observed.detach().requires_grad_(True)
        if self.c['variant']=='baseline':
            latent = observed + self.proximal(torch.cat([observed,observed],1))
        else:
            for _ in range(self.c.get('iterations',3)):
                reblur = torch.stack([warp(latent,flow[:,i]) for i in range(t)]).mean(0)
                gradient = torch.autograd.grad((reblur-observed).square().sum()/2,latent,create_graph=self.training,retain_graph=True)[0]
                latent = latent-self.c.get('step',.1)*gate*gradient
                latent = latent+.1*self.proximal(torch.cat([latent,observed],1))
        reblur = torch.stack([warp(latent,flow[:,i]) for i in range(t)]).mean(0)
        return {'prediction':latent,'reblur':reblur,'flow':flow,'uncertainty':1-gate,'jacobian':matrix.flatten(1,2)}

    def objective(self,b,p):
        j = p['jacobian']
        regularity = (j[:,:,:,1:]-j[:,:,:,:-1]).abs().mean()+(j[:,:,1:]-j[:,:,:-1]).abs().mean()
        return F.l1_loss(p['prediction'],b['y'].float())+.2*F.l1_loss(p['reblur'],b['x'].float())+.01*regularity


def demo(seed,n,c):
    import numpy as np
    from scipy.ndimage import gaussian_filter
    d = image_demo(seed,n,mask=False)
    d['x'] = gaussian_filter(d['y'],sigma=(0,0,1,1)).astype(np.float32)
    return d


def measure(p,d,c):
    import numpy as np
    result = {'reblur_mae':float(np.mean(abs(p['reblur']-d['x']))),'mean_field_uncertainty':float(p['uncertainty'].mean())}
    if 'flow' in d: result['flow_epe']=float(np.linalg.norm(p['flow']-d['flow'],axis=2).mean())
    return result
