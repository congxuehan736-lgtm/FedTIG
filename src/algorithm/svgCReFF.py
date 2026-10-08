import copy
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

def fedavg(states,sizes):
    total=float(sum(sizes));out=copy.deepcopy(states[0])
    for k in out:out[k]=sum(s[k]*float(n) for s,n in zip(states,sizes))/total
    return out

class CReFFClient:
    def __init__(self,dataset,model_fn,args):
        self.dataset=dataset;self.device=args.device
        self.model=model_fn();self.ce=nn.CrossEntropyLoss()

    def train(self,state,args):
        self.model.load_state_dict(state);self.model.train()
        opt=SGD(self.model.parameters(),lr=args.lr)
        loader=DataLoader(self.dataset,batch_size=args.batch_size,shuffle=True)
        for _ in range(args.local_epochs):
            for x,y in loader:
                x,y=x.to(self.device),y.to(self.device)
                opt.zero_grad();_,z=self.model(x)
                self.ce(z,y).backward();opt.step()
        return copy.deepcopy(self.model.state_dict()),len(self.dataset)

    def classifier_gradients(self,state,args):
        self.model.load_state_dict(state);self.model.eval()
        by={}
        loader=DataLoader(self.dataset,batch_size=args.creff_real_batch,shuffle=True)
        seen=set()
        for x,y in loader:
            x,y=x.to(self.device),y.to(self.device)
            for c in torch.unique(y).tolist():
                if c in seen: continue
                idx=(y==c).nonzero().flatten()
                if len(idx)==0: continue
                _,z=self.model(x[idx])
                target=torch.full((len(idx),),c,device=self.device,dtype=torch.long)
                loss=self.ce(z,target)
                params=[self.model.classifier.weight,self.model.classifier.bias]
                gs=torch.autograd.grad(loss,params)
                by[c]=[g.detach().clone() for g in gs]
                seen.add(c)
            if len(seen)>=args.num_classes: break
        return by

class CReFFServer:
    def __init__(self,args,model_fn):
        self.args=args;self.device=args.device;self.model_fn=model_fn
        self.feature_syn=torch.randn(args.num_classes*args.creff_ipc,args.feature_dim,
                                     device=self.device,requires_grad=True)
        self.labels=torch.arange(args.num_classes,device=self.device).repeat_interleave(args.creff_ipc)
        self.ce=nn.CrossEntropyLoss()

    def update_and_retrain(self,state,grad_lists,args):
        # Federate class-wise classifier gradients, optimize synthetic features,
        # then retrain the classifier on balanced synthetic features.
        clf=nn.Linear(args.feature_dim,args.num_classes).to(self.device)
        with torch.no_grad():
            clf.weight.copy_(state['classifier.weight'])
            clf.bias.copy_(state['classifier.bias'])
        opt_f=SGD([self.feature_syn],lr=args.creff_lr_feature)
        params=list(clf.parameters())

        for _ in range(args.creff_match_epoch):
            total=torch.zeros((),device=self.device); valid=0
            for c in range(args.num_classes):
                real=[g[c] for g in grad_lists if c in g]
                if not real: continue
                real_w=sum(g[0] for g in real)/len(real)
                real_b=sum(g[1] for g in real)/len(real)
                f=self.feature_syn[c*args.creff_ipc:(c+1)*args.creff_ipc]
                y=torch.full((args.creff_ipc,),c,device=self.device,dtype=torch.long)
                loss=self.ce(clf(f),y)
                gw=torch.autograd.grad(loss,params,create_graph=True)
                total += ((gw[0]-real_w.detach())**2).sum()/(real_w.detach().norm()**2+1e-12)
                total += ((gw[1]-real_b.detach())**2).sum()/(real_b.detach().norm()**2+1e-12)
                valid+=1
            if valid:
                opt_f.zero_grad();(total/valid).backward();opt_f.step()

        opt=SGD(clf.parameters(),lr=args.creff_lr_net)
        ds=TensorDataset(self.feature_syn.detach(),self.labels)
        for _ in range(args.creff_crt_epoch):
            for x,y in DataLoader(ds,batch_size=args.batch_size,shuffle=True):
                opt.zero_grad();self.ce(clf(x),y).backward();opt.step()

        out=copy.deepcopy(state)
        out['classifier.weight']=clf.weight.detach().clone()
        out['classifier.bias']=clf.bias.detach().clone()
        return out
