import copy
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

def fedavg(states, sizes):
    total=float(sum(sizes)); out=copy.deepcopy(states[0])
    for k in out:
        out[k]=sum(s[k]*float(n) for s,n in zip(states,sizes))/total
    return out

class FedRSClient:
    """
    Restricted Softmax adaptation:
    only labels represented by the local client participate in the local softmax.
    This is the core FedRS mechanism; the surrounding CIFAR-LT/Dirichlet pipeline
    follows the user's project.
    """
    def __init__(self,dataset,model_fn,args):
        self.dataset=dataset; self.device=args.device
        self.model=model_fn(); self.ce=nn.CrossEntropyLoss()

    def _present_classes(self,num_classes):
        cls=set()
        for i in range(len(self.dataset)):
            cls.add(int(self.dataset[i][1]))
        return sorted(cls)

    def train(self,state,args):
        self.model.load_state_dict(state); self.model.train()
        mask=torch.zeros(args.num_classes,dtype=torch.bool,device=self.device)
        mask[self._present_classes(args.num_classes)]=True
        loader=DataLoader(self.dataset,batch_size=args.batch_size,shuffle=True)
        opt=SGD(self.model.parameters(),lr=args.lr)

        for _ in range(args.local_epochs):
            for x,y in loader:
                x,y=x.to(self.device),y.to(self.device)
                opt.zero_grad()
                _,z=self.model(x)
                z=z.masked_fill(~mask.unsqueeze(0),-1e9)
                self.ce(z,y).backward()
                opt.step()
        return copy.deepcopy(self.model.state_dict()),len(self.dataset)
