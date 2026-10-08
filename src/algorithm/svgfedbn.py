import copy
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

def is_bn_key(k):
    k = k.lower()
    return ("bn" in k or "running_mean" in k or "running_var" in k
            or "num_batches_tracked" in k)

def fedbn_aggregate(states, sizes):
    total = float(sum(sizes))
    out = copy.deepcopy(states[0])
    for k in out:
        if not is_bn_key(k):
            out[k] = sum(s[k] * float(n) for s,n in zip(states,sizes)) / total
    return out

class FedBNClient:
    def __init__(self, dataset, model_fn, args):
        self.dataset = dataset
        self.device = args.device
        self.model = model_fn()
        self.bn_state = {k:v.detach().clone() for k,v in self.model.state_dict().items()
                         if is_bn_key(k)}
        self.ce = nn.CrossEntropyLoss()

    def train(self, global_state, args):
        state = copy.deepcopy(global_state)
        state.update(self.bn_state)
        self.model.load_state_dict(state)
        self.model.train()
        loader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True)
        optimizer = SGD(self.model.parameters(), lr=args.lr)

        for _ in range(args.local_epochs):
            for x,y in loader:
                x,y=x.to(self.device),y.to(self.device)
                optimizer.zero_grad()
                _,logits=self.model(x)
                self.ce(logits,y).backward()
                optimizer.step()

        trained = copy.deepcopy(self.model.state_dict())
        self.bn_state = {k:v.detach().clone() for k,v in trained.items() if is_bn_key(k)}
        return trained, len(self.dataset)
