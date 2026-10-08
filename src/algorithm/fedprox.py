import copy
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

def fedavg(states, sizes):
    total = float(sum(sizes))
    out = copy.deepcopy(states[0])
    for k in out:
        out[k] = sum(s[k] * float(n) for s, n in zip(states, sizes)) / total
    return out

class FedProxClient:
    def __init__(self, dataset, model_fn, args):
        self.dataset = dataset
        self.device = args.device
        self.model = model_fn()
        self.ce = nn.CrossEntropyLoss()

    def train(self, global_state, args):
        self.model.load_state_dict(global_state)
        self.model.train()
        global_params = [p.detach().clone() for p in self.model.parameters()]
        loader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True)
        optimizer = SGD(self.model.parameters(), lr=args.lr)

        for _ in range(args.local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                _, logits = self.model(x)
                loss = self.ce(logits, y)
                prox = torch.zeros((), device=self.device)
                for p, gp in zip(self.model.parameters(), global_params):
                    prox = prox + torch.sum((p - gp) ** 2)
                (loss + 0.5 * args.mu * prox).backward()
                optimizer.step()

        return copy.deepcopy(self.model.state_dict()), len(self.dataset)
