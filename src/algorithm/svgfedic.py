import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

def fedavg(states,sizes):
    total=float(sum(sizes));out=copy.deepcopy(states[0])
    for k in out:
        out[k]=sum(s[k]*float(n) for s,n in zip(states,sizes))/total
    return out

class EnsembleHighway(nn.Module):
    """
    FEDIC-style ensemble/calibration gate.
    The original FEDIC repository uses a 3-member ensemble highway.
    Here the number of online clients is configurable for the unified
    CIFAR-10-LT experiment.
    """
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.gate = nn.Linear(feature_dim, 1)
        self.logit_scale = nn.Parameter(torch.ones(num_classes))
        self.logit_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, features, logits):
        mean_feature=torch.stack(features).mean(0)
        mean_logits=torch.stack(logits).mean(0)
        gate=torch.sigmoid(self.gate(mean_feature))
        calibrated=mean_logits*self.logit_scale+self.logit_bias
        return gate*calibrated+(1-gate)*mean_logits

class FEDICServer:
    def __init__(self,args,model_fn):
        self.args=args; self.device=args.device; self.model=model_fn()
        self.highway=EnsembleHighway(args.num_classes,args.feature_dim).to(self.device)
        self.ce=nn.CrossEntropyLoss()
        self.opt_h=Adam(self.highway.parameters(),lr=args.fedic_lr)
        self.opt_s=SGD(self.model.parameters(),lr=args.fedic_student_lr)

    def aggregate(self,states,sizes):
        return fedavg(states,sizes)

    def distill(self,state,client_states,teach_set,model_fn,args):
        self.model.load_state_dict(state)
        self.model.eval(); self.highway.train()
        clients=[]
        for st in client_states:
            m=model_fn();m.load_state_dict(st);m.eval();clients.append(m)

        loader=DataLoader(teach_set,batch_size=args.fedic_batch,shuffle=True)
        for _ in range(args.fedic_server_steps):
            for x,y in loader:
                x,y=x.to(self.device),y.to(self.device)
                feats=[];logs=[]
                with torch.no_grad():
                    for m in clients:
                        f,z=m(x);feats.append(f);logs.append(z)
                teacher=self.highway(feats,logs)
                loss=self.ce(teacher,y)
                self.opt_h.zero_grad();loss.backward();self.opt_h.step()
                break

        self.model.train();self.highway.eval()
        loader=DataLoader(teach_set,batch_size=args.fedic_batch,shuffle=True)
        for _ in range(args.fedic_distill_steps):
            for x,y in loader:
                x,y=x.to(self.device),y.to(self.device)
                with torch.no_grad():
                    feats=[];logs=[]
                    for m in clients:
                        f,z=m(x);feats.append(f);logs.append(z)
                    teacher=self.highway(feats,logs)
                _,student=self.model(x)
                hard=self.ce(student,y)
                soft=F.kl_div(F.log_softmax(student/args.fedic_T,1),
                              F.softmax(teacher/args.fedic_T,1),reduction='batchmean')*(args.fedic_T**2)
                loss=(1-args.fedic_lambda)*hard+args.fedic_lambda*soft
                self.opt_s.zero_grad();loss.backward();self.opt_s.step()
                break
        return copy.deepcopy(self.model.state_dict())
