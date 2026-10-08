"""
CIFAR-10-LT baseline suite for the user's FedLF project.

Runs:
    FedAvg, FedProx, FedBN, FedRS, FEDIC, CReFF

Unified setting:
    CIFAR-10-LT, IF=100, 20 clients, 8 online clients,
    Dirichlet alpha=0.5, 200 rounds, 5 local epochs,
    batch=32, SGD, lr=0.1, seed=42.

The project main.py/fedlf.py are NOT modified.

Important reproducibility note:
The authors' official repositories use different original data/training
pipelines. This script keeps the user's existing CIFAR-10-LT + Dirichlet
pipeline and adapts the algorithmic core. It should be reported as a
re-implementation under a unified setting, not as byte-for-byte reproduction.
"""

import argparse, copy, logging, os, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

from Model.Resnet8 import ResNet_cifar
from Dataset.long_tailed_cifar10 import train_long_tail
from Dataset.sample_dirichlet import clients_indices
from algorithm.fedprox import fedavg, FedProxClient
from algorithm.svgfedbn import FedBNClient, fedbn_aggregate
from algorithm.svgfedrs import FedRSClient
from algorithm.svgfedic import FEDICServer
from algorithm.svgCReFF import CReFFClient, CReFFServer

ROOT=Path(__file__).resolve().parent
LOGDIR=ROOT/"Logs";OUTDIR=ROOT/"outputs"
LOGDIR.mkdir(exist_ok=True);OUTDIR.mkdir(exist_ok=True)

class IndexedDataset(Dataset):
    def __init__(self,base,indices):
        self.base=base;self.indices=list(indices)
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):return self.base[self.indices[i]]

class StandardClient:
    def __init__(self,dataset,model_fn,args):
        self.dataset=dataset;self.device=args.device;self.model=model_fn();self.ce=nn.CrossEntropyLoss()
    def train(self,state,args):
        self.model.load_state_dict(state);self.model.train()
        opt=SGD(self.model.parameters(),lr=args.lr)
        loader=DataLoader(self.dataset,batch_size=args.batch_size,shuffle=True)
        for _ in range(args.local_epochs):
            for x,y in loader:
                x,y=x.to(self.device),y.to(self.device)
                opt.zero_grad();_,z=self.model(x);self.ce(z,y).backward();opt.step()
        return copy.deepcopy(self.model.state_dict()),len(self.dataset)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def build_model(args):
    return ResNet_cifar(resnet_size=8,scaling=4,save_activations=False,
        group_norm_num_groups=None,freeze_bn=False,freeze_bn_affine=False,
        num_classes=args.num_classes).to(args.device)

def evaluate(model,dataset,args):
    model.eval();correct=total=0
    loader=DataLoader(dataset,batch_size=args.test_batch,shuffle=False)
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(args.device),y.to(args.device)
            _,z=model(x);correct+=(z.argmax(1)==y).sum().item();total+=y.numel()
    return correct/max(total,1)

def setup_logger(seed):
    logger=logging.getLogger("cifar10_baselines");logger.setLevel(logging.INFO);logger.handlers.clear()
    fh=logging.FileHandler(LOGDIR/f"cifar10_baselines_seed{seed}.log",mode="w",encoding="utf-8")
    sh=logging.StreamHandler(sys.stdout)
    fmt=logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt);sh.setFormatter(fmt);logger.addHandler(fh);logger.addHandler(sh)
    return logger

def make_data(args,logger):
    root=args.data_root
    if not Path(root).exists():Path(root).mkdir(parents=True,exist_ok=True)
    tf=transforms.Compose([
        transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    tef=transforms.Compose([
        transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    train=datasets.CIFAR10(root,train=True,download=False,transform=tf)
    test=datasets.CIFAR10(root,train=False,download=False,transform=tef)

    # The existing project long-tail function expects label -> original indices.
    labels=np.asarray(train.targets)
    label2=[np.where(labels==c)[0].tolist() for c in range(args.num_classes)]
    img_num_list, per_class=train_long_tail(label2,args.num_classes,args.imb_factor,"exp")
    lt_indices=[i for group in per_class for i in group]

    # Rebuild label->LT indices and use the project's original Dirichlet function.
    lt_label2=[[] for _ in range(args.num_classes)]
    for idx in lt_indices:lt_label2[int(labels[idx])].append(int(idx))
    cidx=clients_indices(lt_label2,args.num_classes,args.num_clients,args.alpha,args.seed)

    client_sets=[IndexedDataset(train,idx) for idx in cidx]
    logger.info("CIFAR-10-LT class counts: %s",img_num_list)
    logger.info("Total LT samples: %d",len(lt_indices))
    logger.info("Client sizes: %s",[len(x) for x in client_sets])
    logger.info("Min/Max/Avg client size: %d/%d/%.2f",
                min(map(len,client_sets)),max(map(len,client_sets)),
                sum(map(len,client_sets))/len(client_sets))

    # FEDIC needs a server teaching set. Use a deterministic 10% holdout
    # from the LT training pool. It is not the test set.
    rng=np.random.RandomState(args.seed)
    teach_n=max(args.fedic_teach_min,int(len(lt_indices)*args.fedic_teach_ratio))
    teach_idx=rng.choice(lt_indices,size=min(teach_n,len(lt_indices)),replace=False).tolist()
    teaching=IndexedDataset(train,teach_idx)
    return client_sets,test,teaching

def run_standard(name,client_cls,client_sets,test,args,logger):
    model_fn=lambda:build_model(args)
    clients=[client_cls(ds,model_fn,args) for ds in client_sets]
    server=model_fn();state=copy.deepcopy(server.state_dict())
    rng=np.random.RandomState(args.seed)
    best=0
    for r in range(1,args.rounds+1):
        selected=rng.choice(args.num_clients,args.online,replace=False)
        states=[];sizes=[]
        logger.info("%s Round %d selected clients: %s",name,r,selected.tolist())
        for cid in selected:
            st,n=clients[int(cid)].train(copy.deepcopy(state),args);states.append(st);sizes.append(n)
        state=fedavg(states,sizes);server.load_state_dict(state)
        if r==1 or r%10==0 or r==args.rounds:
            acc=evaluate(server,test,args);best=max(best,acc)
            logger.info("%s Round %d/%d | Accuracy=%.4f | Best=%.4f",name,r,args.rounds,acc,best)
        if r%args.save_every==0:
            torch.save(state,OUTDIR/f"cifar10_{name.lower()}_seed{args.seed}_round{r}.pth")
    torch.save(state,OUTDIR/f"cifar10_{name.lower()}_seed{args.seed}_final.pth")
    logger.info("%s FINAL Accuracy=%.4f | Best=%.4f",name,evaluate(server,test,args),best)

def run_fedbn(client_sets,test,args,logger):
    model_fn=lambda:build_model(args);clients=[FedBNClient(ds,model_fn,args) for ds in client_sets]
    server=model_fn();state=copy.deepcopy(server.state_dict());rng=np.random.RandomState(args.seed);best=0
    for r in range(1,args.rounds+1):
        selected=rng.choice(args.num_clients,args.online,replace=False);states=[];sizes=[]
        for cid in selected:
            st,n=clients[int(cid)].train(copy.deepcopy(state),args);states.append(st);sizes.append(n)
        state=fedbn_aggregate(states,sizes);server.load_state_dict(state)
        if r==1 or r%10==0 or r==args.rounds:
            acc=evaluate(server,test,args);best=max(best,acc)
            logger.info("FedBN Round %d/%d | Accuracy=%.4f | Best=%.4f",r,args.rounds,acc,best)
    torch.save(state,OUTDIR/f"cifar10_fedbn_seed{args.seed}_final.pth")
    logger.info("FedBN FINAL Accuracy=%.4f | Best=%.4f",evaluate(server,test,args),best)

def run_fedic(client_sets,test,teaching,args,logger):
    model_fn=lambda:build_model(args);server=FEDICServer(args,model_fn)
    clients=[StandardClient(ds,model_fn,args) for ds in client_sets]
    state=copy.deepcopy(server.model.state_dict());rng=np.random.RandomState(args.seed);best=0
    for r in range(1,args.rounds+1):
        selected=rng.choice(args.num_clients,args.online,replace=False);states=[];sizes=[]
        for cid in selected:
            st,n=clients[int(cid)].train(copy.deepcopy(state),args);states.append(st);sizes.append(n)
        state=server.aggregate(states,sizes)
        state=server.distill(state,states,teaching,model_fn,args)
        server.model.load_state_dict(state)
        if r==1 or r%10==0 or r==args.rounds:
            acc=evaluate(server.model,test,args);best=max(best,acc)
            logger.info("FEDIC Round %d/%d | Accuracy=%.4f | Best=%.4f",r,args.rounds,acc,best)
    torch.save(state,OUTDIR/f"cifar10_fedic_seed{args.seed}_final.pth")
    logger.info("FEDIC FINAL Accuracy=%.4f | Best=%.4f",evaluate(server.model,test,args),best)

def run_creff(client_sets,test,args,logger):
    model_fn=lambda:build_model(args);clients=[CReFFClient(ds,model_fn,args) for ds in client_sets]
    server=CReFFServer(args,model_fn);global_model=model_fn();state=copy.deepcopy(global_model.state_dict())
    rng=np.random.RandomState(args.seed);best=0
    for r in range(1,args.rounds+1):
        selected=rng.choice(args.num_clients,args.online,replace=False);states=[];sizes=[];grads=[]
        for cid in selected:
            cid=int(cid)
            grads.append(clients[cid].classifier_gradients(state,args))
            st,n=clients[cid].train(copy.deepcopy(state),args);states.append(st);sizes.append(n)
        state=fedavg(states,sizes)
        state=server.update_and_retrain(state,grads,args)
        global_model.load_state_dict(state)
        if r==1 or r%10==0 or r==args.rounds:
            acc=evaluate(global_model,test,args);best=max(best,acc)
            logger.info("CReFF Round %d/%d | Accuracy=%.4f | Best=%.4f",r,args.rounds,acc,best)
    torch.save(state,OUTDIR/f"cifar10_creff_seed{args.seed}_final.pth")
    logger.info("CReFF FINAL Accuracy=%.4f | Best=%.4f",evaluate(global_model,test,args),best)

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--imb_factor",type=float,default=0.01)
    p.add_argument("--alpha",type=float,default=0.5)
    p.add_argument("--num_clients",type=int,default=20)
    p.add_argument("--online",type=int,default=8)
    p.add_argument("--rounds",type=int,default=200)
    p.add_argument("--local_epochs",type=int,default=5)
    p.add_argument("--batch_size",type=int,default=32)
    p.add_argument("--lr",type=float,default=0.1)
    p.add_argument("--mu",type=float,default=0.01)
    p.add_argument("--test_batch",type=int,default=128)
    p.add_argument("--save_every",type=int,default=50)
    p.add_argument("--num_classes",type=int,default=10)
    p.add_argument("--feature_dim",type=int,default=256)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data_root",default=str(ROOT/"data"/"CIFAR10"))
    p.add_argument("--methods",default="FedAvg,FedProx,FedBN,FedRS,FEDIC,CReFF")
    p.add_argument("--fedic_teach_ratio",type=float,default=0.10)
    p.add_argument("--fedic_teach_min",type=int,default=256)
    p.add_argument("--fedic_lr",type=float,default=1e-3)
    p.add_argument("--fedic_student_lr",type=float,default=0.01)
    p.add_argument("--fedic_batch",type=int,default=32)
    p.add_argument("--fedic_server_steps",type=int,default=1)
    p.add_argument("--fedic_distill_steps",type=int,default=1)
    p.add_argument("--fedic_T",type=float,default=3.0)
    p.add_argument("--fedic_lambda",type=float,default=0.5)
    p.add_argument("--creff_ipc",type=int,default=10)
    p.add_argument("--creff_real_batch",type=int,default=32)
    p.add_argument("--creff_lr_feature",type=float,default=0.1)
    p.add_argument("--creff_lr_net",type=float,default=0.1)
    p.add_argument("--creff_match_epoch",type=int,default=20)
    p.add_argument("--creff_crt_epoch",type=int,default=20)
    return p.parse_args()

def main():
    args=parse();seed_all(args.seed);logger=setup_logger(args.seed)
    logger.info("="*70);logger.info("CIFAR-10-LT BASELINE SUITE")
    logger.info("IF=%.0f | alpha=%.3f | clients=%d | online=%d | rounds=%d | local_epochs=%d | batch=%d | lr=%.3f | seed=%d",
                1/args.imb_factor,args.alpha,args.num_clients,args.online,args.rounds,args.local_epochs,args.batch_size,args.lr,args.seed)
    logger.info("Device: %s",args.device);logger.info("CIFAR-10 path: %s",args.data_root)
    client_sets,test,teaching=make_data(args,logger)
    for name in [x.strip() for x in args.methods.split(",") if x.strip()]:
        logger.info("="*70);logger.info("START %s",name)
        if name=="FedAvg":run_standard(name,StandardClient,client_sets,test,args,logger)
        elif name=="FedProx":run_standard(name,FedProxClient,client_sets,test,args,logger)
        elif name=="FedBN":run_fedbn(client_sets,test,args,logger)
        elif name=="FedRS":run_standard(name,FedRSClient,client_sets,test,args,logger)
        elif name=="FEDIC":run_fedic(client_sets,test,teaching,args,logger)
        elif name=="CReFF":run_creff(client_sets,test,args,logger)
        else: logger.warning("Unknown method skipped: %s",name)
    logger.info("="*70);logger.info("ALL BASELINES FINISHED")

if __name__=="__main__":main()
