import argparse, csv, logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Model.Resnet8 import ResNet_cifar

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
LOG = ROOT / "Logs"

def build_model(device):
    return ResNet_cifar(
        resnet_size=8, scaling=4, save_activations=False,
        group_norm_num_groups=None, freeze_bn=False,
        freeze_bn_affine=False, num_classes=10
    ).to(device)

def load_state(path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    if not isinstance(obj, dict):
        raise ValueError("Unsupported checkpoint: " + str(path))
    return {k[7:] if k.startswith("module.") else k: v for k,v in obj.items()}

def evaluate(model, test, device):
    model.eval()
    loader = DataLoader(test, batch_size=128, shuffle=False, num_workers=0)
    correct = np.zeros(10, dtype=np.int64)
    total = np.zeros(10, dtype=np.int64)
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            _,z=model(x)
            p=z.argmax(1)
            for c in range(10):
                m=(y==c)
                total[c]+=int(m.sum())
                correct[c]+=int((p[m]==y[m]).sum())
    pc=correct/np.maximum(total,1)
    groups={"Head":[0,1,2],"Middle":[3,4,5,6],"Tail":[7,8,9]}
    r={}
    for name,cls in groups.items():
        r[name]=correct[cls].sum()/total[cls].sum()
    r["All"]=correct.sum()/total.sum()
    r["per_class"]=pc
    return r,correct,total

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--data_root",default=str(ROOT/"data"/"CIFAR10"))
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args()
    LOG.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    logpath=LOG/f"cifar10_baselines_seed{args.seed}_metrics.log"
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(logpath,"w",encoding="utf-8"),
                  logging.StreamHandler()])
    logger=logging.getLogger("eval")
    logger.info("="*70)
    logger.info("CIFAR-10-LT saved-checkpoint evaluation")
    logger.info("No training is performed.")
    logger.info("Device: %s",args.device)
    logger.info("Head = classes 0,1,2 | Middle = 3,4,5,6 | Tail = 7,8,9")

    tf=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2023,0.1994,0.2010))])
    test=datasets.CIFAR10(args.data_root,train=False,download=False,transform=tf)

    methods=[
        ("FedAvg",OUT/f"cifar10_fedavg_seed{args.seed}_final.pth"),
        ("FedProx",OUT/f"cifar10_fedprox_seed{args.seed}_final.pth"),
        ("FedBN",OUT/f"cifar10_fedbn_seed{args.seed}_final.pth"),
        ("FedRS",OUT/f"cifar10_fedrs_seed{args.seed}_final.pth"),
        ("FEDIC",OUT/f"cifar10_fedic_seed{args.seed}_final.pth"),
        ("CReFF",OUT/f"cifar10_creff_seed{args.seed}_final.pth")]
    summary=[]
    classes=[]
    for name,path in methods:
        if not path.exists():
            logger.warning("Missing checkpoint: %s",path); continue
        model=build_model(args.device)
        state=load_state(path,args.device)
        missing,unexpected=model.load_state_dict(state,strict=False)
        if missing: logger.warning("%s missing keys: %s",name,list(missing)[:10])
        if unexpected: logger.warning("%s unexpected keys: %s",name,list(unexpected)[:10])
        r,correct,total=evaluate(model,test,args.device)
        logger.info("%s | Head=%.2f%% | Middle=%.2f%% | Tail=%.2f%% | All=%.2f%%",
                    name,100*r["Head"],100*r["Middle"],100*r["Tail"],100*r["All"])
        summary.append([name,r["Head"],r["Middle"],r["Tail"],r["All"]])
        for c in range(10):
            classes.append([name,c,int(correct[c]),int(total[c]),float(r["per_class"][c])])
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    scsv=OUT/f"cifar10_baselines_seed{args.seed}_head_middle_tail.csv"
    with open(scsv,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f);w.writerow(["Method","Head","Middle","Tail","All"])
        w.writerows(summary)
    ccsv=OUT/f"cifar10_baselines_seed{args.seed}_per_class.csv"
    with open(ccsv,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f);w.writerow(["Method","Class","Correct","Total","Accuracy"])
        w.writerows(classes)
    logger.info("Summary CSV: %s",scsv)
    logger.info("Per-class CSV: %s",ccsv)
    logger.info("Metrics log: %s",logpath)

if __name__=="__main__": main()
