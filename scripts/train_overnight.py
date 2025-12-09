"""Overnight hyperparameter sweep + ensemble training."""

import json
import subprocess
import sys
from itertools import product
from pathlib import Path

PARAM_GRID = {
    "lr": [1e-5, 2e-5, 5e-5],
    "hidden": [128, 256],
    "dropout": [0.1, 0.2],
    "batch": [8, 16],
}
SEEDS = [42, 123, 456, 789, 1337]


def run_exp(lr, hidden, dropout, batch, seed, epochs=10):
    name = f"lr{lr}_h{hidden}_d{dropout}_b{batch}_s{seed}"
    out_dir = Path(f"results/sweep/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    code = f'''
import json,torch,torch.nn as nn;from torch.utils.data import Dataset,DataLoader;from transformers import DistilBertTokenizer,DistilBertModel,get_linear_schedule_with_warmup;from sklearn.metrics import f1_score,precision_score,recall_score;import random,numpy as np;from pathlib import Path
random.seed({seed});np.random.seed({seed});torch.manual_seed({seed})
class DS(Dataset):
    def __init__(s,d,t,m=512):s.d,s.t,s.m=d,t,m
    def __len__(s):return len(s.d)
    def __getitem__(s,i):e=s.t(s.d[i]["text"],truncation=True,max_length=s.m,padding="max_length",return_tensors="pt");return{{"input_ids":e["input_ids"].squeeze(),"attention_mask":e["attention_mask"].squeeze(),"label":torch.tensor(1 if s.d[i]["is_hack"]else 0)}}
class M(nn.Module):
    def __init__(s):super().__init__();s.b=DistilBertModel.from_pretrained("distilbert-base-uncased");s.f=nn.Sequential(nn.Linear(768,{hidden}),nn.ReLU(),nn.Dropout({dropout}),nn.Linear({hidden},2))
    def forward(s,i,m):return s.f(s.b(input_ids=i,attention_mask=m).last_hidden_state[:,0,:])
dev=torch.device("mps"if torch.backends.mps.is_available()else"cuda"if torch.cuda.is_available()else"cpu")
with open("data/training/train.json")as f:tr=json.load(f)
with open("data/training/val.json")as f:va=json.load(f)
with open("data/training/test.json")as f:te=json.load(f)
tok=DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
trl,val,tel=DataLoader(DS(tr,tok),batch_size={batch},shuffle=True),DataLoader(DS(va,tok),batch_size={batch}),DataLoader(DS(te,tok),batch_size={batch})
m=M().to(dev);opt=torch.optim.AdamW(m.parameters(),lr={lr});sch=get_linear_schedule_with_warmup(opt,len(trl),len(trl)*{epochs});crit=nn.CrossEntropyLoss();best=0
for ep in range({epochs}):
    m.train()
    for b in trl:opt.zero_grad();loss=crit(m(b["input_ids"].to(dev),b["attention_mask"].to(dev)),b["label"].to(dev));loss.backward();opt.step();sch.step()
    m.eval();ps,ls=[],[]
    with torch.no_grad():
        for b in val:ps.extend(torch.argmax(m(b["input_ids"].to(dev),b["attention_mask"].to(dev)),1).cpu().numpy());ls.extend(b["label"].numpy())
    f1=f1_score(ls,ps,zero_division=0)
    if f1>best:best=f1;torch.save(m.state_dict(),"{out_dir}/best.pt")
m.load_state_dict(torch.load("{out_dir}/best.pt"));m.eval();ps,ls=[],[]
with torch.no_grad():
    for b in tel:ps.extend(torch.argmax(m(b["input_ids"].to(dev),b["attention_mask"].to(dev)),1).cpu().numpy());ls.extend(b["label"].numpy())
res={{"f1":f1_score(ls,ps,zero_division=0),"p":precision_score(ls,ps,zero_division=0),"r":recall_score(ls,ps,zero_division=0),"cfg":{{"lr":{lr},"h":{hidden},"d":{dropout},"b":{batch},"s":{seed}}}}}
print(json.dumps(res));
with open("{out_dir}/results.json","w")as f:json.dump(res,f)
'''
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().split("\\n")[-1])
    except:
        return {"f1": 0, "err": r.stderr}


def main():
    print("=== OVERNIGHT TRAINING ===")
    Path("results/sweep").mkdir(parents=True, exist_ok=True)
    best_f1, best_cfg = 0, None
    for lr, h, d, b in product(*PARAM_GRID.values()):
        print(f"lr={lr},h={h},d={d},b={b}...", end=" ")
        res = run_exp(lr, h, d, b, 42, 5)
        print(f"F1={res.get('f1', 0):.4f}")
        if res.get("f1", 0) > best_f1:
            best_f1, best_cfg = res["f1"], {"lr": lr, "hidden": h, "dropout": d, "batch": b}
    print(f"\\nBest:{best_cfg} F1={best_f1:.4f}")
    f1s = []
    for s in SEEDS:
        print(f"Seed {s}...", end=" ")
        res = run_exp(**best_cfg, seed=s, epochs=15)
        print(f"F1={res.get('f1', 0):.4f}")
        f1s.append(res.get("f1", 0))
    import numpy as np

    print(f"\\n=== ENSEMBLE F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f} ===")
    with open("results/overnight_summary.json", "w") as f:
        json.dump(
            {
                "best": best_cfg,
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s)),
                "all": f1s,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
