# anomoly_detection_cogdl
CogDl based GNN model for anomoly detection on Amazon &amp; YelpChi datasets

This repo provides a collection of cogdl baselines for YelpChi & Amazon dataset which are put under ./dataset/

## Environments
Implementing environment:  
- numpy = 1.21.2  
- pytorch >= 1.6.0  
- pillow = 9.1.1
- cogdl = 0.5.3

## Training(Before training you should unzip YelpChi.zip & Amazon.zip under ./dataset/)

- **MLP**
```bash
python gnn.py --model mlp --dataset Amazon --epochs 100 --runs 10 --device 0
```
```bash
python gnn.py --model mlp --dataset YelpChi --epochs 100 --runs 10 --device 0
```

- **GCN**
```bash
python gnn.py --model gcn --dataset YelpChi --epochs 100 --runs 10 --device 0
```

- **GraphSAGE**
```bash
python gnn.py --model graphsage --dataset YelpChi --epochs 100 --runs 10 --device 0
```

- **GIN**
```bash
python gnn.py --model gin --dataset YelpChi --epochs 100 --runs 10 --device 0
```

- **GAT**
```bash
python gnn.py --model gat --dataset YelpChi --epochs 100 --runs 10 --device 0
```

- **Grand**
```bash
python gnn.py --model grand --dataset YelpChi --epochs 100 --runs 10 --device 0
```

- **SGC**
```bash
python gnn.py --model sgc --dataset YelpChi --epochs 100 --runs 10 --device 0
```

- **SIGN**
```bash
python gnn.py --model sign --dataset YelpChi --epochs 100 --runs 10 --device 0
```

- **SAGN**
```bash
python gnn.py --model sagn --dataset YelpChi --epochs 100 --runs 10 --device 0
```

- **You can find more models on cogdl https://cogdl.readthedocs.io/en/latest/index.html**


## Results:
Performance on **YelpChi**(10 runs):

|   | YelpChi(40% Training)  |   |   |    | Amazon(40% Training)  |   |
|  :----  |  ---- |  ---- |  ----|  :----  |  ---- |  ---- |
| Methods   | Test MacroF1  | Test AUC  |   |  Methods | Test MacroF1  | Test AUC  |
|    |    |   |   |    |  |   |
| SIGN | 0.9421 ± 0.0031 | 0.9213 ± 0.0042 | | SIGN | 0.9421 ± 0.0031 | 0.9213 ± 0.0042 |
| SAGN | 0.9421 ± 0.0031 | 0.9213 ± 0.0042 | | SAGN | 0.9421 ± 0.0031 | 0.9213 ± 0.0042 |
| GIN | 0.8965 ± 0.0257 | 0.8983 ± 0.0320 | | GIN | 0.8965 ± 0.0257 | 0.8983 ± 0.0320 |
| GraphSAGE| 0.9213 ± 0.0022 | 0.8986 ± 0.0021 | | GraphSAGE| 0.9213 ± 0.0022 | 0.8986 ± 0.0021 |
| GCN | 0.9374 ± 0.0011 | 0.8629 ± 0.0047 | | GCN | 0.9374 ± 0.0011 | 0.8629 ± 0.0047 |
| MLP | 0.9035 ± 0.0033 | 0.7587 ± 0.0031 | | MLP | 0.9035 ± 0.0033 | 0.7587 ± 0.0031 |
| Grand  | 0.6317 ± 0.0018 | 0.6292 ± 0.0051 | | Grand | 0.6317 ± 0.0018 | 0.6292 ± 0.0051 |
| SGC | 0.6187 ± 0.0046 | 0.6136 ± 0.0043 | | SGC | 0.6187 ± 0.0046 | 0.6136 ± 0.0043 |
