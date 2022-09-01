# anomoly_detection_cogdl
CogDl based GNN model for anomoly detection on Amazon &amp; YelpChi datasets

This repo provides a collection of cogdl baselines for YelpChi & Amazon dataset which are put under ./dataset/

## Environments
Implementing environment:  
- numpy = 1.21.2  
- pytorch >= 1.6.0  
- pillow = 9.1.1
- cogdl = 0.5.3

## Training
(Before training you should unzip YelpChi.zip & Amazon.zip under ./dataset/)

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
| SIGN | 0.7232 ± 0.0031 | 0.8543 ± 0.0042 | | SIGN | 0.9179 ± 0.0012 | 0.9687 ± 0.0011 |
| SAGN | 0.7114 ± 0.0039 | 0.8431 ± 0.0035 | | SAGN | 0.9020 ± 0.0041 | 0.9613 ± 0.0012 |
| GraphSAGE| 0.6620 ± 0.0022 | 0.7903 ± 0.0021 | | GraphSAGE| 0.8213 ± 0.0042 | 0.8759 ± 0.0030 |
| GCN | 0.6012 ± 0.0011 | 0.6886 ± 0.0047 | | GCN | 0.7674 ± 0.0011 | 0.8629 ± 0.0047 |
| GIN | 0.5965 ± 0.0257 | 0.6883 ± 0.0320 | | GIN | 0.7565 ± 0.0257 | 0.8583 ± 0.0320 |
| MLP | 0.9035 ± 0.0033 | 0.7587 ± 0.0031 | | MLP | 0.9035 ± 0.0033 | 0.7587 ± 0.0031 |
| SGC | 0.5219 ± 0.0046 | 0.5853 ± 0.0043 | | SGC | 0.7148 ± 0.0033 | 0.8652 ± 0.0052 |
|  Latest work   |  |   |  |    |   |   |
| BWGNN(ICLM22') | 0.7696 ± 0.0016 | 0.9054 ± 0.0021  | |  | 0.9229 ± 0.0023 | 0.9806 ± 0.0016 |
| H2-Fdetector(WWW22') | 0.6944 ± 0.0031 | 0.8869 ± 0.0039 | |  | 0.8320 ± 0.0023 | 0.9689 ± 0.0022 |

