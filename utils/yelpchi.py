import torch
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split

def YelpChi_Dataloader(datapath):
    # Load data
    print('read_YelpChi')
    folder = datapath

    print('Convert to DGL Graph.')
    yelp_path = datapath
    yelp = scio.loadmat(yelp_path)
    feats = yelp['features'].todense()
    features = feats
    lbs = yelp['label'][0]
    labels = torch.from_numpy(lbs)
    homo = yelp['homo']
    homo = homo + homo.transpose()
    homo = homo.tocoo()

    # Create cogdl graph
    x = torch.tensor(features, dtype=torch.float).contiguous()
    y = labels

    edge_index = torch.tensor(np.array([homo.row, homo.col]), dtype=torch.int64).contiguous()

    print('Generate dataset partition.')
    train_ratio = 0.4
    test_ratio = 0.67
    index = list(range(len(lbs)))
    dataset_l = len(lbs)
    train_idx, rest_idx, train_lbs, rest_lbs = train_test_split(index, lbs, stratify=lbs, train_size=train_ratio,
                                                                random_state=2, shuffle=True)
    valid_idx, test_idx, _, _ = train_test_split(rest_idx, rest_lbs, stratify=rest_lbs, test_size=test_ratio,
                                                 random_state=2, shuffle=True)
    train_mask = torch.zeros(dataset_l, dtype=torch.bool)
    train_mask[np.array(train_idx)] = True
    valid_mask = torch.zeros(dataset_l, dtype=torch.bool)
    valid_mask[np.array(valid_idx)] = True
    test_mask = torch.zeros(dataset_l, dtype=torch.bool)
    test_mask[np.array(test_idx)] = True
    # edge_type = torch.tensor(edge_type, dtype=torch.float)

    return x, edge_index, y, train_mask, valid_mask, test_mask