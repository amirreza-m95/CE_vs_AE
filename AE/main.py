import numpy as np
from scipy.sparse import csr_matrix
import pickle
import torch

from attacks import targetedNettack, RLS2Vattack, FGAattack

def data_loader(filename):
  with open(filename, 'rb') as file:
    graphBA = pickle.load(file)
  return graphBA




def main():
    with open(f'AE/dataset/syn_data.pkl', 'rb') as file:
        data = pickle.load(file)

    adj = torch.tensor(data[0], dtype=torch.float32)
    features = torch.tensor(data[1], dtype=torch.float32)
    y_train = data[2]
    y_val = data[3]
    y_test = data[4]
    idx_train, idx_val, idx_test = data[5], data[6], data[7]
    edge_labels = torch.tensor(data[8])
    labels_oneHot = np.array(np.logical_or(y_train, np.logical_or(y_val, y_test)), dtype=int)
    labels = torch.tensor(np.argmax(labels_oneHot, axis=1))
    
    targetNode = 543
    # modified_adj_nettack = targetedNettack(targetNode, features, labels, adj, idx_test, idx_train, idx_val)

    # RLS2Vattack()
    FGAattack()
    




if __name__ == "__main__":
    main()