import numpy as np
from scipy.sparse import csr_matrix
import pickle
import torch

from AE.attacks import targetedNettack, RLS2Vattack, FGAattack

def data_loader(filename):
  with open(filename, 'rb') as file:
    graphBA = pickle.load(file)
  return graphBA




def main():
    targetNode = 543
    modified_adj_nettack = targetedNettack(targetNode)

    # RLS2Vattack()
    FGAattack()
    




if __name__ == "__main__":
    main()