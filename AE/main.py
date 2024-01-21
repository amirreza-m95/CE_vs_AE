import sys
sys.path.append('AE/')

import numpy as np
from scipy.sparse import csr_matrix
import pickle
import torch

from attacks import targetedNettack, RLS2Vattack, FGAattack

def data_loader(filename):
  with open(filename, 'rb') as file:
    graphBA = pickle.load(file)
  return graphBA


# sbatch --job-name=CEvsAE --mail-user=amir.reze@uibk.ac.at --time=10:00:00 --mem=120G /home/amir.reza/jobs/single-node-gpu.job "python "

def main():
    targetNode = 543
    # modified_adj_nettack = targetedNettack(targetNode=543, num_perturbation=2)

    RLS2Vattack()
    # FGAattack()
    




if __name__ == "__main__":
    main()