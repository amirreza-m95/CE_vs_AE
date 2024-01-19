import numpy as np
from scipy.sparse import csr_matrix
import pickle
import torch

from CE.gnn_cff.scripts.exp_node_ba_shapes import mainCF2_shapesXp
from AE.main import targetedNettack

from CE.gnn_cff.utils.utils import draw_subgraph_around_node, draw_subgraph_with_3hop_neighbors, adj_matrix2dgl


def compare_sparse_matrices(matrix1, matrix2):
    differences = (matrix1 != matrix2).tocoo()
    
    if differences.nnz == 0:
        print("Matrices are identical.")
    else:
        print("Differences found at the following positions:")
        for i, j, _ in zip(differences.row, differences.col, differences.data):
            print(f"Position ({i}, {j}): Matrix1={matrix1[i, j]}, Matrix2={matrix2[i, j]}")



def main():
    #Nettack on targeted Node 543
    adj, modified_adj = targetedNettack(targetNode=23, num_perturbation=10)

    mainCF2_shapesXp()



if __name__ == "__main__":
    main()