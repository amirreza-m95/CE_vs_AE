import os
import numpy as np
import torch
from utils.argument import arg_parse_exp_node_tree_cycles
from models.explainer_models import NodeExplainerEdgeMulti
from models.gcn import GCNNodeTreeCycles
from utils.preprocessing.tree_cycles_preprocessing import TreeCyclesDataset
import sys


if __name__ == "__main__":
    torch.manual_seed(1000)
    np.random.seed(0)
    np.set_printoptions(threshold=sys.maxsize)
    exp_args = arg_parse_exp_node_tree_cycles()
    print("argument:\n", exp_args)
    model_path = exp_args.model_path
    train_indices = np.load(os.path.join(model_path, 'train_indices.pickle'), allow_pickle=True)
    test_indices = np.load(os.path.join(model_path, 'test_indices.pickle'), allow_pickle=True)
    G_dataset = TreeCyclesDataset(load_path=os.path.join(model_path))
    # targets = np.load(os.path.join(model_path, 'targets.pickle'), allow_pickle=True)  # the target node to explain
    graphs = G_dataset.graphs
    labels = G_dataset.labels
    targets = G_dataset.targets
    if exp_args.gpu:
        device = torch.device('cuda:%s' % exp_args.cuda)
    else:
        device = 'cpu'
    base_model = GCNNodeTreeCycles(G_dataset.feat_dim, 32, num_classes=2, if_exp=True).to('cpu')# change to cpu
    base_model.load_state_dict(torch.load(os.path.join(model_path, 'model.model'), map_location=torch.device('cpu'))) #added map_location part
    #  fix the base model
    for param in base_model.parameters():
        param.requires_grad = False

    # Create explainer
    explainer = NodeExplainerEdgeMulti(
        base_model=base_model,
        G_dataset=G_dataset,
        args=exp_args,
        test_indices=test_indices,
        # fix_exp=6
    )

    explainer.explain_nodes_gnn_stats()
