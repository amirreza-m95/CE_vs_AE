from scipy.sparse import csr_matrix
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import torch
import pickle

import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np



def flatten_adjacency_matrix(matrix):
    # Flatten the upper triangular part of the matrix (excluding the diagonal)
    flattened_weights = matrix[np.triu(np.ones_like(matrix), k=1) == 1]
    return flattened_weights

def unflatten2adjacency_matrix(flattened_weights, n):
    # Create an empty matrix with zeros
    matrix = torch.zeros((n, n))
    
    # Fill the upper triangular part of the matrix (excluding the diagonal) with flattened weights
    matrix[torch.triu_indices(n, 1)] = flattened_weights
    
    # Symmetrize the matrix
    matrix = matrix + matrix.T
    
    return matrix

def adj_matrix2dgl(adjacency_matrix):
    # Convert the adjacency matrix to a DGL graph

    # Convert the adjacency matrix to a CSR matrix
    csr_matrix_representation = csr_matrix(adjacency_matrix)

    g = dgl.DGLGraph(csr_matrix_representation)

    # You can add node and edge features if needed
    # For example, adding a node feature 'feat' with random values
    g.ndata['feat'] = torch.rand(g.num_nodes())

    return g

def flat_tensor2adj_matrix(flat_tensor, num_nodes):
    adjacency_matrix = flat_tensor.view(num_nodes, num_nodes)
    return adjacency_matrix

def draw_subgraph_with_3hop_neighbors(adjacency_matrix, node_number):
    # Convert the adjacency matrix to a networkx graph
    graph = nx.DiGraph(adjacency_matrix)

    # Extract the neighbors of the specified node
    neighbors = set(graph.neighbors(node_number))

    # Extract neighbors of neighbors (2-hop neighbors)
    neighbors_of_neighbors = set()
    for neighbor in neighbors:
        neighbors_of_neighbors.update(graph.neighbors(neighbor))

    # Extract neighbors of neighbors of neighbors (3-hop neighbors)
    neighbors_of_3hop_neighbors = set()
    for neighbor in neighbors_of_neighbors:
        neighbors_of_3hop_neighbors.update(graph.neighbors(neighbor))

    # Remove the specified node, its neighbors, and neighbors of neighbors from the set
    neighbors_of_3hop_neighbors.difference_update(neighbors_of_neighbors)
    neighbors_of_3hop_neighbors.discard(node_number)

    # Create a subgraph with the specified node, its neighbors, neighbors of neighbors, and 3-hop neighbors
    subgraph_nodes = [node_number] + list(neighbors) + list(neighbors_of_neighbors) + list(neighbors_of_3hop_neighbors)
    subgraph = graph.subgraph(subgraph_nodes)

    # Draw the subgraph using matplotlib
    pos = nx.spring_layout(subgraph)  # You can use different layout algorithms
    nx.draw(subgraph, pos, with_labels=True, font_weight='bold', node_color='skyblue', font_color='black', node_size=700)

    plt.title(f'Subgraph with 3-hop Neighbors around Node {node_number}')
    plt.show()

def draw_subgraph_around_node(graph, target_node):
    # Convert dgl.heterograph to networkx graph
    nx_graph = graph.to_networkx().to_undirected()

    # Get neighbors of the target node
    neighbors = list(nx_graph.neighbors(target_node))

    # Add the target node and its neighbors to the subgraph
    subgraph_nodes = [target_node] + neighbors
    subgraph = nx_graph.subgraph(subgraph_nodes)

    # Draw the subgraph
    pos = nx.spring_layout(subgraph)  # You can use different layout algorithms
    nx.draw(subgraph, pos, with_labels=True, font_weight='bold', node_color='skyblue', node_size=800)
    plt.title(f"Subgraph around Node {target_node}")
    plt.show()

def draw_graph(graph):
    # Convert dgl.heterograph to networkx graph
    nx_graph = graph.to_networkx().to_undirected()

    # Draw the subgraph
    pos = nx.spring_layout(nx_graph)  # You can use different layout algorithms
    nx.draw(nx_graph, pos, with_labels=True, font_weight='bold', node_color='skyblue', node_size=800)
    plt.title(f"Graph")
    plt.show()

def get_masked_graph(graph):
    return adj_matrix2dgl(flat_tensor2adj_matrix(graph.edata['weight'],graph.num_nodes()))

def draw_graph_from_tensorAdj(adjacency_matrix):
    """
    Draw a graph using networkx based on the given adjacency matrix in torch.tensor format.

    Parameters:
    - adjacency_matrix (torch.Tensor): The adjacency matrix in torch.tensor format.

    Returns:
    - None
    """
    # Convert torch.tensor to a NumPy array
    adjacency_matrix = adjacency_matrix.detach().numpy()

    # Create a graph from the adjacency matrix
    graph = nx.DiGraph(adjacency_matrix)

    # Draw the graph
    pos = nx.spring_layout(graph)  # You can choose a different layout if needed
    nx.draw(graph, pos, with_labels=True, font_weight='bold')
    plt.show()

def save_graph(graph, file_path):
    # Save the object to a file
    with open(file_path + '.pkl', 'wb') as file:
        pickle.dump(graph, file)

def top_two_max_indices(tensor_array):
    # Convert the input list to a NumPy array
    tensor_array = np.array(tensor_array)

    # Get the indices of the top two maximum values
    top_two_indices = np.argsort(tensor_array)[-2:][::-1]

    return top_two_indices

def graph_export(graphs, exp_dict, graphNo):
    num_nodes = graphs.G_dataset.graphs[graphNo].num_nodes()
    save_graph(graphs.G_dataset.graphs[graphNo], f'g{graphNo}_n{num_nodes}_t{graphs.G_dataset.targets[graphNo]}')
    draw_graph_from_tensorAdj(exp_dict[graphNo].view(num_nodes,num_nodes))
    exp_dict[graphNo][exp_dict[graphNo] < 0.6] = 0
    draw_graph_from_tensorAdj(exp_dict[graphNo].view(num_nodes,num_nodes))
    topTwoVal = float(exp_dict[graphNo][(top_two_max_indices(exp_dict[graphNo].detach())[1])])
    exp_dict[graphNo][exp_dict[graphNo] < topTwoVal] = 0
    draw_graph_from_tensorAdj(exp_dict[graphNo].view(num_nodes,num_nodes))
    

# Export Graph
# save_graph(self.G_dataset.graphs[684], 'g684_n18_t17')

# exp_dict[684][exp_dict[684] < 0.7] = 0

