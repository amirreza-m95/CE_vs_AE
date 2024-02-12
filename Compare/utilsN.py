import torch
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp

def draw_graph_from_tensorAdj(adjacency_matrix):
    """
    Draw a graph using networkx based on the given adjacency matrix in torch.tensor format.

    Parameters:
    - adjacency_matrix (torch.Tensor): The adjacency matrix in torch.tensor format.

    Returns:
    - None
    """
    # Convert torch.tensor to a NumPy array
    adjacency_matrix = adjacency_matrix.numpy()

    # Create a graph from the adjacency matrix
    graph = nx.DiGraph(adjacency_matrix)

    # Draw the graph
    pos = nx.spring_layout(graph)  # You can choose a different layout if needed
    nx.draw(graph, pos, with_labels=True, font_weight='bold')
    plt.show()

# Example usage:
# Create a random adjacency matrix for testing
# adjacency_matrix = torch.randint(2, size=(5, 5), dtype=torch.float32)
# draw_graph_from_adjacency_matrix(adjacency_matrix)

# whole graph
#input: dgl.heterograph
def draw_graph(graph):
    # Convert dgl.heterograph to networkx graph
    nx_graph = graph.to_networkx().to_undirected()

    # Draw the subgraph
    pos = nx.spring_layout(nx_graph)  # You can use different layout algorithms
    nx.draw(nx_graph, pos, with_labels=True, font_weight='bold', node_color='skyblue', node_size=800)
    plt.title(f"Graph")
    plt.show()


#input: adjacency matrix
def draw_subgraph_adj_3hop(adjacency_matrix, node_number, model_type):
    if model_type == 'net':
      graph = nx.DiGraph(adjacency_matrix)
    else:
      # Convert the adjacency matrix to a networkx graph
      edges = torch.nonzero(adjacency_matrix, as_tuple=False).tolist()
      graph = nx.DiGraph(edges)

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

# Example usage:
# Replace the following with your actual adjacency matrix and node number
# Example adjacency matrix for a small graph
# example_adjacency_matrix = sp.csr_matrix([[0, 1, 1, 0, 0],
#                                           [1, 0, 1, 1, 0],
#                                           [1, 1, 0, 1, 1],
#                                           [0, 1, 1, 0, 1],
#                                           [0, 0, 1, 1, 0]])

# # Node number to focus on
# example_node_number = 2

# Draw the subgraph with 3-hop neighbors
# draw_subgraph_with_3hop_neighbors(example_adjacency_matrix, example_node_number)
    
# input dgl.heterograph
def draw_subgraph_dgl(graph, target_node):
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




def draw_subgraph_adj(adjacency_matrix, node_number, model_type):
    if model_type == 'net':
      graph = nx.DiGraph(adjacency_matrix)
    else:
      # Convert the adjacency matrix to a networkx graph
      edges = torch.nonzero(adjacency_matrix, as_tuple=False).tolist()
      graph = nx.DiGraph(edges)

    # Extract the neighbors of the specified node
    neighbors = list(graph.neighbors(node_number))

    # Create a subgraph with the specified node and its neighbors
    subgraph = graph.subgraph([node_number] + neighbors)


    # Define colors for different classes
    class_colors = {'A': 'red', 'B': 'blue'}

    # Draw the subgraph using matplotlib
    pos = nx.spring_layout(subgraph)  # You can use different layout algorithms
    nx.draw(subgraph, pos, with_labels=True, font_weight='bold', node_color='skyblue', font_color='black', node_size=500)

    plt.title(f'Subgraph around Node {node_number}')
    plt.show()

# Example usage:
# Replace the following with your own sparse matrix and node number
# Example adjacency matrix for a small graph
# # example_adjacency_matrix = sp.csr_matrix([[0, 1, 1, 0, 0],
#                                           [1, 0, 1, 1, 0],
#                                           [1, 1, 0, 1, 1],
#                                           [0, 1, 1, 0, 1],
#                                           [0, 0, 1, 1, 0]])

# # Node number to focus on
# example_node_number = 2

# Draw the subgraph
# draw_subgraph_adj(example_adjacency_matrix, example_node_number)





#util
def flat_tensor2adj_matrix(flat_tensor, num_nodes):
    adjacency_matrix = flat_tensor.view(num_nodes, num_nodes)
    return adjacency_matrix

def adj_matrix2dgl(adjacency_matrix):
    # Convert the adjacency matrix to a DGL graph

    # Convert the adjacency matrix to a CSR matrix
    csr_matrix_representation = csr_matrix(adjacency_matrix)

    g = dgl.DGLGraph(csr_matrix_representation)

    # You can add node and edge features if needed
    # For example, adding a node feature 'feat' with random values
    g.ndata['feat'] = torch.rand(g.num_nodes())

    return g

def get_masked_graph(graph):
    return adj_matrix2dgl(flat_tensor2adj_matrix(graph.edata['weight'],graph.num_nodes()))