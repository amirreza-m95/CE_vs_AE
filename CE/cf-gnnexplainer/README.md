# CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks



## Requirements
Torch:
- [ ] pip install --no-cache-dir torch==1.13.1+cu117  --extra-index-url https://download.pytorch.org/whl/cu117

torch-geometric:
- [ ] pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
- [ ] pip install --no-cache-dir torch_geometric==2.2
- [ ] pip install pandas


## Improvements
- Accuracy function in train.py was deprecated. Added the similar function(line 17).

