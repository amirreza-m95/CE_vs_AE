# Counterfactual Explanation_vs_Adversarial Examples


## Table of Contents
- [Requirements](#requirements)
- [CFF](#cFF)
- [CF_GNNExplainer](#cF_GNNExplainer)
- [CLEAR](#cLEAR)
- [GCFExplainer](#gCFExplainer)
- [RCExplainer](#rCExplainer)
- [Adversarials](#adversarials)

## Requirements
- Python 3.7
- pytorch 1.9.0
- cuda 10.2
- dgl-cuda10.2 (Edit graphconv.py)
- torch-geometric

  Detailed requirements are available in requirements.txt in root folder.


## CFF
1. To run the experiments, dgl-cuda library is required. https://docs.dgl.ai/install/index.html.
2. After installing dgl-cuda library, Please replace the graphconv.py file in the dgl library with the file (with the same name) provided in the CE/gnn_cff folder. This is for the relaxation purpose as described in the paper.
3. The training and explaining are independent phases. For you convenience, we provide pre-trained GNN model params. Under the CE/gnn_cff folder, run:
    ```
    unzip log.zip
    ```
4. To set the python path, under the project root folder, run the following command:
    ```
    export PYTHONPATH=$PYTHONPATH:"$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/CE/gnn_cff"
    ```
5. As an example, to generate explanations for node classification task (Tree-Cycles dataset as example), run:
    ```
    python CE/gnn_cff/scripts/exp_node_tree_cycles.py
    ```
6. For generating explanations for graph classification task (Mutag0 dataset as example), run:
    ```
    python CE/gnn_cff/scripts/exp_graph.py
    ```
7. The codes for training GNNs are also provided. For instance, for training GNN for node classification, run:
    ```
    python CE/gnn_cff/scripts/train_node_classification.py
    ```
    for graph classification, run:
    ```
    python CE/gnn_cff/scripts/train_graph_classification.py
    ```



## CF-GNNExplainer 
#### Training original models

To train the original GNN models for the BA-shapes dataset in the paper, cd into CE/cf-gnnexplainer/src and run this command:

```train
python train.py --dataset=syn1
```

>📋  For the Tree-Cycles dataset, the dataset argument should be "syn4". For the Tree-Grid dataset, it should be "syn5". All hyperparameter settings are listed in the defaults, and all models have the same hyperparameters. 


#### Training CF-GNNExplainer

To train CF-GNNExplainer for each dataset, run the following commands:

```train
python main_explain.py --dataset=syn1 --lr=0.01 --beta=0.5 --n_momentum=0.9 --optimizer=SGD
python main_explain.py --dataset=syn4 --lr=0.1 --beta=0.5 --optimizer=SGD
python main_explain.py --dataset=syn5 --lr=0.1 --beta=0.5 --optimizer=SGD
```

>📋  This will create another folder in the main directory called 'results', where the results files will be stored.


#### Evaluation

To evaluate the CF examples, run the following command:

```eval
python evaluate.py --path=../results/<NAME OF RESULTS FILE>
```
>📋  This will print out the values for each metric.

#### Pre-trained Models

The pretrained models are available in the models folder


#### Results

Our model achieves the following performance:

| Model name         | Dataset        | Fidelity       |  Size |    Sparsity   | Accuracy    |
| ------------------ |---------------- | -------------- | -------------- | -------------- |   -------------- |
| CF-GNNExplainer   |     Tree-Cycles  |      0.21       |      2.09           |       0.90        |      0.94       |
| CF-GNNExplainer   |     Tree-Grid    |      0.07       |       1.47          |      0.94         |     0.96        |
| CF-GNNExplainer   |     BA-Shapes    |      0.39       |       2.39          |       0.99        |      0.96        |



### GCFExplainer

We have already provided gnn and neurosed base models. If you want to run this method using your own dataset, firstly, you have to train your own gnn and neurosed base models.

- For gnn base models, you can use our [gnn.py](gnn.py) module.
- For neurosed base models, please follow [neurosed](https://github.com/idea-iitd/greed) repository.

If neurosed model is hard to train, you will have to update our importance function to use your graph edit distance function.

#### Generating Counterfactual Candidates

To generate counterfactual candidates for AIDS dataset with the default hyperparameters, run this command:

```train
python vrrw.py --dataset aids
```

The counterfactual candidates and meta information is saved under `results/{dataset}/runs/`. You can check other available training options with:

```train_option
python vrrw.py --help
```

#### Generating Summary Counterfactuals

To generate counterfactual summary set for AIDS dataset from the candidates with the default hyperparameters, run this command:

```summary
python summary.py --dataset aids
```

The coverage and cost performance under different number of summary size will be printed on screen. You can check other available summary options with:

```summary_option
python summary.py --help
```

## CLEAR


### Dataset
Datasets can be found in [link](https://virginia.box.com/s/941v9pwh83lfw5vnwfbgcertlsoivg5j).

### Run Experiment
#### Step 1:  Training a graph prediction model
Train a graph prediction model (i.e., the model which needs explanation). The trained prediction models used in this paper can be directly loaded from ```./model_save/```.

If you want to train them from scratch, run the following command (here we use the dataset *imdb_m* as an example):
```
python train_pred.py  --dataset imdb_m --epochs 600 --lr 0.001
```
Or you can also use any other graph prediction models instead.

#### Step 2: Generating counterfactual explanations
```
python main.py --dataset imdb_m --experiment_type train
```
Here, when ```experiment_type``` is set to *train* or *test*, the model CLEAR will be trained or loaded from a saved file. When it is set to *baseline*, you can run the random perturbation based baselines (INSERT, REMOVE, RANDOM) by setting ```baseline_type```.


## RCExplainer

### Datasets

**Datasets are NOT required separately for inference or for training of the explanation methods. For ease, we have clubbed the datasets with the trained GNN pytorch models. Download trained models from this link.**
 https://drive.google.com/file/d/14Tlv_beU8sGVk22AKsgseRF6dJEkXJAn/view?usp=sharing 
 
 **Place 'ckpt' folder in 'RCExplainer/gcn_interpretation/gnn-model-explainer/'.**

Datasets are ONLY required for training GNN model from scratch and are available for download from the following link if required:
https://drive.google.com/file/d/1k3VjFPvHWM71Lfb9AXjHcC1MlJkMvoX5/view?usp=sharing
For training GNN from scratch, please refer training GNN section. A copy of the datasets should be placed in `RCExplainer/gcn_interpretation/datasets` for this.


### Pre-trained Models 
 Download all trained models from this link:
https://drive.google.com/file/d/14Tlv_beU8sGVk22AKsgseRF6dJEkXJAn/view?usp=sharing
and place 'ckpt' folder in `RCExplainer/gcn_interpretation/gnn-model-explainer`

GNN model + dataset is saved in `RCExplainer/gcn_interpretation/gnn-model-explainer/ckpt/{dataset_name}/{dataset_name}_base.pth.tar`

Explainer models are saved in `RCExplainer/gcn_interpretation/gnn-model-explainer/ckpt/{dataset_name}/{method_name}/{method_name}.pth.tar`


## Explaining Trained GNN Models

This section covers how to evaluate already trained GNN models


### Evaluating for fidelity-sparsity

The graph classification datasets are described below. We evaluate fidelity and sparsity on these datasets. 

| Dataset Name | Reference name  | Ckpt dir          | Explainer dir | Relevant class |
| ---          | ---        | ---               | --- | --- |
| BA-2Motifs    | BA_2Motifs       | `ckpt/BA_Motifs_3gc`   | TBD | 1 |
| Mutagenicity | Mutagenicity       | `ckpt/orig_Mutagenicity`   | TBD | 0 |
| NCI1  | NCI1       |`ckpt/NCI3_3gc`    | TBD | 1|

Below are commands for different datasets
1) Mutagenicity
```
python explainer_main.py --bmname Mutagenicity --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/Mutagenicity" --exp-path "./ckpt/Mutagenicity/RCExplainer/rcexplainer.pth.tar" --multigraph-class 0 --eval

```

2) BA_2Motifs
```
python explainer_main.py --bmname BA_2Motifs --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/BA_2Motifs" --exp-path "./ckpt/BA_2Motifs/RCExplainer/rcexplainer.pth.tar" --multigraph-class 1 --eval

```

3) NCI1
```
python explainer_main.py --bmname NCI1 --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/NCI1" --exp-path "./ckpt/NCI1/RCExplainer/rcexplainer.pth.tar" --multigraph-class 1 --eval

```

General method for running:

```
python explainer_main.py --bmname {Reference name} --num-gc-layers 3 --explainer-method rcexplainer --gpu --eval --graph-mode --ckptdir {Ckpt dir}  --exp-path  {Explainer dir} --multigraph-class {Relevant class}
```

### Evaluating for noise robustness 

For noise robustness results, simply add the `--noise` flag to the above commands. For example, for graph classification:

```
python explainer_main.py --bmname {Reference name} --num-gc-layers 3 --explainer-method rcexplainer --gpu --eval --graph-mode --ckptdir {Ckpt dir}  --exp-path  {Explainer dir} --multigraph-class {Relevant class} --noise
```

Example:
```
python explainer_main.py --bmname Mutagenicity --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/Mutagenicity" --exp-path "./ckpt/Mutagenicity/RCExplainer/rcexplainer.pth.tar" --multigraph-class 0 --eval

```

### Evaluating node classification for accuracy and AUC

The node classification datasets are described below. We refer to `RCExplainer/gcn_interpretation/gnn-model-explainer/ckpt` as `ckpt`

| Dataset Name | Reference name  | Ckpt dir          | Explainer dir |AUC | Node Accuracy |
| ---          | ---        | ---               | --- |--- | --- |
| BA-Shapes    | syn1       | `ckpt/syn1`   | TBD |0.998 | 0.973 |
| BA-Community | syn2       | `ckpt/syn2`   | TBD |0.995 | 0.916 |
| Tree-Grid  | syn3       |`ckpt/syn3`    | TBD |0.993 | 0.993 |
| Tree-Cycles   | syn4       |`ckpt/syn4`    | TBD |0.995 | 0.974 |


Below are commands for node classification datasets 

1) syn1 dataset (BA-Shapes)
```
python explainer_main.py --bmname syn1 --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/syn1" --exp-path "./ckpt/syn1/RCExplainer/rcexplainer.pth.tar" --eval
```


2) syn2 dataset (BA-Community)
```
python explainer_main.py --bmname syn2 --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/syn2" --exp-path "./ckpt/syn2/RCExplainer/rcexplainer.pth.tar" --eval
```

3) syn3 dataset (Tree-Grid)
```
python explainer_main.py --bmname syn3 --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/syn3" --exp-path "./ckpt/syn3/RCExplainer/rcexplainer.pth.tar" --eval
```


4) syn4 dataset (Tree-Cycles)
```
python explainer_main.py --bmname syn4 --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/syn4" --exp-path "./ckpt/syn4/RCExplainer/rcexplainer.pth.tar" --eval
```

General method for running:

```
python explainer_main.py --bmname {Dataset name} --num-gc-layers 3 --explainer-method rcexplainer --gpu --eval --ckptdir {Ckpt dir}  --exp-path  {Explainer dir}
```


## Training Explainers

To train an explainer, you will have to pass in appropriate hyperparameters for each method.

For RCExplainer, this may include the learning rate `--lr 0.001`, the boundary and inverse boundary coefficient (which make up lambda) `--boundary_c 3 --inverse_boundary_c 12`, the size coefficient `--size_c 0.01`, the entropy coefficient `--ent_c 10`, and the boundary loss version `--bloss-version "sigmoid"` and name of folder  where logs and trained models should be stored `--prefix training_dir`. 
The logs and models will be saved at `RCExplainer/gcn_interpretation/gnn-model-explainer/ckpt/{dataset_name}/{prefix}/`


### Graph classification
Below are commands for training 

1) Mutagenicity
```
python explainer_main.py --bmname Mutagenicity --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/Mutagenicity" --multigraph-class 0  --prefix "rcexp_mutag" --lr 0.001 --size_c 0.001 --ent_c 8.0 --boundary_c 3 --inverse_boundary_c 12 --bloss-version "sigmoid" --epochs 601

```

2) BA_2Motifs
```
python explainer_main.py --bmname BA_2Motifs --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/BA_2Motifs" --multigraph-class 1 --prefix "rcexp_ba2motifs" --lr 0.001 --size_c 0.001 --ent_c 8.0 --boundary_c 3 --inverse_boundary_c 12 --bloss-version "sigmoid" --epochs 601

```

3) NCI1
```
python explainer_main.py --bmname NCI1 --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/NCI1" --multigraph-class 1 --prefix "rcexp_nci1" --lr 0.001 --size_c 0.001 --ent_c 8.0 --boundary_c 3 --inverse_boundary_c 12 --bloss-version "sigmoid" --epochs 601

```


###  Node classification

Below are commands for training


Training explainer on syn2 (BA-Community) 
```
python explainer_main.py --bmname syn2 --num-gc-layers 3 --explainer-method rcexplainer --gpu --lr 0.001 --boundary_c 10 --inverse_boundary_c 5 --size_c 0.09 --ent_c 10.0 --bloss-version "sigmoid" --topk 4 --prefix test-rc-syn2 --ckptdir "./ckpt/syn2_emb/"
```

1) syn1 dataset (BA-Shapes)
```
python explainer_main.py --bmname syn1 --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/syn1 --lr 0.001 --boundary_c 12.0 --inverse_boundary_c 3.0 --size_c 0.09 --ent_c 10.0 --bloss-version "sigmoid" --prefix test-rc-syn1" --epochs 601
```


2) syn2 dataset (BA-Community)
```
python explainer_main.py --bmname syn2 --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/syn2"  --lr 0.001 --boundary_c 12.0 --inverse_boundary_c 3.0 --size_c 0.09 --ent_c 10.0 --bloss-version "sigmoid" --prefix test-rc-syn2" --epochs 601
```

3) syn3 dataset (Tree-Grid)
```
python explainer_main.py --bmname syn3 --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/syn3"  --lr 0.001 --boundary_c 12.0 --inverse_boundary_c 3.0 --size_c 0.09 --ent_c 10.0 --bloss-version "sigmoid" --prefix test-rc-syn3" --epochs 601
```


4) syn4 dataset (Tree-Cycles)
```
python explainer_main.py --bmname syn4 --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/syn4" --lr 0.001 --boundary_c 12.0 --inverse_boundary_c 3.0 --size_c 0.09 --ent_c 10.0 --bloss-version "sigmoid" --prefix test-rc-syn4" --epochs 601
```

## Other baselines
For training other baselines, replace ``--explainer-method rcexplainer`` in above commands with ``--explainer-method {baseline}`` where baseline can be picked from these methods  gnnexplainer | pgexplainer | pgmexplainer | rcexp_noldb. 

Some of the trained baselines and info can be downloaded from this url:
https://drive.google.com/file/d/1t1i8kNjtGkqhI43ehGFvlxvbEan7HL80/view?usp=sharing
For inference on these methods, simply change --exp-path in above inference commands to the provide explainer models. Inference of pgexplainer and rcexp_noldb baseline models is supported by rcexplainer, so keep the --explainer-method rcexplainer in inference commands. 


## Training GNN models from scratch

To train the GNN models from scratch, more setup is required:

### Dependencies

It is necessary to install [pytorch geometric][pytorch-geometric.readthedocs.io/en/latest/notes/installation.html] matching your CUDA (10.1) and PyTorch version (1.8.0). 

Please run the following to install this package:  

``--trusted-host`` and ``--no-cache-dir`` are required to circumvent network restrictions

Please replace TORCH and CUDA versions as appropriate. 

```
conda install cudatoolkit-dev=10.1 -c pytorch
export TORCH=1.8.1
export CUDA=10.1
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html --trusted-host pytorch-geometric.com --no-cache-dir
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html --trusted-host pytorch-geometric.com --no-cache-dir
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html --trusted-host pytorch-geometric.com --no-cache-dir
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html --trusted-host pytorch-geometric.com --no-cache-dir
pip install torch-geometric --trusted-host pytorch-geometric.com --no-cache-dir

```


### Training

A model can be trained on a node classification dataset by:

```
python train.py --dataset syn1 --gpu
```

A model can be trained on a graph classification dataset by:

```
python train.py --bmname Mutagenicity --gpu --datadir {/path/to/benchmark/dataset}
```

For more details on training GNN models from scratch, check this GNNExplainer url
https://github.com/RexYing/gnn-model-explainer
