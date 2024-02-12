# Counterfactual Explanation_vs_Adversarial Examples
Counterfacutal Explanation vs Adversarial Examples


### CFF Guide
1. To run the experiments, dgl-cuda library is required. https://docs.dgl.ai/install/index.html.
2. After installing dgl-cuda library, Please replace the graphconv.py file in the dgl library with the file (with the same name) provided in the gnn_cff root folder. This is for the relaxation purpose as described in the paper.
3. The training and explaining are independent phases. For you convenience, we provide pre-trained GNN model params. Under the project root folder, run:
    ```
    unzip log.zip
    ```
4. To set the python path, under the project root folder, run:
    ```
    source setup.sh
    ```
5. As an example, to generate explanations for node classification task (Tree-Cycles dataset as example), run:
    ```
    python scripts/exp_node_tree_cycles.py
    ```
6. For generating explanations for graph classification task (Mutag0 dataset as example), run:
    ```
    python scripts/exp_graph.py
    ```
7. The codes for training GNNs are also provided. For instance, for training GNN for node classification, run:
    ```
    python scripts/train_node_classification.py
    ```
    for graph classification, run:
    ```
    python scripts/train_graph_classification.py
    ```
- Moving Graphconv.py file and changing transform to transforms in library import.
- add this graph conv file beside script


Torch:
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
dgl: 
#### If you have installed dgl-cuXX package, please uninstall it first.
pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

Running Xp BA shapes CFF:

    export PYTHONPATH=$PYTHONPATH:"$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/CE/gnn_cff"


### CF-GNNExplainer Guide
#### Training original models

To train the original GNN models for the BA-shapes dataset in the paper, cd into src and run this command:

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

We have already provided gnn and neurosed base models. If you want to run our method using your own dataset, firstly, you have to train your own gnn and neurosed base models.

- For gnn base models, you can use our [gnn.py](gnn.py) module.
- For neurosed base models, please follow [neurosed](https://github.com/idea-iitd/greed) repository.

If neurosed model is hard to train, you will have to update our importance function to use your graph edit distance function.

## Generating Counterfactual Candidates

To generate counterfactual candidates for AIDS dataset with the default hyperparameters, run this command:

```train
python vrrw.py --dataset aids
```

The counterfactual candidates and meta information is saved under `results/{dataset}/runs/`. You can check other available training options with:

```train_option
python vrrw.py --help
```

## Generating Summary Counterfactuals

To generate counterfactual summary set for AIDS dataset from the candidates with the default hyperparameters, run this command:

```summary
python summary.py --dataset aids
```

The coverage and cost performance under different number of summary size will be printed on screen. You can check other available summary options with:

```summary_option
python summary.py --help
```
