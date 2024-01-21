# CE_vs_AE
Counterfacutal vs Adversarial Examples


CFF
- Moving Graphconv.py file and changing transform to transforms in library import.
- add this graph conv file beside script


Torch:
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
dgl: 
# If you have installed dgl-cuXX package, please uninstall it first.
pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

Running Xp BA shapes CFF:

    export PYTHONPATH=$PYTHONPATH:"$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/CE/gnn_cff"
