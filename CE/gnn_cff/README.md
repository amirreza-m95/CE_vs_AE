
###for cuda 11.7 using pip/linux
pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
##### This may result in changing your torch version, track your torch version.


pip install matplotlib


### change graphconv.py
Path_to_your_pythonVenv/venv/lib/python3.10/site-packages/dgl/nn/pytorch/conv/graphconv.py


### insert the path
export PYTHONPATH=$PYTHONPATH:"$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/CE/gnn_cff"

### unzip the log file the gnn_cff directory
unzip log.zip