
- export PYTHONPATH=$PYTHONPATH:"$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/CE/gnn_cff"

CF2 opt=0.6 --> python CE/gnn_cff/scripts/exp_node_ba_shapes.py

python CE/gnn_cff/scripts/exp_node_ba_shapes.py --alp 1

python CE/gnn_cff/scripts/exp_node_ba_shapes.py --alp 0

sbatch --job-name=CEvsAE --mail-user=amir.reze@uibk.ac.at --time=10:00:00 --mem=80G /home/amir.reza/jobs/single-node-gpu.job "python CE/gnn_cff/scripts/exp_node_ba_shapes.py " 