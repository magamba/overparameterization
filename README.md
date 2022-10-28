# README

## Set up

To install all required dependencies, run:
```bash
pip install -r requirements.txt
```

## Configure environment

To configure the computing environment for running the experiments, you can set the following environment variables
```bash
SAVE_DIR="/path/to/savedir" # basedir where to store results
NAME="exp_name" # name of the experiment run
DATA_DIR="/path/to/dataset/dir" # path where to load data from
DEVICE="cuda" # "cpu" or "cuda"
WORKERS=1 # number of CPU worker processes
```
or use the following command line arguments with `train.py` and `compute_stats.py`:
```bash
 --save-dir="/path/to/savedir"
 --name="exp_name"
 --data-dir="/path/to/dataset/dir"
 --device="cuda"
 --workers="1"
```

# Train a model

To train a model on CIFAR-10 with 20% noisy labels, run
```bash
python train.py --device=cuda \
                --name='test' \
                --save-dir='checkpoints' \
                --data-dir="./data" \
                --workers=4 \
                --data="cifar10" \
                --label-noise=0.2 \
                --model="resnet18_64" \
                --optimizer=sgd \
                --batch-size=128 \
                --epochs=300 \
                --augmentation \
                --momentum=0.9 \
                --weight-decay=0 \
                --learning-rate=0.005 \
                --lr-warmup-epochs=5 \
                --lr-warmup-initial=0.001 \
                --seed=42 \
                --train-split=49000 \
                --val-split=1000 \
                --eval-every=10 
```
model checkpoints for the corresponding run will be compressed to a single zip archive, and stored to `./checkpoints/test/resnet18_64/cifar10/augmentation/seed-42/checkpoints.zip`

# Compute statistics

To compute data-driven measures, the user must specify a metric to compute and what model checkpoint to load. A model checkpoint is identified by specifying a network architecture, dataset, training seed, checkpoint and dataset split:
```bash
python compute_stats.py --device="cuda" \
                        --name=NAME \
                        --save-dir=SAVE_DIR \
                        --workers=4 \
                        --data-dir=DATA_DIR \
                        --data=cifar10 \
                        --dataset-split=train \
                        --label-noise=0.2 \
                        --model=resnet18_64 \
                        --augmentation \
                        --seed 42 \
                        --checkpoint=1 \
                        --train-split=49000 \
                        --val-split=1000 \
                        --num-samples=49000 \
                        --batch-size=140 \
                        --metric=jacobian_operator_norm
```
Results are stored in uncompressed json format to `SAVE_DIR/NAME/MODEL/DATA/TRAINING_SETTING/seed-SEED/OUT_NAME-BATCH_ID.json`, where `BATCH_ID` denotes the statistics computed for batch number `BATCH_ID`.

The number of data points to use to compute each metric is controlled by `--num-samples`. **Note:** `--batch-size` should divide `--num-samples`.

## Normalization

The network's Jacobian can be rescaled using several strategies, specified with the argument `--normalization`:
- `None`: network output and Jacobian matrix are unnormalized (default).
- `softmax`: applies softmax to the network output, before computing the Jacobian w.r.t. the network's input.
- `logsoftmax`: applies logsoftmax to the network output, before computing the Jacobian w.r.t. the network's input.
- `crossentropy`: applies crossentropy to the network output, before computing the Jacobian w.r.t. the network's input.

## Seeding

The following seeds should be set to control randomness:
- `--data-split-seed`: used for splitting train/validation set.
- `--label-seed`: used to corrupt training labels.
- `--mc-sample-seed`: used for sampling batches from the dataset (`compute_stats.py` only).
- `--seed`: used for initializing models during training, and creating and shuffling batches of data (`train.py` only).

# Large-scale experiments

To speed up computation, if multiple GPUs are available, several parallel instances of `compute_stats.py` can be launched, and computation can be parallelized in the batch dimension. 

It is recommended to launch one instance of `compute_stats.py` for each dataset/model/dataset_split/checkpoint, and to further process batches in parallel across several independent compute jobs.
To do so, one instance of `compute_stats.py` can be lauched for each available GPU. Then, the `--skip=NUM` argument can be used to split the available training points across GPU workers. For instance, for 2 GPU workers, processing `256` training points and using batch size `128`:
```bash
# GPU 0
  --num-samples=256 --batch-size=128
# GPU 1
  --num-samples=256 --batch-size=128 --skip=128
```
will assign the first batch of 128 training points to rank 0 and the others to rank 1.

## Speedig up computation

By default, Jacobian matrices are estimated with one backward pass per output dimension of the network. If enough GPU memory is available, the Jacobian can be computed in a single backward pass for all output dimensions at once. The memory cost to perform this operation increases by a factor of `K`, denoting the output dimension of the network (e.g. `K=10` for CIFAR-10). To enable this option, use `--bigmem`.

# CONTRIBUTING

The source code is organized as follows:
- Model definitions can be found under `models`.
- Data augmentation transforms as well as label corruption algorithms are found under `core/data.py`.
- Metrics are defined `core/metrics.py`, with torch scripted helper functions available in `core/metric_helpers.py`.
- Data generation strategies are initialized in `core/strategies.py`.
- The main script for launching experiments is `compute_stats.py`.
