# HPGCN

## Installation

```shell
git clone 
cd HPGCN
conda env create -f hpgcn.yaml
conda activate hpgcn
pip install -e .
```

## Data Preparation

PYSKL provides links to the pre-processed skeleton pickle annotations.

- NTU RGB+D: [NTU RGB+D Download Link](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl)
- NTU RGB+D 120: [NTU RGB+D 120 Download Link](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl)
- NTU RGB+D 120: [NTU RGB+D 120 actions labelled](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
- Kinetics-Skeleton: [Kinetics-Skeleton Download Link](https://download.openmmlab.com/mmaction/pyskl/data/k400/k400_hrnet.pkl)
- FineGYM: [FineGYM Download Link](https://download.openmmlab.com/mmaction/pyskl/data/gym/gym_hrnet.pkl)


For Kinetics-Skeleton, since the skeleton annotations are large, please use the [Kinetics Annotation Link](https://www.dropbox.com/scl/fi/5phx0m7bok6jkphm724zc/kpfiles.zip?rlkey=sz26ljvlxb6gwqj5m9jvynpg8&st=47vcw2xb&dl=0) to download the `kpfiles` and extract it under `$HPGCN/data/k400` for Kinetics-Skeleton. 

## Training & Testing

We support distributed training on a single server with multiple GPUs.

```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# For example
bash tools/dist_train.sh configs/ntu60_xview/jm.py 1 --validate --test-last --test-best
```

```shell
# Testing
bash tools/dist_test.sh {config_name} {checkpoint_file} {num_gpus} {other_options}
# For example
bash tools/dist_test.sh configs/ntu60_xview/jm.py checkpoints/CHECKPOINT.pth 1 --eval top_k_accuracy --out result.pkl
```

```shell
# Ensemble the results
cd tools
python ensemble.py
```

## Acknowledgements

This work is mainly based on [PYSKL](https://github.com/kennymckormick/pyskl). We also refer to [ProtoGCN](https://github.com/firework8/ProtoGCN).

Thanks their work!
