# RW-GAT

# Requirements

* python=3.9
* numpy=1.22.4
* torch=1.11.0
* torch-geometric=2.0.4
* names-generator=0.1.0
* python-box=6.0.2

# Training 
First update the `config.yml` yaml file, used to set the hyperparams and model architecture.

Training, validation and testing can be started with:
`python train.py -c path/to/config.yml`
