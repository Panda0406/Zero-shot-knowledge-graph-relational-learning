# Generative Adversarial Zero-Shot Relational Learning for Knowledge Graphs

PyTorch implementation of zero-shot knowledge graph learning model described in our AAAI 2020 paper [Generative Adversarial Zero-Shot Relational Learning for Knowledge Graphs](https://arxiv.org/pdf/2001.02332.pdf). In this work, we propose a solution for zero-shot knowledge graph relational learning via generative adversarial learning.

## Steps to run the experiments

### Requirements
* ``Python 3.6 ``
* ``>= PyTorch 1.0.0``
* ``tqdm``

### Datasets and word embeddings
* NELL-ZS: Please download([Baidu Yun](https://pan.baidu.com/s/1rvhmrl36KCfHYCnggHYl5Q) with code ``be85`` or Google Drive) and put it into this directory.
* Wiki-ZS: Please download([Baidu Yun](https://pan.baidu.com/s/1GfhHLvd_LS_kfm05fvlKfg) with code ``9zh5`` or Google Drive) and put it into directory ``./origin_data``.

### Training on NELL-ZS dataset
* python trainer.py --pretrain_feature_extractor


### Reference
```
@article{qin2020generative,
  title={Generative Adversarial Zero-Shot Relational Learning for Knowledge Graphs},
  author={Qin, Pengda and Wang, Xin and Chen, Wenhu and Zhang, Chunyun and Xu, Weiran and Wang, William Yang},
  journal={arXiv preprint arXiv:2001.02332},
  year={2020}
}
```
