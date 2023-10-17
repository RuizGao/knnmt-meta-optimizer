# kNN-MT is Meta-Optimizer on OPL

Code for our EMNLP 2023 paper "Nearest Neighbor Machine Translation is Meta-Optimizer on Output Projection Layer". 
Please cite our paper if you find this repository helpful in your research:

The implementation is build upon [fairseq](https://github.com/pytorch/fairseq), and heavily inspired by [adaptive-knn-mt](https://github.com/zhengxxn/adaptive-knn-mt), many thanks to the authors for making their code avaliable.

## Requirements and Installation

* pytorch version >= 1.5.0
* python version >= 3.6
* faiss-gpu >= 1.6.5
* pytorch_scatter = 2.0.5
* 1.19.0 <= numpy < 1.20.0

You can install this project by
```
pip install --editable ./
```

