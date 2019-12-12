# 3D Transfer Learning - PointDAN

This repo contains the source code and dataset for our NeurIPS 2019 paper:

[**PointDAN: A Multi-Scale 3D Domain Adaption Network for Point Cloud Representation**](http://papers.nips.cc/paper/8940-pointdan-a-multi-scale-3d-domain-adaption-network-for-point-cloud-representation)
<br>
2019 Conference on Neural Information Processing Systems (NeurIPS 2019)
<br>
[paper](http://papers.nips.cc/paper/8940-pointdan-a-multi-scale-3d-domain-adaption-network-for-point-cloud-representation),
[arXiv](https://arxiv.org/abs/1911.02744),
[bibtex](http://papers.nips.cc/paper/8940-pointdan-a-multi-scale-3d-domain-adaption-network-for-point-cloud-representation/bibtex)

![PointDAN](/Figs/PointDAN.png)

## Introduction
Domain Adaptation (DA) approaches achieved significant improvements in a wide range of machine learning and computer vision tasks (i.e., classification, detection, and segmentation). However, as far as we are aware, there are few methods yet to achieve domain adaptation directly on 3D point cloud data. The unique challenge of point cloud data lies in its abundant spatial geometric information, and the semantics of the whole object is contributed by including regional geometric structures.  Specifically, most general-purpose DA methods that struggle for global feature alignment and ignore local geometric information are not suitable for 3D domain alignment. In this paper, we propose a novel 3D Domain Adaptation Network for point cloud data (PointDAN). PointDAN jointly aligns the global and local features in multi-level. For local alignment, we propose Self-Adaptive (SA) node module with an adjusted receptive field to model the discriminative local structures for aligning domains. To represent hierarchically scaled features, node-attention module is further introduced to weight the relationship of SA nodes across objects and domains. For global alignment, an adversarial-training strategy is employed to learn and align global features across domains. Since there is no common evaluation benchmark for 3D point cloud DA scenario, we build a general benchmark (i.e., PointDA-10) extracted from three popular 3D object/scene datasets (i.e., ModelNet, ShapeNet and ScanNet) for cross-domain 3D objects classification fashion. Extensive experiments on PointDA-10 illustrate the superiority of our model over the state-of-the-art general-purpose DA methods.


## Dataset
![PointDA-10](/Figs/PointDA-10.png)
The [PointDA-10](https://drive.google.com/file/d/1LO6ec90UTXWw2QOin30adfDQqEAgApzq/view?usp=sharing) dataset is extracted from three popular 3D object/scene datasets (i.e., [ModelNet](https://modelnet.cs.princeton.edu/), [ShapeNet](https://shapenet.cs.stanford.edu/iccv17/) and [ScanNet](http://www.scan-net.org/)) for cross-domain 3D objects classification.

## Requirements
- Python 3.6
- PyTorch 1.0


## File Structure
```
Code will come soon.
```


## Citation

If you find this project useful for your research, please kindly cite our paper:

```bibtex
@incollection{NIPS2019_8940,
title = {PointDAN: A Multi-Scale 3D Domain Adaption Network for Point Cloud Representation},
author = {Qin, Can and You, Haoxuan and Wang, Lichen and Kuo, C.-C. Jay and Fu, Yun},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {7190--7201},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/8940-pointdan-a-multi-scale-3d-domain-adaption-network-for-point-cloud-representation.pdf}
}
```

## Contact
If you have any questions, please contact [Can Qin](qin.ca@husky.neu.edu) or [Haoxuan You](haoxuan.you@columbia.edu).
