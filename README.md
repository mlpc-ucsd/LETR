# LETR: Line Segment Detection Using Transformers without Edges

## Introduction 
This repository contains the official code and pretrained models for [Line Segment Detection Using Transformers without Edges](https://arxiv.org/abs/2101.01909). [Yifan Xu*](https://yfxu.com/), [Weijian Xu*](https://weijianxu.com/), [David Cheung](https://github.com/sawsa307), and [Zhuowen Tu](https://pages.ucsd.edu/~ztu/). CVPR2021 (**Oral**)

In this paper, we present a holistically end-to-end algorithm for line segment detection with transformers that is post-processing and heuristics-guided intermediate processing (edge/junction/region detection) free. Our method, named LinE segment TRansformers (LETR), tackles the three main problems in this domain, namely edge element detection, perceptual grouping, and holistic inference by three highlights in detection transformers (DETR) including tokenized queries with integrated encoding and decoding, self-attention, and joint queries respectively.

<img src="figures/pipeline.svg" alt="Model Pipeline" width="720" />


## Code 

Coming Soon

### Citation

If you use this code for your research, please cite our paper:
```
@article{xu2021line,
  title={Line Segment Detection Using Transformers without Edges},
  author={Xu, Yifan and Xu, Weijian and Cheung, David and Tu, Zhuowen},
  journal={arXiv preprint arXiv:2101.01909},
  year={2021}
}
```
### Acknowledgments

This code is based on the implementations of [**DETR: End-to-End Object Detection with Transformers**](https://github.com/facebookresearch/detr). 