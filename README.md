# [TMLR 2024] NuTime

<h4 align="center">

NuTime: Numerically Multi-Scaled Embedding for Large-Scale Time-Series Pretraining

[Chenguo Lin](https://chenguolin.github.io), [Xumeng Wen](https://github.com/xumwen), [Wei Cao](https://weicao1990.github.io/), [Congrui Huang](https://dblp.org/pid/26/8737.html), [Jiang Bian](https://sites.google.com/view/jiangbian), [Stephen Lin](https://www.microsoft.com/en-us/research/people/stevelin/), [Zhirong Wu](https://www.microsoft.com/en-us/research/people/wuzhiron/)

[![OpenReview](https://img.shields.io/badge/OpenReview-Page-blue)](https://openreview.net/forum?id=TwiSBZ0p9u)
[![arXiv](https://img.shields.io/badge/arXiv-2310.07402-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2310.07402)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

<p>
    <img width="730" alt="pipeline", src="./assets/pipeline.png">
</p>

</h4>

This repository contains the official implementation of the paper: [NuTime: Numerically Multi-Scaled Embedding for Large-Scale Time-Series Pretraining](https://arxiv.org/abs/2310.07402), which is accepted by TMLR 2024.
In this work, we propose the <b>NuTime</b> model for large-scale time series pretraining. The model is based on the Transformer architecture, which takes input as a set of tokens from non-overlapping windows. Each window is represented by its normalized shape, the window mean and the window standard deviation. We develop a <b>numerically multi-scaled embedding</b> method (NME) for representing the scalar values of mean and std. The model can <u>take raw values of time-series data as input <b>without</b> any data normalization and transformation</u>.

Feel free to contact me (chenguolin@stu.pku.edu.cn) or open an issue if you have any questions or suggestions.


## 📢 News
- **2024-07-10**: NuTime is accepted by TMLR 2024.


## 📋 TODO
- [ ] Release the training and evaluation code
- [ ] Release the self-supervised pretrained NuTime
- [ ] Release the large-scale merged datasets for pretraining


## 📚 Citation
If you find our work helpful, please consider citing:
```bibtex
@article{lin2024nutime,
  title={NuTime: Numerically Multi-Scaled Embedding for Large-Scale Time-Series Pretraining},
  author={Chenguo Lin and Xumeng Wen and Wei Cao and Congrui Huang and Jiang Bian and Stephen Lin and Zhirong Wu},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2024}
}
```
