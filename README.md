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

This repository contains the official implementation of the paper: [NuTime: Numerically Multi-Scaled Embedding for Large-Scale Time-Series Pretraining](https://arxiv.org/abs/2310.07402), which is accepted to TMLR 2024.
In this work, we propose the <b>NuTime</b> model for large-scale time series pretraining. The model is based on the Transformer architecture, which takes input as a set of tokens from non-overlapping windows. Each window is represented by its normalized shape, the window mean and standard deviation. We develop a <b>numerically multi-scaled embedding</b> method (NME) for representing the scalar values of mean and std. The model can <u>take raw values of time-series data in <b>any numerical scales</b> as input <b>without</b> any data normalization and transformation</u>.

Feel free to contact me (chenguolin@stu.pku.edu.cn) or open an issue if you have any questions or suggestions.


## 📢 News
- **2024-11-12**: Checkpoint of the self-supervised pretrained NuTime is released.
- **2024-11-12**: Codes about data preprocessing, training, evaluation are released.
- **2024-07-15**: It might take some time to clean the entire codebase for releasing, so we first provide the code about **window & mean & std embeddings**, which is the essential part of the proposed NuTime, at [here](./src/models/encoders/WindowNormEncoder.py).
- **2024-07-10**: NuTime is accepted to TMLR 2024.


## 📋 TODO
- [x] Release the training and evaluation code
- [x] Release the self-supervised pretrained NuTime


## 🔧 Installation
Please install PyTorch according to your CUDA version first. There are not restrictions on the torch version, feel free to use your preferred one.
```bash
git clone https://github.com/chenguolin/NuTime.git
cd NuTime
bash settings/setup.sh
```


## 📊 Dataset
Please refer to [src/data/preprocess.py](./src/data/preprocess.py).
We provide the script to preprocess the data including: `UCR`, `UEA`, `SleepEDF`, `Epilepsy`, etc.
The processed and splitted `Epilpesy` dataset is provided in [datasets/Epilepsy](./datasets/Epilepsy) for example.


## 🚀 Usage
- The core part of our work is `WindowNormEncoder` in [src/models/encoders/WindowNormEncoder.py](./src/models/encoders/WindowNormEncoder.py) and `WinT` in [src/models/networks.py](./src/models/networks.py). You can directly view the code for implementation details. Other codes are merely for data preprocessing, training, evaluation and ablation study, which could be ignored essentially.

- Checkpoint of the self-supervised (i.e., BYOL-style) pretrained NuTime (with `9` multi-scaled embeddings) is provided in [ckpt/checkpoint_bias9.pth](./ckpt/checkpoint_bias9.pth)

### Finetune Pretrained NuTime for Epilepsy dataset
```bash
python3 src/pipeline.py --config_file configs/demo_ft_epilepsy.json
```


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
