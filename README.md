## Code of our paper "Enhancing Cross-lingual Natural Language Inference by Prompt-learning from Cross-lingual Templates"

## Prerequisites

 * Python 3.7
 * tensorflow==2.4.0
 * keras_bert==0.88.0
 * transformers==4.12.3
 * sentencepiece==0.1.96

Our codes are based on the [bert4keras](https://github.com/bojone/bert4keras) framework.

### Datasets
We use XNLI and PAWS-X in our experiments.

| Datasets           | Download Links                                                       |
|--------------------|----------------------------------------------------------------------|
| XNLI               | https://cims.nyu.edu/~sbowman/xnli/                                  |
| XNLI (fewshot)     | https://github.com/mprompting/xlmrprompt                             |
| PAWS-X             | https://github.com/google-research-datasets/paws/tree/master/pawsx   |

Put the datasets into the [PCT/datasets/]().

We also provide the code for converting the pytorch checkpoint to the tensorflow version (see the directory ``torch2tf``).

## Use examples
You can run our models on the google [colab platform](https://colab.research.google.com/).

1. Install requirements:

``! pip install keras_bert transformers sentencepiece tensorflow==2.4.0``

2. Train model on TPUs:

``! python PCT/train_xnli.py``

3. Evaluate model on TPUs:

``! python PCT/predict_xnli.py``

We also provide the notebook file, please see ``MAIN.ipynb``

## Citation
Please consider citing the following paper if you find our codes helpful. Thank you!

```
@inproceedings{QiWDC22,
  author    = {Kunxun Qi and Hai Wan and Jianfeng Du and Haolan Chen},
  title     = {Enhancing Cross-lingual Natural Language Inference by Prompt-learning
               from Cross-lingual Templates},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational
               Linguistics},
  pages     = {1910--1923},
  year      = {2022}
}
```


