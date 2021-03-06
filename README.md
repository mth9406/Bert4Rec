# BERT4Rec-PyTorch
A PyTorch implementation of the BERT4Rec
Here are two datasets used in this repo... After downloading the datasets, you can put them in the folder `datasets/`

* [YOOCHOOSE](https://www.kaggle.com/chadgostopp/recsys-challenge-2015)
* [DIGNETICA](https://competitions.codalab.org/competitions/11161)

# How to use
First of all, run the file `datasets/preprocess.py` to preprocess "YOOCHOOSE" or "DIGNETICA".

```bash
usage: preprocess.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  dataset name: diginetica/yoochoose/sample
```

```bash
usage: main.py [-h] [--dataset_path DATASET_PATH] [--batch_size BATCH_SIZE]
               [--epoch EPOCH] [--lr LR] [--lr_dc LR_DC]
               [--lr_dc_step LR_DC_STEP] [--model_type MODEL_TYPE] [--N N]
               [--hidden_dim HIDDEN_DIM] [--num_head NUM_HEAD]
               [--inner_dim INNER_DIM] [--max_length MAX_LENGTH] [--test]
               [--topk TOPK] [--valid_portion VALID_PORTION]
```

# Optional arguments
```bash
optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        dataset directory path: datasets/diginetica or
                        datasets/yoochoose1_4/ or datasets/yoochoose1_64
  --batch_size BATCH_SIZE
                        input batch size
  --epoch EPOCH         the number of epochs to train for
  --lr LR               learning rate
  --lr_dc LR_DC         learning rate decay rate
  --lr_dc_step LR_DC_STEP
                        the number of steps after which the learning rate
                        decay
  --model_type MODEL_TYPE
                        the model type, 0: BERT4Rec
  --N N                 the number of transformer encoder layers, default= 12
  --hidden_dim HIDDEN_DIM
                        the size of hidden dimension, default= 512
  --num_head NUM_HEAD   the number of heads, default= 8
  --inner_dim INNER_DIM
                        the size of inner_dim of FCN layer, default= 2048
  --max_length MAX_LENGTH
                        max length of a session, default= 100
  --test                test
  --topk TOPK           the number of top score items selected for calculating
                        recall and mrr metrics, default= 20
  --valid_portion VALID_PORTION
                        split the portion of training set as validation set
```
# Experiment results
![image](https://user-images.githubusercontent.com/51608554/165471487-564a3fb0-5640-4d75-815a-e56c58563c87.png)
