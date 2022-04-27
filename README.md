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
					Test scores						Valid scores					
	Config				Diginetica		YOOCHOOSE-4		YOOCHOOSE-64		Diginetica		YOOCHOOSE-4		YOOCHOOSE-64	
Model	hidden_dim	inner_dim	N	num_head	Recall@20	MRR@20	Recall@20	MRR@20	Recall@20	MRR@20	Recall@20	MRR@20	Recall@20	MRR@20	Recall@20	MRR@20
BERT	16	64	1	2												
	32	128	1	2	50.39%	17.27%	68.91%	28.50%	68.11%	28.66%	55.68%	19.54%	70.39%	30.68%	67.35%	30.21%
	64	256	1	2	50.28%	17.52%	69.94%	29.84%	68.99%	29.99%	54.81%	19.24%	71.35%	32.00%	68.05%	30.93%
	128	512	1	2	48.93%	16.59%	69.58%	30.14%	68.28%	29.81%	53.55%	17.27%	71.58%	32.59%	67.96%	31.29%
	256	1024	1	2												
GLBERT	16	64	1	2												
	32	128	1	2	50.49%	17.63%	69.16%	29.03%	67.60%	28.80%	54.80%	19.18%	70.65%	31.25%	67.42%	30.39%
	64	256	1	2	49.48%	17.49%	70.04%	30.41%	67.31%	29.38%	53.62%	18.89%	71.54%	32.68%	66.95%	30.77%
	128	512	1	2	47.60%	16.45%	66.65%	29.19%	66.65%	29.19%	51.79%	17.97%	66.62%	30.64%	66.62%	30.64%
![image](https://user-images.githubusercontent.com/51608554/165471487-564a3fb0-5640-4d75-815a-e56c58563c87.png)
