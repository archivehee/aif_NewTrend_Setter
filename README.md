# Process Flow

[DATA]
0. Data selection
Amazon Review Dataset (2023)
LINK: https://amazon-reviews-2023.github.io/

Domain: Home_and_Kitchen
with 23.2M users, 3.7M items, 67.4M Reviews

1. Data Precheck
data_precehck.ipynb
- Item perspective summary about <REACHTIME ~# interactions> is written in 'home_itm_summary.csv'
- Preview of Item metadata (To select which fields text into embedding by nomic-embed-text:v1.5)

2. Data Split : General Experimental Settings
- In detail is in data_precheck
- Leave-one-out construction
(interaction time t as TEST, time t-1 as VALID and the others as TRAIN)
{train/valid/test} set manifest is in manifest.json 
- executed by data_split_rnd.py

2-1. Extreme cold situation
- executed by data_split_rnd.py : very high cold ratio in test

3. Text Embedding Generation
- nomic-embed-text:v1.5 by calling API KEY
- output: aigs/NTS/data/Home_and_Kitchen/itm_txt_emb/itm_txt_emb_home.pkl
- Item index order matched to item2id.txt

[Experiment : Comparison Model]
- LightGCN (SIGIR, 2020)
from https://github.com/gusye1234/LightGCN-PyTorch

cd ./model/LightGCN-PyTorch/code
# Home_and_Kitchen
with 99 negative sampling + 1 ground truth sample

python main.py \
  --dataset "Home_and_Kitchen" \
  --model "lgn" \
  --topks "[10,20]" \
  --dropout 0.1 \
  --bpr_batch 256 \
  --recdim 64 \
  --layer 3 \
  --lr 0.001

[TEST] {'hr': array([0.17342227, 0.29604926]), 'ndcg': array([0.09022834, 0.12088627])}

[Experiment : Ours]
