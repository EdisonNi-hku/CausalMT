This repo contains the codes of Causality for Machine Translation.

- `code/` stores the scripts to run NMT models.
- `data/` stores the code to generate MT data between cause and effect languages.
  - `data/raw_datasets` data of five language pairs with clear translation direction.
  - `data/bpe_datasets` BPE parsed `data/raw_datasets`
  - `data/raw_topic_datasets` data of five language pairs with clear translation direction and topic controlled.
  - `data/topic_bpe_datasets` BPE parsed `data/raw_topic_datasets`, filtered with length ratio 1.5.

### Configure the Environment
copy and run the following:
```
# $DIR_ABS_PATH="a path that you prefer"

# 1. prepare conda environment
conda create -n fairseq python==3.7 --yes
source activate fairseq
cd $DIR_ABS_PATH
cd code/tools/fairseq
pip install --editable ./
cd $DIR_ABS_PATH
pip install -r code/requirements.txt
pip install sacrebleu==1.5.1
mkdir -p data/run_nmt/ssl
```

### How to Run NMT Models
#### Supervised Experiments
To run the supervised experiments, the first step is to prepare the supervised training data with causal:anticausal=0:100, 25:75, 50:50, 75:25, 100:0. Following is an example of preparing en-es data.
```
# Translation from En to Es
bash code/supervise_mix.sh 0 100 0 0 0 data/bpe_datasets
bash code/supervise_mix.sh 0 75 25 0 0 data/bpe_datasets
bash code/supervise_mix.sh 0 50 50 0 0 data/bpe_datasets
bash code/supervise_mix.sh 0 75 25 0 0 data/bpe_datasets
bash code/supervise_mix.sh 0 0 100 0 0 data/bpe_datasets

# Translation from Es to En
bash code/supervise_mix.sh 0 100 0 1 0 data/bpe_datasets
bash code/supervise_mix.sh 0 75 25 1 0 data/bpe_datasets
bash code/supervise_mix.sh 0 50 50 1 0 data/bpe_datasets
bash code/supervise_mix.sh 0 75 25 1 0 data/bpe_datasets
bash code/supervise_mix.sh 0 0 100 1 0 data/bpe_datasets
```
Here, `bash code/supervise_mix.sh {lang_ix} {cau_ratio} {ant_ratio} {mt_dir} {if_ssl} {dataset}`
- `lang_ix` (language pair index) can be `0` (en-es), `1` (es-fr), `2` (en-fr), `3` (de-en), `4` (de-fr)
- `cau_ratio` (causal ratio) is the percentage of causal training data mixed, can be a float number between `0~100`. Causal and anti-causal are defined according to the alphabetical order of languages, e.g. for `en-es`, `en->es` is causal and `es->en` is anti-causal
- `ant_ratio` (anti-causal ratio) is the percentage of anti-causal training data mixed, similar to `cau_ratio`
- `mt_dir` (machine translation direction) can be `0`(alphabetical order e.g. from en to es) or `1`(reverse alphabetical order e.g. from es to en)
- `if_ssl` (if using SSL to train) can be `0` (no ssl), `1` (self-training), `4` (back-translation)
The commands will make directories like `data/run_nmt/ssl/all_100_0_en-es_ssl0`
- `dataset` location of the dataset used

The next step is to run the experiments
```
# Supervised MT, translate from en to es, data constituent: `en-to-es`:`es-to-en`=100:0
bash code/script_ssl_bt_conf.sh 0 100-0 0 0 1 0,1,2,3,4,5,6,7
```
Here, `bash code/script_ssl_bt_conf.sh {lang_ix} {directory} {mt_dir} {if_ssl} {hyp} {CUDA_VISIBLE_DEVICES}`
- `lang_ix` (language pair index) can be `0` (en-es), `1` (es-fr), `2` (en-fr), `3` (de-en), `4` (de-fr)
- `directory` the directory used, e.g. `data/run_nmt/ssl/all_{100_0}_en-es_ssl0` => directory = `100_0`
- `mt_dir` (machine translation direction) can be `0`(alphabetical order e.g. from en to es) or `1`(reverse alphabetical order e.g. from es to en)
- `if_ssl` (if using SSL to train) can be `0` (no ssl), `1` (self-training), `4` (back-translation)
- `hyp` (hyperparameter) index of current hyperparameter set

Finally, test the supervised model with `code/sup_test.sh`
```
bash code/sup_test.sh 0 100-0 0 0 1 0
```
Here, `bash code/sup_test.sh {lang_ix} {directory} {mt_dir} {if_ssl} {hyp} {CUDA_VISIBLE_DEVICES}`

#### Self-training and Back-translation
Again, step one is data preparation. Following is an example of en-es pair
```
bash code/semi_supervise_mix.sh 0 data/bpe_datasets
```
Here, `bash code/supervise_mix.sh {lang_ix} {dataset}`
- `lang_ix` (language pair index) can be `0` (en-es), `1` (es-fr), `2` (en-fr), `3` (de-en), `4` (de-fr)
- `dataset` location of the dataset used

This script construct all folders for self-training(e.g. `data/run_nmt/ssl/all_50_50_en-es_ssl1`, `data/run_nmt/ssl/all_50_50_es-en_ssl1`) and back-translation(e.g. `data/run_nmt/ssl/all_50_50_en-es_ssl4`, `data/run_nmt/ssl/all_50_50_es-en_ssl4`). The labeled data in semi-supervised experiments are all 50%:50% mixed by causal and anti-causal data

Then, run ST and BT through following commands
```
# SSL by Self-training, supervised data (50% en->es) : (50% es->en), unlabeled data 100% en->es(only English side), translate from en to es
bash code/script_ssl_st_ratio.sh 0 50_50 0 1 0,1,2,3,4,5,6,7
```
Here, `bash code/script_ssl_st_ratio.sh {lang_ix} {directory} {mt_dir} {hyp} {CUDA_VISIBLE_DEVICES}`
Self-training can be tested through
```
bash code/st_test.sh 0 50-50 0 2 1 0
```
Here, `bash code/st_test.sh {lang_ix} {directory} {mt_dir} {if_ssl} {hyp} {CUDA_VISIBLE_DEVICES}`
```
# SSL by Back-translation, supervised data (50% en->es) : (50% es->en), unlabeled data 100% es->en(only Spanish side), translate from en to es
bash code/script_ssl_bt_ratio.sh 0 50_50 0 4 1 0,1,2,3,4,5,6,7
```
Here, `bash code/script_ssl_bt_conf.sh {lang_ix} {directory} {mt_dir} {if_ssl} {hyp} {CUDA_VISIBLE_DEVICES}`
Back-translation can be tested through
```
bash code/bt_test.sh 0 50-50 0 4 1 0
```
Here, `bash code/bt_test.sh {lang_ix} {directory} {mt_dir} {if_ssl} {hyp} {CUDA_VISIBLE_DEVICES}`
