This repo contains the codes of Causality for Machine Translation.

- `code/` stores the scripts to run NMT models.
- `data/` stores the code to generate MT data between cause and effect languages.

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

```
### How to Run NMT Models

To run the spanish-french samples for (supervised MT, Back-translation, Self-training), execute the following command.
```
# Supervised MT, translate from fr to es
bash code/script_ssl_bt_conf.sh 1 fr-es 1 0 1 0,1,2,3,4,5,6,7

# SSL by Self-training, supervised data fr-es, translate from fr to es
bash code/script_ssl_st_ratio.sh 1 fr-es 1 1 0,1,2,3,4,5,6,7

# SSL by Back-translation, supervised data fr-es, translate from es to fr
bash code/script_ssl_bt_ratio.sh 1 fr-es 0 4 1 0,1,2,3,4,5,6,7
```