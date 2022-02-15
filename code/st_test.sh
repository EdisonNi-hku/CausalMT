#!/usr/bin/env bash
set -e

src_tgt_ix=${1:-0}
ratio=${2:-0}
mt_dir=${3:-0}
if_ssl=${4:-0}
hypo=${5:-0}
gpu=${6:-0}
SRC_TGTS=(
    "en es"
    "es fr"
    "en fr"
    "de en"
    "de fr"
)

src_tgt=${SRC_TGTS[$src_tgt_ix]}
set -- $src_tgt
if [ $mt_dir == 0 ]; then
    src=$1
    tgt=$2
else
    tgt=$1
    src=$2
fi

src_tgt=${src}-${tgt}
folder=data/run_nmt/ssl/all_${ratio}_${src_tgt}_ssl${if_ssl}
bpe=$folder/bpe
ckpt=$folder/ckpt_arch0_hyp${hypo}_ep1000
forward=$ckpt/iter0.model
final=$ckpt/iter1.model.finetune

cmd="CUDA_VISIBLE_DEVICES=$gpu fairseq-preprocess \
    --source-lang $src --target-lang $tgt \
    --joined-dictionary \
    --srcdict $folder/data-bin/dict.${src}.txt
    --testpref $bpe/cau_test \
    --destdir $folder/data-bin/cau_t --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20"
#echo $cmd
eval $cmd

cmd1="CUDA_VISIBLE_DEVICES=$gpu fairseq-preprocess \
    --source-lang $src --target-lang $tgt \
    --joined-dictionary \
    --srcdict $folder/data-bin/dict.${src}.txt \
    --testpref $bpe/ant_test \
    --destdir $folder/data-bin/ant_t --thresholdtgt 0 --thresholdsrc 0 --workers 20"
#echo $cmd1
eval $cmd1
CUDA_VISIBLE_DEVICES=$gpu python code/tools/fairseq/fairseq_cli/generate.py \
    $folder/data-bin/cau_t \
    --source-lang $src --target-lang $tgt \
    --path $forward/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > $forward/hypothesis_cau.txt

CUDA_VISIBLE_DEVICES=$gpu python code/tools/fairseq/fairseq_cli/generate.py \
    $folder/data-bin/ant_t \
    --source-lang $src --target-lang $tgt \
    --path $forward/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > $forward/hypothesis_ant.txt

CUDA_VISIBLE_DEVICES=$gpu python code/tools/fairseq/fairseq_cli/generate.py \
    $folder/data-bin/cau_t \
    --source-lang $src --target-lang $tgt \
    --path $final/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > $final/hypothesis_cau.txt

CUDA_VISIBLE_DEVICES=$gpu python code/tools/fairseq/fairseq_cli/generate.py \
    $folder/data-bin/ant_t \
    --source-lang $src --target-lang $tgt \
    --path $final/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > $final/hypothesis_ant.txt

echo "forward_cau"
tail $forward/hypothesis_cau.txt
echo "forward_ant"
tail $forward/hypothesis_ant.txt
echo "final_cau"
tail $final/hypothesis_cau.txt
echo "final_ant"
tail $final/hypothesis_ant.txt
