#!/usr/bin/env bash
set -e

src_tgt_ix=${1:-0}
ratio=${2:-0}
mt_dir=${3:-0}
if_ssl=${4:-0}
tune_hyper=${5:-0}
gpu=${6:-0}

SRC_TGTS=(
    "en es"
    "es fr"
    "en fr"
    "en pt"
    "es pt"
    "en pl"
    "bn en"
    "bn es"
    "de en"
    "cs en"
    "ar en"
    "fr pt"
    "en hu"
    "ar es"
    "en ru"
    "es ru"
    "fr it"
    "en sv"
    "ar fr"
    "fr ru"
    "en it"
    "es it"
    "es jp"
    "en fi"
    "en jp"
    "el en"
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

forward_continue_train=0
reverse_continue_train=0
src_tgt=${src}-${tgt}
folder=data/run_nmt/ssl/all_${ratio}_${src_tgt}_ssl${if_ssl}
forward_folder_ckpt=$folder/ckpt_bt_hyp${tune_hyper}_forward
reverse_folder_ckpt=$folder/ckpt_bt_hyp${tune_hyper}_reverse

echo "GPU$gpu: $forward_folder_ckpt, tune_hyper=$tune_hyper"

# Binarize the dataset
if [ ! -f $folder/data-bin/parallel/train.${src_tgt}.${src}.bin ]; then
    rm $folder/data-bin/parallel/* 2>/dev/null || true
    # Parallel data
    CUDA_VISIBLE_DEVICES=$gpu fairseq-preprocess \
        --joined-dictionary \
        --source-lang $src --target-lang $tgt \
        --trainpref $folder/bpe/train --validpref $folder/bpe/valid --testpref $folder/bpe/cau_test \
        --destdir $folder/data-bin/parallel --thresholdtgt 0 --thresholdsrc 0 \
        --workers 20
fi

if [ $forward_continue_train == 1 ]; then
    cont_arg="--restore-file $forward_folder_ckpt/checkpoint_last.pt "
    cont_tee="-a "
else
    rm $forward_folder_ckpt/* 2>/dev/null || true
    mkdir $forward_folder_ckpt/ 2>/dev/null || true
fi
cmd="
CUDA_VISIBLE_DEVICES=$gpu python code/tools/fairseq/train.py --fp16 \
    $folder/data-bin/parallel \
    --source-lang $src --target-lang $tgt \
    --arch transformer --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 4096 --update-freq 8 \
    --warmup-init-lr '1e-07' --min-lr '1e-09' \
    --encoder-layers 6 --decoder-layers 6 \
    --max-epoch 1000 \
    --eval-bleu \
    --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --no-epoch-checkpoints \
    --save-dir $forward_folder_ckpt $cont_arg \
    --log-format=json 2>&1 | tee $cont_tee $forward_folder_ckpt/train.log
"

echo "$cmd" 2>&1 | tee -a $forward_folder_ckpt/train.log
eval $cmd

chmod u-w $forward_folder_ckpt/train.log

# Evaluate forward model
CUDA_VISIBLE_DEVICES=$gpu python code/tools/fairseq/fairseq_cli/generate.py \
    --source-lang $src --target-lang $tgt \
    $folder/data-bin/parallel \
    --path $forward_folder_ckpt/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > $forward_folder_ckpt/hypothesis.txt

cat $forward_folder_ckpt/hypothesis.txt

echo "Test set translation outputs saved to $forward_folder_ckpt/hypothesis.txt"

# Train reverse model
if [ $reverse_continue_train == 1 ]; then
    cont_arg="--restore-file $reverse_folder_ckpt/checkpoint_last.pt "
    cont_tee="-a "
else
    rm $reverse_folder_ckpt/* 2>/dev/null || true
    mkdir $reverse_folder_ckpt/ 2>/dev/null || true
fi

cmd_r="
CUDA_VISIBLE_DEVICES=$gpu python code/tools/fairseq/train.py --fp16 \
    $folder/data-bin/parallel \
    --source-lang $tgt --target-lang $src \
    --arch transformer --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 4096 --update-freq 8 \
    --warmup-init-lr '1e-07' --min-lr '1e-09' \
    --encoder-layers 6 --decoder-layers 6 \
    --max-epoch 1000 \
    --eval-bleu \
    --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --no-epoch-checkpoints \
    --save-dir $reverse_folder_ckpt $cont_arg \
    --log-format=json 2>&1 | tee $cont_tee $reverse_folder_ckpt/train.log
"

echo "$cmd_r" 2>&1 | tee -a $reverse_folder_ckpt/train.log
eval $cmd_r

chmod u-w $reverse_folder_ckpt/train.log

# Evaluate reverse model
CUDA_VISIBLE_DEVICES=$gpu python code/tools/fairseq/fairseq_cli/generate.py \
    $folder/data-bin/parallel \
    --source-lang $tgt --target-lang $src \
    --path $reverse_folder_ckpt/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > $reverse_folder_ckpt/hypothesis.txt

cat $reverse_folder_ckpt/hypothesis.txt

echo "Test set translation outputs saved to $reverse_folder_ckpt/hypothesis.txt"

# Monolingual data
CUDA_VISIBLE_DEVICES=$gpu fairseq-preprocess \
    --source-lang $tgt --target-lang $src \
    --joined-dictionary \
    --testpref $folder/bpe/unsup \
    --destdir $folder/data-bin/mono --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20 \
    --only-source \
    --srcdict $folder/data-bin/parallel/dict.$tgt.txt

if [ ! -f $folder/data-bin/mono/dict.$src.txt ]; then
    cp $folder/data-bin/parallel/dict.$src.txt $folder/data-bin/mono
fi
# Generate pseudo parallel data
bt_out=$folder/bt_out_${tune_hyper}
mkdir $bt_out
CUDA_VISIBLE_DEVICES=$gpu fairseq-generate --fp16 \
    $folder/data-bin/mono \
    --path $reverse_folder_ckpt/checkpoint_best.pt \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 4096 \
    --sampling --beam 1 > $bt_out/sampling.out; \

python code/tools/fairseq/examples/backtranslation/extract_bt_data.py \
    --minlen 1 --maxlen 250 --ratio 1.5 \
    --output $bt_out/bt_data --srclang $src --tgtlang $tgt \
    $bt_out/sampling.out

# Binarize the filtered BT data and combine it with the parallel data:
fairseq-preprocess \
    --source-lang $src --target-lang $tgt \
    --joined-dictionary \
    --srcdict $folder/data-bin/parallel/dict.${src}.txt \
    --trainpref $bt_out/bt_data \
    --destdir $folder/data-bin/bt_${tune_hyper} \
    --workers 20

# Combine the bt data and parallel data
PARA_DATA=$(readlink -f $folder/data-bin/parallel)
BT_DATA=$(readlink -f $folder/data-bin/bt_${tune_hyper})
COMB_DATA=$folder/data-bin/para_plus_bt_${tune_hyper}
mkdir -p $COMB_DATA
for LANG in $src $tgt; do \
    ln -s ${PARA_DATA}/dict.$LANG.txt ${COMB_DATA}/dict.$LANG.txt; \
    for EXT in bin idx; do \
        ln -s ${PARA_DATA}/train.${src_tgt}.$LANG.$EXT ${COMB_DATA}/train.${src_tgt}.$LANG.$EXT; \
        ln -s ${BT_DATA}/train.${src_tgt}.$LANG.$EXT ${COMB_DATA}/train1.${src_tgt}.$LANG.$EXT; \
        ln -s ${PARA_DATA}/valid.${src_tgt}.$LANG.$EXT ${COMB_DATA}/valid.${src_tgt}.$LANG.$EXT; \
        ln -s ${PARA_DATA}/test.${src_tgt}.$LANG.$EXT ${COMB_DATA}/test.${src_tgt}.$LANG.$EXT; \
    done; \
done

# Train the final model
final_folder_ckpt=$folder/ckpt_bt_hyp${tune_hyper}_final
final_continue_train=0

if [ $final_continue_train == 1 ]; then
    cont_arg="--restore-file $final_folder_ckpt/checkpoint_last.pt "
    cont_arg="--restore-file $final_folder_ckpt/checkpoint_last.pt "
    cont_arg="--restore-file $final_folder_ckpt/checkpoint_last.pt "
    cont_tee="-a "
else
    rm $final_folder_ckpt/* 2>/dev/null || true
    mkdir $final_folder_ckpt/ 2>/dev/null || true
fi

cmd="
CUDA_VISIBLE_DEVICES=$gpu fairseq-train --fp16 \
    $folder/data-bin/para_plus_bt_${tune_hyper} \
    --upsample-primary 16 \
    --source-lang $src --target-lang $tgt \
    --arch transformer --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --warmup-init-lr '1e-07' --min-lr '1e-09' \
    --encoder-layers 6 --decoder-layers 6 \
    --max-tokens 4096 --update-freq 8 \
    --max-epoch 500 \
    --eval-bleu \
    --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --no-epoch-checkpoints \
    --save-dir $final_folder_ckpt $cont_arg \
    --log-format=json 2>&1 | tee $cont_tee $final_folder_ckpt/train.log
"
echo "$cmd" 2>&1 | tee -a $final_folder_ckpt/train.log
eval $cmd

chmod u-w $final_folder_ckpt/train.log

CUDA_VISIBLE_DEVICES=$gpu python code/tools/fairseq/fairseq_cli/generate.py \
    $folder/data-bin/parallel \
    --source-lang $src --target-lang $tgt \
    --path $final_folder_ckpt/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > $final_folder_ckpt/hypothesis.txt

