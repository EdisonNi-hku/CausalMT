set -e
#trap "exit 1" ABRT TERM


src_tgt_ix=${1:-0}
ratio=${2:-0}
mt_dir=${3:-0}
#num=$((4 + $src_tgt_ix*4 + $cau_dir*2 + $mt_dir*1))
hyp=${4:-0}
gpu=${5:-0}
onec=${6:-0}
twoc=${7:-0}
thrc=${8:-0}
#if [ $num == 5 ]; then
#    batch_size=32
#else
    batch_size=128
#fi


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

#cau_eff=${cau}-${eff}
src_tgt=${src}-${tgt}

src_bt=${src}_bt
tgt_bt=${tgt}_bt


prep_data_from_ssl1=2
prep_noising=1
train_iter0=1
backtranslate=1
train_pseudo=1
train_finetune=1



debug=0

if [ $debug = 1 ]; then
    train_steps=10
    finetune_epoch=10
    pretrain_steps=10
    gpu=7
else
    pretrain_epochs=1000
    train_epochs=1000
    finetune_epoch=1000
fi

model=transformer
#model=transformer_wmt_en_de
dropout=0.3


# Prepare the data folder for self training by copying from the DAE folder
#FOLDER0=data/run_nmt/ssl/all_${ratio}_${src_tgt}_ssl0
FOLDER1=data/run_nmt/ssl/all_${ratio}_${src_tgt}_ssl1
FOLDER2=data/run_nmt/ssl/all_${ratio}_${src_tgt}_ssl2
<<!
read -p "[cuda=${gpu}] Language pair is ${FOLDER2}, prep_data_from_ssl1=${prep_data_from_ssl1}, prep_noising=${prep_noising}, train_iter0=${train_iter0}, backtranslate=${backtranslate}, train_pseudo=${train_pseudo}, train_finetune=${train_finetune}. Are you sure? (y/n)" yn
case $yn in
    [Yy]* ) ;;
    [Nn]* ) exit;;
    * ) echo "Please answer yes or no.";;
esac
!


if [ $prep_data_from_ssl1 = 2 ]; then
    # Binarize the dataset
    if [ ! -f $FOLDER1/data-bin/train.${src_tgt}.${src}.bin ]; then
        rm $FOLDER1/data-bin/* 2>/dev/null || true
        CUDA_VISIBLE_DEVICES=$gpu fairseq-preprocess \
            --joined-dictionary \
            --source-lang $src --target-lang $tgt \
            --trainpref $FOLDER1/bpe/train --validpref $FOLDER1/bpe/valid --testpref $FOLDER1/bpe/cau_test \
            --destdir $FOLDER1/data-bin --thresholdtgt 0 --thresholdsrc 0 \
            --workers 20
    fi
fi

if [ $prep_data_from_ssl1 -ge 1 ]; then
    mkdir -p $FOLDER2/bpe/ $FOLDER2/ckpt_arch0_hyp0_ep1000/  || true
    cp $FOLDER1/bpe/unsup.${src} $FOLDER2/bpe/unsup.${src}
    cp -a $FOLDER1/data-bin/ $FOLDER2/
fi

BPE_DIR=$FOLDER2/bpe
BIN_DIR=$FOLDER2/data-bin
CKPT_DIR=$FOLDER2/ckpt_arch0_hyp${hyp}_ep1000

# Add noise to the monolingual corpus for later usage:
if [ $prep_noising = 1 ]; then
    python code/tools/he2020selftraining/paraphrase/paraphrase.py \
      --paraphraze-fn noise_bpe \
      --word-dropout 0.2 \
      --word-blank 0.2 \
      --word-shuffle 3 \
      --data-file ${BPE_DIR}/unsup.${src} \
      --output ${BPE_DIR}/unsup_noise.${src} \
      --bpe-type subword
fi

# Train the translation model with 30K updates:
# ---- supervised training ---- #

SAVE=${CKPT_DIR}/iter0.model
if [ $onec = 1 ]; then
    mk=""
    oneload="--restore-file ${CKPT_DIR}/iter0.model/checkpoint_last.pt"
else
    mk="mkdir -p ${SAVE}"
fi

if [ $train_iter0 = 1 ]; then
    eval $mk
    cmd="
    CUDA_VISIBLE_DEVICES=${gpu} fairseq-train $BIN_DIR \
         --max-epoch ${pretrain_epochs} \
         -a ${model} -s ${src} -t ${tgt} --fp16 \
         --task translation \
         --share-all-embeddings \
         --optimizer adam --adam-betas '(0.9, 0.98)' \
         --lr 5e-4 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
         --warmup-updates 4000 --warmup-init-lr '1e-07' \
         --dropout ${dropout} --weight-decay 0.0001 \
         --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
         --max-tokens 4096 \
         --encoder-layers 6 --decoder-layers 6 \
         --eval-bleu \
         --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
         --eval-bleu-detok moses \
         --eval-bleu-remove-bpe \
         --eval-bleu-print-samples \
         --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
         --no-epoch-checkpoints \
         --save-dir ${SAVE} $oneload \
         --update-freq 8 \
         --log-format=json 2>&1 | tee -a ${SAVE}/train.log
    "
    echo "$cmd" 2>&1 | tee -a ${SAVE}/train.log
    eval $cmd

   # wait $!

    # ---- check performance ---- #
    cmd="CUDA_VISIBLE_DEVICES=${gpu} fairseq-generate $BIN_DIR --source-lang ${src} --target-lang ${tgt} \
        --path ${SAVE}/checkpoint_best.pt --beam 5 --batch-size ${batch_size} --remove-bpe \
        2>&1 | tee -a ${SAVE}/hypothesis.txt
    "
    echo "$cmd"
    eval $cmd
   # wait $!
fi


# ---- back translate ---- #
if [ $backtranslate = 1 ]; then
    model_path=${SAVE}/checkpoint_best.pt
#    model_path=$FOLDER1/ckpt_arch0_hyp0_ep1000/checkpoint_best.pt
    bt_file=$CKPT_DIR/unsup_bt.${tgt}

    cmd="cat ${BPE_DIR}/unsup.${src} | \
        CUDA_VISIBLE_DEVICES=${gpu} fairseq-interactive $BIN_DIR \
        --path ${model_path} \
        --beam 5 --fp16 --batch-size 100 \
        --buffer-size 100 > $bt_file
    "
    echo "$cmd"
    eval $cmd

    grep ^H $bt_file | cut -f3- > ${BPE_DIR}/unsup_bt.${tgt_bt}
    rm $bt_file

    # preprocess the pseudo parallel corpus
    cp ${BPE_DIR}/unsup_noise.${src} ${BPE_DIR}/unsup_bt.${src_bt}

    fairseq-preprocess --source-lang ${src_bt} --target-lang ${tgt_bt} \
           --trainpref ${BPE_DIR}/unsup_bt --srcdict ${BIN_DIR}/dict.${src}.txt \
           --tgtdict ${BIN_DIR}/dict.${tgt}.txt \
           --destdir ${BIN_DIR} --workers 16


    cd ${BIN_DIR}
    cp valid.${src_tgt}.${src}.idx valid.${src_bt}-${tgt_bt}.${src_bt}.idx
    cp valid.${src_tgt}.${src}.bin valid.${src_bt}-${tgt_bt}.${src_bt}.bin
    cp valid.${src_tgt}.${tgt}.idx valid.${src_bt}-${tgt_bt}.${tgt_bt}.idx
    cp valid.${src_tgt}.${tgt}.bin valid.${src_bt}-${tgt_bt}.${tgt_bt}.bin

    cd ~-
fi




# self training

DIR_PSEUDO_TRAIN=${CKPT_DIR}/iter1.model
if [ $twoc = 1 ]; then
    mktwo=""
    twoload="--restore-file ${CKPT_DIR}/iter1.model/checkpoint_last.pt"
else
    mktwo="mkdir -p ${DIR_PSEUDO_TRAIN}"
fi

# ---- pseudo training ---- #
if [ $train_pseudo = 1 ]; then
    eval $mktwo 

    cmd="
    CUDA_VISIBLE_DEVICES=${gpu} fairseq-train $BIN_DIR \
         --max-epoch ${train_epochs} \
         -a ${model} -s ${src_bt} -t ${tgt_bt} \
         --task translation \
         --share-all-embeddings --fp16 \
         --optimizer adam --adam-betas '(0.9, 0.98)' \
         --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
         --warmup-updates 4000 --warmup-init-lr '1e-07' \
         --dropout ${dropout} --weight-decay 0.0001 \
         --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
         --max-tokens 4096 \
         --encoder-layers 6 --decoder-layers 6 \
         --eval-bleu \
         --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
         --eval-bleu-detok moses \
         --eval-bleu-remove-bpe \
         --eval-bleu-print-samples \
         --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
         --no-epoch-checkpoints \
         --save-dir ${DIR_PSEUDO_TRAIN} $twoload \
         --update-freq 8 \
         --log-format=json 2>&1 | tee -a ${DIR_PSEUDO_TRAIN}/train.log
    "
        echo "$cmd" 2>&1 | tee -a ${DIR_PSEUDO_TRAIN}/train.log
        eval $cmd
   # wait $!

    # ---- check performance ---- #
    cmd="CUDA_VISIBLE_DEVICES=${gpu} fairseq-generate $BIN_DIR --source-lang ${src} --target-lang ${tgt} \
        --path ${DIR_PSEUDO_TRAIN}/checkpoint_best.pt --beam 5 --batch-size ${batch_size} --remove-bpe \
        2>&1 | tee -a ${DIR_PSEUDO_TRAIN}/hypothesis.txt
    "
    echo "$cmd"
    eval $cmd
   # wait $!
fi


# ---- fine-tuning ---- #
DIR_FINETUNE=${CKPT_DIR}/iter1.model.finetune
if [ $thrc = 1 ]; then
    mkthr=""
    thrload="--restore-file ${CKPT_DIR}/iter1.model.finetune/checkpoint_last.pt"
    reset=""
    cp=""
else
    mkthr="mkdir -p ${DIR_FINETUNE} "
    cp="cp ${DIR_PSEUDO_TRAIN}/checkpoint_best.pt ${DIR_FINETUNE}/checkpoint_load.pt"
    thrload="--restore-file ${CKPT_DIR}/iter1.model.finetune/checkpoint_load.pt"
    reset="--reset-optimizer --reset-meters --reset-dataloader"
fi

if [ $train_finetune = 1 ]; then
    eval $mkthr
    eval $cp

    #cp ${DIR_PSEUDO_TRAIN}/checkpoint_best.pt ${DIR_FINETUNE}/checkpoint_load.pt
    cmd="CUDA_VISIBLE_DEVICES=${gpu} fairseq-train $BIN_DIR \
         --max-epoch ${finetune_epoch} \
         -a ${model} -s ${src} -t ${tgt} \
         --task translation \
         --share-all-embeddings --fp16 \
         --optimizer adam --adam-betas '(0.9, 0.98)' \
         --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
         --warmup-updates 4000 --warmup-init-lr '1e-07' \
         --dropout ${dropout} --weight-decay 0.0001 \
         --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
         --max-tokens 4096 \
         --encoder-layers 6 --decoder-layers 6 \
         --eval-bleu \
         --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
         --eval-bleu-detok moses \
         --eval-bleu-remove-bpe \
         --eval-bleu-print-samples \
         --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
         --no-epoch-checkpoints \
         --save-dir ${DIR_FINETUNE} $thrload\
         $reset \
         --update-freq 8 \
         --log-format=json 2>&1 | tee -a ${DIR_FINETUNE}/train.log
    "
    echo "$cmd" 2>&1 | tee -a ${DIR_PSEUDO_TRAIN}/train.log
    eval $cmd
   # wait $!
fi


# ---- check performance ---- #

cmd="CUDA_VISIBLE_DEVICES=${gpu} fairseq-generate $BIN_DIR --source-lang ${src} --target-lang ${tgt} \
    --path ${DIR_FINETUNE}/checkpoint_best.pt --beam 5 --batch-size ${batch_size} --remove-bpe \
    2>&1 | tee -a ${DIR_FINETUNE}/hypothesis.txt
"
echo "$cmd"
eval $cmd
#wait $!



