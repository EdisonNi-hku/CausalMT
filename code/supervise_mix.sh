#!/usr/bin/env bash
set -e

src_tgt_ix=${1:-0}
ratio_causal=${2:-0}
ratio_anti=${3:-0}
mt_dir=${4:-0}
if_ssl=${5:-0}

data_folder=${6:-0}

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
    first=$1
    second=$2
else
    tgt=$1
    src=$2
    first=$1
    second=$2
fi

ratio=${ratio_causal}_${ratio_anti}
src_tgt=${src}-${tgt}
folder=data/run_nmt/ssl/all_${ratio}_${src_tgt}_ssl${if_ssl}
bpe=$folder/bpe

if [ ! -d $folder ];then
    mkdir $folder
    mkdir $bpe
fi

len_causal_train=`wc -l ${data_folder}/${first}-${second}/train.${src} | cut -d ' ' -f 1`
len_causal_train=`echo "scale=6;${len_causal_train} * ${ratio_causal} / 100" | bc`
len_causal_train=${len_causal_train%.*}

len_anti_train=`wc -l ${data_folder}/${second}-${first}/train.${src} | cut -d ' ' -f 1`
len_anti_train=`echo "scale=6;${len_anti_train} * ${ratio_anti} / 100" | bc`
len_anti_train=${len_anti_train%.*}

touch $bpe/train.${src}
head -n ${len_causal_train} ${data_folder}/${first}-${second}/train.${src} >> $bpe/train.${src}
head -n ${len_anti_train} ${data_folder}/${second}-${first}/train.${src} >> $bpe/train.${src}
touch $bpe/train.${tgt}
head -n ${len_causal_train} ${data_folder}/${first}-${second}/train.${tgt} >> $bpe/train.${tgt}
head -n ${len_anti_train} ${data_folder}/${second}-${first}/train.${tgt} >> $bpe/train.${tgt}

SCRIPTS=code/tools/mosesdecoder/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

#perl $CLEAN -ratio 1.5 ${bpe}/sup.train ${src} ${tgt} ${bpe}/train 1 250

echo "len_causal_train: $len_causal_train"
echo "len_anti_train: $len_anti_train"
len_causal_valid=`wc -l ${data_folder}/${first}-${second}/valid.${src} | cut -d ' ' -f 1`
len_causal_valid=`echo "scale=6;${len_causal_valid} * ${ratio_causal} / 100" | bc`
len_causal_valid=${len_causal_valid%.*}

len_anti_valid=`wc -l ${data_folder}/${second}-${first}/valid.${src} | cut -d ' ' -f 1`
len_anti_valid=`echo "scale=6;${len_anti_valid} * ${ratio_anti} / 100" | bc`
len_anti_valid=${len_anti_valid%.*}

echo "len_causal_valid: $len_causal_valid"
echo "len_anti_valid: $len_anti_valid"
touch $bpe/valid.${src}
head -n ${len_causal_valid} ${data_folder}/${first}-${second}/valid.${src} >> $bpe/valid.${src}
head -n ${len_anti_valid} ${data_folder}/${second}-${first}/valid.${src} >> $bpe/valid.${src}
touch $bpe/valid.${tgt}
head -n ${len_causal_valid} ${data_folder}/${first}-${second}/valid.${tgt} >> $bpe/valid.${tgt}
head -n ${len_anti_valid} ${data_folder}/${second}-${first}/valid.${tgt} >> $bpe/valid.${tgt}

#cp ${data_folder}/unsup.${tgt}-${src}.${tgt}.tok.bpe ${bpe}/unsup.${tgt}
cp ${data_folder}/${first}-${second}/test.${src} ${bpe}/cau_test.${src}
cp ${data_folder}/${first}-${second}/test.${tgt} ${bpe}/cau_test.${tgt}
cp ${data_folder}/${second}-${first}/test.${src} ${bpe}/ant_test.${src}
cp ${data_folder}/${second}-${first}/test.${tgt} ${bpe}/ant_test.${tgt} 

