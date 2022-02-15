#!/usr/bin/env bash
set -e

src_tgt_ix=${1:-0}
data_folder=${2:-0}

SRC_TGTS=(
    "en es"
    "es fr"
    "en fr"
    "de en"
    "de fr"
)

src_tgt=${SRC_TGTS[$src_tgt_ix]}
set -- $src_tgt
first=$1
second=$2

ant_dir=${second}-${frist}
cau_dir=${frist}-${second}

src_tgt=${src}-${tgt}
st_folder1=data/run_nmt/ssl/all_50-50_${first}-${second}_ssl1
st_folder2=data/run_nmt/ssl/all_50-50_${second}-${first}_ssl1
bt_folder1=data/run_nmt/ssl/all_50-50_${first}-${second}_ssl4
bt_folder2=data/run_nmt/ssl/all_50-50_${second}-${first}_ssl4
st_bpe1=$st_folder1/bpe
bt_bpe1=$bt_folder1/bpe
st_bpe2=$st_folder2/bpe
bt_bpe2=$bt_folder2/bpe

if [ ! -d $folder ];then
    mkdir $st_folder1
    mkdir $st_bpe1
fi

half_train=`wc -l ${data_folder}/${cau_dir}/train.${first} | cut -d ' ' -f 1`
half_train=`echo "scale=6;${half_train} * 50 / 100" | bc`
half_train=${half_train%.*}

half_valid1=`wc -l ${data_folder}/${cau_dir}/valid.${first} | cut -d ' ' -f 1`
half_valid1=`echo "scale=6;${half_valid1} * 50 / 100" | bc`
half_valid1=${half_valid1%.*}

half_valid2=`wc -l ${data_folder}/${ant_dir}/valid.${first} | cut -d ' ' -f 1`
half_valid2=`echo "scale=6;${half_valid2} * 50 / 100" | bc`
half_valid2=${half_valid2%.*}

quater_train=`wc -l ${data_folder}/${cau_dir}/train.${first} | cut -d ' ' -f 1`
quater_train=`echo "scale=6;${quater_train} * 25 / 100" | bc`
quater_train=${quater_train%.*}

# mix supervised data (causal : anticausal = 50 : 50)
touch $st_bpe1/train.${first}
head -n ${quater_train} ${data_folder}/${cau_dir}/train.${first} >> $st_bpe1/train.${first}
head -n ${quater_train} ${data_folder}/${ant_dir}/train.${first} >> $st_bpe1/train.${first}
touch $st_bpe1/train.${second}
head -n ${quater_train} ${data_folder}/${cau_dir}/train.${second} >> $st_bpe1/train.${second}
head -n ${quater_train} ${data_folder}/${ant_dir}/train.${second} >> $st_bpe1/train.${second}

touch $st_bpe1/valid.${first}
head -n ${half_valid1} ${data_folder}/${cau_dir}/valid.${first} >> $st_bpe1/valid.${first}
head -n ${half_valid2} ${data_folder}/${ant_dir}/valid.${first} >> $st_bpe1/valid.${first}
touch $st_bpe1/valid.${second}
head -n ${half_valid1} ${data_folder}/${cau_dir}/valid.${second} >> $st_bpe1/valid.${second}
head -n ${half_valid2} ${data_folder}/${ant_dir}/valid.${second} >> $st_bpe1/valid.${second}

cp ${data_folder}/${first}-${second}/test.${src} ${st_bpe1}/cau_test.${src}
cp ${data_folder}/${first}-${second}/test.${tgt} ${st_bpe1}/cau_test.${tgt}
cp ${data_folder}/${second}-${first}/test.${src} ${st_bpe1}/ant_test.${src}
cp ${data_folder}/${second}-${first}/test.${tgt} ${st_bpe1}/ant_test.${tgt}

# supervised data are the same for all ssl experiments
cp -r $st_folder1 $st_folder2
cp -r $st_folder1 $bt_folder1
cp -r $st_folder1 $bt_folder2

# prepare unsupervised data
tail -n ${half_train} ${data_folder}/${cau_dir}/train.${first} > $st_bpe1/unsup.${first}
tail -n ${half_train} ${data_folder}/${ant_dir}/train.${second} > $st_bpe2/unsup.${second}
tail -n ${half_train} ${data_folder}/${cau_dir}/train.${first} > $bt_bpe2/unsup.${first}
tail -n ${half_train} ${data_folder}/${ant_dir}/train.${second} > $bt_bpe1/unsup.${second}


