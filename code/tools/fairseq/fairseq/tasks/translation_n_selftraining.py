# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import OrderedDict, defaultdict
from argparse import Namespace
import json
import itertools
import logging
import os
import torch
from fairseq import options
import numpy as np

from fairseq import metrics, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    RoundRobinZipDatasets,
)

from fairseq.data.reversed_backtranslation_dataset import PseudoLabelDataset

from fairseq.models import FairseqMultiModel
from fairseq.tasks import register_task, LegacyFairseqTask

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


# ported from UnsupervisedMT
def parse_lambda_config(x):
    """
    Parse the configuration of lambda coefficient (for scheduling).
    x = "3"                  # lambda will be a constant equal to x
    x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease
                             # to 0 during the first 1000 iterations
    x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000
                             # iterations, then will linearly increase to 1 until iteration 2000
    """
    split = x.split(',')
    if len(split) == 1:
        return float(x), None
    else:
        split = [s.split(os.pathsep) for s in split]
        assert all(len(s) == 2 for s in split)
        assert all(k.isdigit() for k, _ in split)
        assert all(int(split[i][0]) < int(split[i + 1][0]) for i in
                   range(len(split) - 1))
        return float(split[0][1]), [(int(k), float(v)) for k, v in split]


def load_langpair_dataset(
        data_path, split,
        src, src_dict,
        tgt, tgt_dict,
        combine, dataset_impl, upsample_primary,
        left_pad_source, left_pad_target, max_source_positions,
        max_target_positions, prepend_bos=False, load_alignments=False,
        truncate_source=False, append_source_id=False,
        num_buckets=0,
        shuffle=True,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path,
                                '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path,
                                  '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path,
                                  '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    'Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict,
                                                      dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict,
                                                      dataset_impl)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset,
                                         src_dict.index('[{}]'.format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset,
                                             tgt_dict.index('[{}]'.format(tgt)))
        eos = tgt_dict.index('[{}]'.format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path,
                                  '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None,
                                                            dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset, eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
    )


@register_task('translation_n_selftraining')
class TranslationNSelfTrainingTask(LegacyFairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner; \
                            however, valid and test data are always in the first directory to \
                            avoid the need for repeating them in all directories')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None,
                            metavar='TARGET',
                            help='target language')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str,
                            metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str,
                            metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int,
                            metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int,
                            metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true',
                            default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--num-batch-buckets', default=0, type=int,
                            metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

        # add self-training parameters to the parser
        parser.add_argument('--lambda-parallel-config', default="1.0", type=str,
                            metavar='CONFIG',
                            help='cross-entropy reconstruction coefficient (parallel data). '
                                 'use fixed weight during training if set to floating point number. '
                                 'use piecewise linear function over number of updates to schedule the '
                                 'weight with the format: w0:step0,w1:step1,...')
        parser.add_argument('--lambda-self-training-config', default="1.0",
                            type=str, metavar='CONFIG',
                            help='Cross-entropy reconstruction coefficient (self training)'
                                 'use fixed weight during training if set to floating point number. '
                                 'use piecewise linear function over number of updates to schedule the '
                                 'weight with the format: w0:step0,w1:step1,...')
        parser.add_argument('--lambda-confidence-threshold-config', default="0.5",
                            type=str, metavar='CONFIG',
                            help='Pseudo label with confidence score higher than this threshold will '
                                 'join the next training iteration.')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true',
                            default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ',
                            default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        parser.add_argument('--log-corpus', default='False', type=str,
                            metavar='BOOL',
                            help='whether to store generated label')
        parser.add_argument('--corpus-dir', type=str,
                            help='specify where to store the generated corpus, ')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        # self.dicts is for MultitaskTransformerModel.build_model()
        self.dicts = {args.source_lang: src_dict,
                      args.target_lang: tgt_dict, }

        self.lang_pair = '{}-{}'.format(args.source_lang, args.target_lang)
        self.model_lang_pairs = [self.lang_pair]
        self.log_hypos = True
        self.generate_progress = 0
        self.generate_size = 0
        self.valid_size = 0
        self.pseudo_set = None
        self.epoch = 0
        self.pseudo_ratio = 1
        self.log_corpus = args.log_corpus
        self.corpus_dir = args.corpus_dir

        self.lambda_parallel, self.lambda_parallel_steps = parse_lambda_config(
            args.lambda_parallel_config)
        self.lambda_self_training, self.lambda_self_training_steps = parse_lambda_config(
            args.lambda_self_training_config)
        self.lambda_confidence_threshold, self.lambda_confidence_threshold_steps = parse_lambda_config(
            args.lambda_confidence_threshold_config)
        if self.lambda_confidence_threshold < 0:
            self.lambda_self_training = 0.0

        if (
                self.lambda_self_training > 0.0 or self.lambda_self_training_steps is not None):
            self_training_pair = "%s-%s" % (args.source_lang, args.source_lang)
            self.self_training_pair = self_training_pair

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                'Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info(
            '[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info(
            '[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang
        lang_pair = self.lang_pair

        def split_exists(split, src, tgt, lang):
            filename = os.path.join(data_path,
                                    '{}.{}-{}.{}'.format(split, src, tgt,
                                                         lang))
            return indexed_dataset.dataset_exists(filename,
                                                  impl=self.args.dataset_impl)

        def load_indexed_dataset(path, dictionary):
            return data_utils.load_indexed_dataset(path, dictionary, self.args.dataset_impl)

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split('-')
            src_dataset, tgt_dataset = src_datasets[lang_pair], tgt_datasets[lang_pair]
            return LanguagePairDataset(
                src_dataset, src_dataset.sizes, self.dicts[src],
                tgt_dataset, tgt_dataset.sizes, self.dicts[tgt],
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
            )

        # load parallel datasets
        src_datasets, tgt_datasets = {}, {}
        if self.lambda_parallel > 0.0 or self.lambda_parallel_steps is not None:
            src, tgt = self.args.source_lang, self.args.target_lang
            if split_exists(split, src, tgt, src):
                prefix = os.path.join(data_path, '{}.{}-{}.'.format(split, src, tgt))
            src_datasets[lang_pair] = load_indexed_dataset(prefix + src, self.dicts[src])
            tgt_datasets[lang_pair] = load_indexed_dataset(prefix + tgt, self.dicts[tgt])
            logger.info('parallel-{} {} {} examples'.format(data_path, split, len(src_datasets[lang_pair])))
        key_n_dataset = [(lang_pair, language_pair_dataset(lang_pair))]
        if split == 'valid':
            self.valid_size = len(key_n_dataset[0][1])

        # load self-training pseudo dataset
        if (self.lambda_self_training > 0.0 or self.lambda_self_training_steps is not None) and split.startswith(
                "train"):
            src, tgt = self.args.source_lang, self.args.target_lang
            self.pseudo_set = PseudoLabelDataset(
                tgt_dict=self.dicts[tgt],
                src_dict=self.dicts[src],
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
            )
            self.pseudo_set.raw_data_growing.append(key_n_dataset[0][1][0])  # sentry element
            self.pseudo_set.sizes = np.array([list(key_n_dataset[0][1].size(0))], dtype=np.int64)
            logger.info('self_training-{}: {} {} {} examples'.format(
                src, data_path, split, len(self.pseudo_set),
            ))
            key_n_dataset.append(
                (self.self_training_pair, self.pseudo_set)
            )

        if (self.lambda_self_training > 0.0 or self.lambda_self_training_steps is not None) and split.startswith(
                "valid"):
            src, tgt = self.args.source_lang, self.args.target_lang
            filename = os.path.join(data_path,
                                    '{}.{}-None.{}'.format('train', src, src))
            src_dataset = load_indexed_dataset(filename, self.dicts[src])
            src_dataset_valid = LanguagePairDataset(src_dataset, src_dataset.sizes,
                                                    self.dicts[src],
                                                    tgt_dict=self.dicts[tgt],
                                                    left_pad_source=self.args.left_pad_source,
                                                    left_pad_target=self.args.left_pad_target, )

            key_n_dataset.append(('generate', src_dataset_valid))

        if split == 'test':
            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict(key_n_dataset),
                eval_key=self.lang_pair,
            )
        else:
            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict(key_n_dataset)
            )
            if split == 'valid':
                self.generate_size = len(self.dataset(split))

    def build_dataset_for_inference(self, src_tokens, src_lengths,
                                    constraints=None):
        return LanguagePairDataset(src_tokens, src_lengths,
                                   self.source_dictionary,
                                   tgt_dict=self.target_dictionary,
                                   constraints=constraints)

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        if not isinstance(model, FairseqMultiModel):
            raise ValueError(
                'SemisupervisedTranslationTask requires a FairseqMultiModel architecture')

        # model = super().build_model(args)

        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(
                getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args_beam5 = json.loads(getattr(args, 'eval_bleu_args', '{"beam": 1}') or '{"beam": 1}')
            self.sequence_generator_beam5 = self.build_generator([model], Namespace(
                **gen_args_beam5))

            gen_args_beam1 = json.loads('{"beam": 1, "max_len_a": 1.2, "max_len_b": 10}')
            self.sequence_generator_beam1 = self.build_generator([model], Namespace(
                **gen_args_beam1))

        return model

    def valid_step(self, sample, model, criterion):
        model.eval()
        if self.lambda_self_training > 0.0:
            if self.generate_progress == 0:
                self.epoch += 1
                src, tgt = self.args.source_lang, self.args.target_lang
                key_n_dataset = []
                key_n_dataset.append((self.lang_pair, self.datasets['train'].datasets[self.lang_pair]))
                self.pseudo_set = PseudoLabelDataset(
                    tgt_dict=self.dicts[tgt],
                    src_dict=self.dicts[src],
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                )
                self.pseudo_set.raw_data_growing.append(key_n_dataset[0][1][0])  # sentry element
                self.pseudo_set.sizes = np.array([list(key_n_dataset[0][1].size(0))], dtype=np.int64)
                key_n_dataset.append((self.self_training_pair, self.pseudo_set))
                self.datasets['train'] = RoundRobinZipDatasets(
                    OrderedDict(key_n_dataset)
                )

            paired_sample = sample[self.lang_pair]
            pseudo_sample = sample['generate']
            gen_out = self.inference_step(self.sequence_generator_beam1, [model], pseudo_sample,
                                          prefix_tokens=None)
            self.generate_progress += int(pseudo_sample['nsentences'])
            logger.info("Generating progress: " + str(self.generate_progress) + '/' + str(self.generate_size))

            high_confidence = []
            perplexities = []

            if self.args.eval_bleu_remove_bpe is not None:
                bpe_cont = self.args.eval_bleu_remove_bpe.rstrip()
                bpe_toks = {
                    i
                    for i in range(len(self.target_dictionary))
                    if self.source_dictionary[i].endswith(bpe_cont)
                }
            else:
                bpe_cont = None
                bpe_toks = None

            for id, srcs, lens, hypos in zip(
                    pseudo_sample['id'],
                    pseudo_sample['net_input']['src_tokens'],
                    pseudo_sample['net_input']['src_lengths'],
                    gen_out
            ):
                hypo = hypos[0]

                tokens = hypo['tokens']
                tgt_len = tokens.numel()
                pos_scores = hypo['positional_scores'].float()

                skipped_toks = 0
                if bpe_toks is not None:
                    for i in range(tgt_len - 1):
                        if tokens[i].item() in bpe_toks:
                            skipped_toks += 1
                            pos_scores[i + 1] += pos_scores[i]
                            pos_scores[i] = 0
                            
                inf_scores = pos_scores.eq(float('inf')) | pos_scores.eq(float('-inf'))
                if inf_scores.any():
                    pos_scores = pos_scores[(~inf_scores).nonzero()]

                count = pos_scores.numel() - skipped_toks
                ln_perplexity = -pos_scores.sum().cpu() / count
                perplexities.append(ln_perplexity)

                if ln_perplexity < self.lambda_confidence_threshold:
                    high_confidence.append(
                            {'id': id.item(), 'source': srcs.cpu(), 'source_length': lens.cpu(),
                             'target': hypo['tokens'].cpu(), 'target_length': tgt_len}
                    )

            logger.info("ln_perplexity mean: " + str(np.mean(perplexities)) + ", variance: " + str(np.var(perplexities)) 
                            + ", max: " + str(np.max(perplexities)) + ", min: " + str(np.min(perplexities)))
            logger.info("Instances with high confidence in this batch: " + str(len(high_confidence)))

            if len(high_confidence) > 0:
                s = self.tgt_dict.string(
                    high_confidence[0]['target'].int().cpu(),
                    self.args.eval_bleu_remove_bpe,
                    unk_string=(
                        "UNKNOWNTOKENINHYP"
                    ),
                )
                if self.tokenizer:
                    s = self.tokenizer.decode(s)
                logger.info("example pseudo label: " + s)

            if self.log_corpus and (self.epoch % 10 == 0 or self.epoch == 1):
                with open(os.path.join(self.corpus_dir, self.lang_pair + '_' + str(self.lambda_confidence_threshold) + '_'
                          + str(self.lambda_self_training), 'pseudo_corpus_' + str(self.epoch) + '.txt'), 'a+') as f:
                    for i in high_confidence:
                        s = self.tgt_dict.string(
                            i['target'].int().cpu(),
                            self.args.eval_bleu_remove_bpe,
                            unk_string=(
                                "UNKNOWNTOKENINHYP"
                            ),
                        )
                        if self.tokenizer:
                            s = self.tokenizer.decode(s)
                        f.write(s)
                        f.write('\n')
                        self.pseudo_set.update(i)
            else:
                for i in high_confidence:
                    self.pseudo_set.update(i)

            lang_pair = self.lang_pair
            if self.generate_progress == self.generate_size:
                key_n_dataset = []
                key_n_dataset.append((lang_pair, self.datasets['train'].datasets[lang_pair]))
                key_n_dataset.append((self.self_training_pair, self.pseudo_set))
                self.datasets['train'] = RoundRobinZipDatasets(
                    OrderedDict(key_n_dataset)
                )
        else:
            paired_sample = sample[self.lang_pair]

        lang_pair = self.lang_pair

        if self.generate_progress > self.valid_size:
            logging_output = {'loss': torch.tensor(0, device='cuda:0'), 'nll_loss': torch.tensor(0, device='cuda:0'),
             'ntokens': 0, 'nsentences': 0, 'sample_size': 0, '_bleu_sys_len': 0, '_bleu_ref_len': 0,
             '_bleu_counts_0': 0, '_bleu_totals_0': 0, '_bleu_counts_1': 0, '_bleu_totals_1': 0,
             '_bleu_counts_2': 0, '_bleu_totals_2': 0, '_bleu_counts_3': 0, '_bleu_totals_3': 0}
            loss = torch.tensor(0, device='cuda:0')
            sample_size = 0

        else:
            with torch.no_grad():
                loss, sample_size, logging_output = criterion(
                    model.models[lang_pair], paired_sample)

            if self.args.eval_bleu:
                bleu = self._inference_with_bleu(self.sequence_generator_beam5, paired_sample,
                                                 model)
                logging_output['_bleu_sys_len'] = bleu.sys_len
                logging_output['_bleu_ref_len'] = bleu.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                    logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu_counts_' + str(i)))
                totals.append(sum_logs('_bleu_totals_' + str(i)))
            new_c, new_t = [], []
            for c in counts:
                if isinstance(c, int):
                    new_c.append(c)
                else:
                    new_c.append(c.cpu())
            for t in totals:
                if isinstance(t, int):
                    new_t.append(t)
                else:
                    new_t.append(t.cpu())
            counts, totals = new_c, new_t
            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters['_bleu_counts'].sum,
                        total=meters['_bleu_totals'].sum,
                        sys_len=meters['_bleu_sys_len'].sum,
                        ref_len=meters['_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        if len(self.datasets.values()) == 0:
            return {self.lang_pair:
                        (self.args.max_source_positions, self.args.max_target_positions)}
        return OrderedDict([
            (key, (self.args.max_source_positions, self.args.max_target_positions))
            for split in self.datasets.keys()
            for key in self.datasets[split].datasets.keys()
        ])

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample,
                                      prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        if self.lambda_confidence_threshold < 0:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        if self.args.eval_bleu_print_samples and self.log_hypos:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
            self.log_hypos = False
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def train_step(self, sample, model, criterion, optimizer, update_num,
                   ignore_grad=False):
        model.train()
        if self.lambda_self_training > 0.0:
            self.log_hypos = True
            self.generate_progress = 0

        if update_num > 0:
            self.update_step(update_num)

        agg_loss, agg_sample_size, agg_logging_output = 0., 0., defaultdict(float)

        def forward_backward(model, samples, logging_output_key, weight):
            nonlocal agg_loss, agg_sample_size, agg_logging_output
            if samples is None or len(samples) == 0:
                return
            with torch.autograd.profiler.record_function("forward"):
                loss, sample_size, logging_output = criterion(model, samples)
            if ignore_grad:
                loss *= 0
            else:
                loss *= weight
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[k] += logging_output[k]
                agg_logging_output[logging_output_key] += logging_output[k]

        if self.lambda_parallel > 0.0:
            lang_pair = self.lang_pair
            forward_backward(model.models[lang_pair], sample[lang_pair],
                             lang_pair, self.lambda_parallel)

        if self.lambda_self_training > 0.0:
            sample_key = self.self_training_pair
            forward_backward(model.models[self.lang_pair],
                             sample[sample_key], sample_key,
                             self.lambda_self_training * self.pseudo_ratio)

        return agg_loss, agg_sample_size, agg_logging_output

    def update_step(self, num_updates):
        def lambda_step_func(config, n_iter):
            """
            Update a lambda value according to its schedule configuration.
            """
            ranges = [i for i in range(len(config) - 1) if
                      config[i][0] <= n_iter < config[i + 1][0]]
            if len(ranges) == 0:
                assert n_iter >= config[-1][0]
                return config[-1][1]
            assert len(ranges) == 1
            i = ranges[0]
            x_a, y_a = config[i]
            x_b, y_b = config[i + 1]
            return y_a + (n_iter - x_a) * float(y_b - y_a) / float(x_b - x_a)

        if self.lambda_parallel_steps is not None:
            self.lambda_parallel = lambda_step_func(self.lambda_parallel_steps,
                                                    num_updates)
        if self.lambda_self_training_steps is not None:
            self.lambda_self_training = lambda_step_func(
                self.lambda_self_training_steps, num_updates)

        if self.lambda_confidence_threshold_steps is not None:
            self.lambda_confidence_threshold = lambda_step_func(
                self.lambda_confidence_threshold_steps, num_updates)

        if self.pseudo_set is not None:
            self.pseudo_ratio = len(self.pseudo_set) / len(self.datasets['train'])
