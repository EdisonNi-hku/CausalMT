# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
    NoisingDataset,
    RoundRobinZipDatasets,
)

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


@register_task('translation_n_denoising')
class TranslationNDenoisingTask(LegacyFairseqTask):
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

        # add denosing parameters to the parser
        parser.add_argument('--lambda-parallel-config', default="1.0", type=str,
                            metavar='CONFIG',
                            help='cross-entropy reconstruction coefficient (parallel data). '
                                 'use fixed weight during training if set to floating point number. '
                                 'use piecewise linear function over number of updates to schedule the '
                                 'weight with the format: w0:step0,w1:step1,...')
        parser.add_argument('--lambda-denoising-config', default="1.0",
                            type=str, metavar='CONFIG',
                            help='Cross-entropy reconstruction coefficient (denoising autoencoding)'
                                 'use fixed weight during training if set to floating point number. '
                                 'use piecewise linear function over number of updates to schedule the '
                                 'weight with the format: w0:step0,w1:step1,...')
        parser.add_argument('--max-word-shuffle-distance', default=3.0,
                            type=float, metavar='N',
                            help='maximum word shuffle distance for denoising autoencoding data generation')
        parser.add_argument('--word-dropout-prob', default=0.1, type=float,
                            metavar='N',
                            help='word dropout probability for denoising autoencoding data generation')
        parser.add_argument('--word-blanking-prob', default=0.2, type=float,
                            metavar='N',
                            help='word blanking probability for denoising autoencoding data generation')

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

        self.lambda_parallel, self.lambda_parallel_steps = parse_lambda_config(
            args.lambda_parallel_config)
        self.lambda_denoising, self.lambda_denoising_steps = parse_lambda_config(
            args.lambda_denoising_config)

        if (
                self.lambda_denoising > 0.0 or self.lambda_denoising_steps is not None):
            denoising_lang_pair = "%s-%s" % (args.source_lang, args.source_lang)
            self.model_lang_pairs += [denoising_lang_pair]
            self.denoising_lang_pair = denoising_lang_pair
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
        if (self.lambda_parallel > 0.0 or self.lambda_parallel_steps is not None or not split.startswith("train")):
            src, tgt = self.args.source_lang, self.args.target_lang
            if split_exists(split, src, tgt, src):
                prefix = os.path.join(data_path, '{}.{}-{}.'.format(split, src, tgt))
            src_datasets[lang_pair] = load_indexed_dataset(prefix + src, self.dicts[src])
            tgt_datasets[lang_pair] = load_indexed_dataset(prefix + tgt, self.dicts[tgt])
            logger.info('parallel-{} {} {} examples'.format(data_path, split, len(src_datasets[lang_pair])))
        key_n_dataset = [(lang_pair, language_pair_dataset(lang_pair))]

        # load denoising autoencoder dataset
        if (self.lambda_denoising > 0.0 or self.lambda_denoising_steps is not None) and split.startswith("train"):
            filename = os.path.join(data_path,
                                    '{}.{}-None.{}'.format(split, src, src))
            src_dataset1 = load_indexed_dataset(filename, self.dicts[src])
            src_dataset2 = load_indexed_dataset(filename, self.dicts[src])
            noising_dataset = NoisingDataset(
                src_dataset1,
                self.dicts[src],
                seed=1,
                max_word_shuffle_distance=self.args.max_word_shuffle_distance,
                word_dropout_prob=self.args.word_dropout_prob,
                word_blanking_prob=self.args.word_blanking_prob,
            )
            noising_dataset = LanguagePairDataset(
                    noising_dataset,
                    src_dataset1.sizes,
                    self.dicts[src],
                    src_dataset2,
                    src_dataset2.sizes,
                    self.dicts[src],
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
            )
            logger.info('denoising-{}: {} {} {} examples'.format(
                src, data_path, split, len(noising_dataset),
            ))
            key_n_dataset.append(
                (self.denoising_lang_pair, noising_dataset)
            )


        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict(key_n_dataset),
            eval_key=self.lang_pair if split !='train' else None
        )


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

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([model], Namespace(
                **gen_args))
        return model

    def valid_step(self, sample, model, criterion):
        model.eval()

        lang_pair = self.lang_pair
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(
                model.models[lang_pair], sample)

        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample,
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
        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def train_step(self, sample, model, criterion, optimizer, update_num,
                   ignore_grad=False):
        model.train()

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

        if self.lambda_denoising > 0.0:
            src = self.args.source_lang
            sample_key = self.denoising_lang_pair
            forward_backward(model.models['{0}-{0}'.format(src)],
                             sample[sample_key], sample_key,
                             self.lambda_denoising)

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
        if self.lambda_denoising_steps is not None:
            self.lambda_denoising = lambda_step_func(
                self.lambda_denoising_steps, num_updates)
