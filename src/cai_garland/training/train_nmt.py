#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import random

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from cai_common.utils.hydra_training_args import HydraSeq2SeqTrainingArguments
from cai_common.utils.tensorboard_callback import CAITensorboardCallback
from colorama import init as init_colorama, Fore as ForeColor
from tqdm.auto import tqdm

import datasets
import numpy as np
from datasets import load_dataset, load_metric

from cai_garland.models.factory import make_encoder_decoder, make_bilingual_tokenizer, get_model_type
from cai_garland.data.siamese_collator import SiameseDataCollatorForSeq2Seq
from cai_garland.data.knowledge_collator import KnowledgeDataCollatorForSeq2Seq
from cai_garland.data.knowledge_injection_dataset import KnowledgeInjectionDataset
from cai_garland.training.cai_trainer_seq2seq import CAISeq2SeqTrainer

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, default_data_collator, set_seed, HfArgumentParser
from transformers.trainer_utils import IntervalStrategy, get_last_checkpoint, RemoveColumnsCollator

try:
    from peft import LoraConfig, TaskType, get_peft_model
    peft_available = True
except:
    peft_available = False

from cai_garland.models.cai_nllb_tokenizer import CAINllbTokenizerFast


logger = logging.getLogger(__name__)

init_colorama()
cs = ConfigStore()
cs.store(group="training", name="huggingface_training_args", node=HydraSeq2SeqTrainingArguments)

hf_args = None


def preprocess_function(
    examples,
    tokenizer=None,
    padding=None,
    ignore_pad_token_for_loss=None,
    siamese=None,
    tgt_lang_code=None
):
    if tokenizer is None or \
        padding is None or \
        ignore_pad_token_for_loss is None or \
        siamese is None \
    :
        raise ValueError("One of the argument to preprocess_function is missing")

    inputs = [ex for ex in examples["source"]]
    targets = [ex for ex in examples["target"]]

    inputs = tokenizer(inputs, padding=padding)

    if tgt_lang_code is not None:
        old_tgt_lang = tokenizer.target_tokenizer.tgt_lang
        tokenizer.target_tokenizer.tgt_lang = tgt_lang_code
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(targets, padding=padding)
    if tgt_lang_code is not None:
        tokenizer.target_tokenizer.tgt_lang = old_tgt_lang

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and ignore_pad_token_for_loss:
        if siamese:
            raise NotImplementedError("pad_to_max_length is not implemented with Siamese models")
        targets["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in targets["input_ids"]
        ]

    inputs["labels"] = targets["input_ids"]
    if "tokens_fixed" in inputs:
        del inputs["tokens_fixed"]
    return inputs


def load_hf_dataset(
    cfg, data_cfg, training_cfg, tokenizer, padding, siamese, skip_eval, eor_token_id, max_target_length
):
    logger.info("  Loading training dataset")
    train_split_name = data_cfg.get('train_split_name', datasets.splits.Split.TRAIN)
    train_dataset = load_dataset(data_cfg.dataset_loader, data_cfg.dataset_config, split=train_split_name)
    logger.info("  Loading validation dataset")
    validation_split_name = data_cfg.get('validation_split_name', datasets.splits.Split.VALIDATION)
    if validation_split_name is None:
        eval_dataset = None
    elif validation_split_name == "train":
        split_datasets = train_dataset.train_test_split(
            train_size=1 - data_cfg.validation_sampling_rate, seed=training_cfg.seed)
        train_dataset, eval_dataset = split_datasets['train'], split_datasets['test']
        logger.info("  Split into training and validation")
        del split_datasets
    else:
        eval_dataset = load_dataset(
            data_cfg.dataset_loader,
            data_cfg.dataset_config,
            split=data_cfg.get('validation_split_name', datasets.splits.Split.VALIDATION)
        )
    logger.info("  Loading test dataset")
    test_split_name = data_cfg.get('test_split_name', datasets.splits.Split.TEST)
    if test_split_name is not None:
        test_dataset = load_dataset(data_cfg.dataset_loader, data_cfg.dataset_config, split=test_split_name)
    else:
        test_dataset = None

    logger.info("  Setting up context injection if needed")
    inject_context = data_cfg.get('inject_context', False)
    train_dataset = train_dataset.add_column(name="inject_context", column=[inject_context]*len(train_dataset))
    if eval_dataset is not None:
        eval_dataset = eval_dataset.add_column(name="inject_context", column=[inject_context]*len(eval_dataset))
    if test_dataset is not None:
        test_dataset = test_dataset.add_column(name="inject_context", column=[inject_context]*len(test_dataset))

    logger.info(f"    Training size   = {len(train_dataset)}")
    if eval_dataset is not None:
        logger.info(f"    Validation size = {len(eval_dataset)}")
    else:
        logger.info("    No validation set")
    if test_dataset is not None:
        logger.info(f"    Test size       = {len(test_dataset)}")
    else:
        logger.info("    No test set")

    if data_cfg.get('score_cutoff', None) is not None:
        logger.info("  Filtering dataset by score cutoff")
        pre_filter_len = len(train_dataset)
        train_dataset = train_dataset.filter(lambda x: x['score'] > data_cfg.score_cutoff, desc="Score filter")
        logger.info(f"  {1 - len(train_dataset) / pre_filter_len} has been filtered out, {len(train_dataset)} "
                     "examples left.")

    tgt_lang_code = getattr(data_cfg, "tgt_lang_code", None)
    logger.info("  Preprocessing training dataset")
    with training_cfg.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=cfg.training_preprocess.preprocessing_num_workers,
            load_from_cache_file=not cfg.overwrite_cache,
            desc="Running tokenizer on train dataset",
            fn_kwargs={
                "tokenizer": tokenizer,
                "padding": padding,
                "ignore_pad_token_for_loss": cfg.training_preprocess.ignore_pad_token_for_loss,
                "siamese": siamese,
                "tgt_lang_code": tgt_lang_code
            }
        )
    if not skip_eval:
        if eval_dataset is not None:
            logger.info("  Preprocessing validation dataset")
            with training_cfg.main_process_first(desc="validation dataset map pre-processing"):
                eval_dataset = eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=cfg.training_preprocess.preprocessing_num_workers,
                    load_from_cache_file=not cfg.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "padding": padding,
                        "ignore_pad_token_for_loss": cfg.training_preprocess.ignore_pad_token_for_loss,
                        "siamese": siamese,
                        "tgt_lang_code": tgt_lang_code
                    }
                )
        if test_dataset is not None:
            logger.info("  Preprocessing test dataset")
            with training_cfg.main_process_first(desc="test dataset map pre-processing"):
                test_dataset = test_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=cfg.training_preprocess.preprocessing_num_workers,
                    load_from_cache_file=not cfg.overwrite_cache,
                    desc="Running tokenizer on test dataset",
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "padding": padding,
                        "ignore_pad_token_for_loss": cfg.training_preprocess.ignore_pad_token_for_loss,
                        "siamese": siamese,
                        "tgt_lang_code": tgt_lang_code
                    }
                )

    logger.info("  Filtering out long examples")
    is_not_long_example_lambda = lambda x: not is_long_example(
        x, eor_token_id, cfg.model.max_source_length, max_target_length)
    pre_filter_len = len(train_dataset)
    train_dataset = train_dataset.filter(is_not_long_example_lambda, desc="Training filter")
    logger.info(f"  Training: {1 - len(train_dataset) / pre_filter_len} has been filtered out, {len(train_dataset)} "
                 "examples left.")

    if eval_dataset is not None:
        pre_filter_len = len(eval_dataset)
        eval_dataset = eval_dataset.filter(is_not_long_example_lambda, desc="Validation filter")
        logger.info(f"  Validation: {1 - len(eval_dataset) / pre_filter_len} has been filtered out.")
        logger.info(f"  Validation: {1 - len(eval_dataset) / pre_filter_len} has been filtered out, "
                    f"{len(eval_dataset)} examples left.")

        logger.info(f"  Original validation set, prior to rebalancing and resampling, size is {len(eval_dataset)}")
        validation_register_rebalance_frac = data_cfg.get('validation_register_rebalance_frac', None)
        if validation_register_rebalance_frac is not None:
            register_eval_dataset = eval_dataset.filter(
                lambda ex: tokenizer.source_tokenizer.eor_token in ex['tibetan'])
            no_register_eval_dataset = eval_dataset \
                .filter(lambda ex: not tokenizer.source_tokenizer.eor_token in ex['tibetan'])
            no_register_eval_dataset = no_register_eval_dataset \
                .shuffle(seed=training_cfg.seed) \
                .select(range(
                    min(
                        len(register_eval_dataset) * validation_register_rebalance_frac,
                        len(no_register_eval_dataset)
                    )
                ))
            eval_dataset = datasets.interleave_datasets([register_eval_dataset, no_register_eval_dataset])
            logger.info(f"  Rebalanced validation set to size {len(eval_dataset)}")

        validation_subsampling_rate = data_cfg.get('validation_subsampling_rate', None)
        if validation_subsampling_rate is not None:
            eval_dataset = eval_dataset.train_test_split(
                train_size=validation_subsampling_rate, seed=training_cfg.seed)['train']
            logger.info(f"  Subsampled validation set to size {len(eval_dataset)}")

    if test_dataset is not None:
        pre_filter_len = len(test_dataset)
        if pre_filter_len > 0:
            test_dataset = test_dataset.filter(is_not_long_example_lambda, desc="Test filter")
            logger.info(f"  Test: {1 - len(test_dataset) / pre_filter_len} has been filtered out.")
            logger.info(f"  Test: {1 - len(test_dataset) / pre_filter_len} has been filtered out, {len(test_dataset)} "
                        "examples left.")
        else:
            logger.info("  No test data to filter.")

    return train_dataset, eval_dataset, test_dataset


def is_long_example(example, eor_token_id, max_source_length, max_target_length):
    tokens = example['input_ids'][1:-1]
    register_splits = [idx for idx, token in enumerate(tokens) if token == eor_token_id]
    register_splits = [0] + register_splits + [len(tokens)]
    if max([x2 - x1 for x1, x2 in zip(register_splits[:-1], register_splits[1:])]) > max_source_length - 2:
        return True
    if len(example['labels']) > max_target_length:
        return True
    return False


@hydra.main(version_base="1.2", config_path="./train_nmt.config", config_name="config")
def main(cfg):
    cfg.training.output_dir = HydraConfig.get().run.dir
    training_cfg = HydraSeq2SeqTrainingArguments.as_hf_training_args(cfg.training, hf_args)
    training_cfg.logging_dir = os.path.join(cfg.training.output_dir, "tb_logs")
    skip_eval = cfg.training.evaluation_strategy == IntervalStrategy.NO

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(
        f"Process rank: {training_cfg.local_rank}, device: {training_cfg.device}, n_gpu: {training_cfg.n_gpu}, "
        f"distributed training: {bool(training_cfg.local_rank != -1)}, 16-bits training: {training_cfg.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_cfg.output_dir) and not training_cfg.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_cfg.output_dir)
        if last_checkpoint is not None and training_cfg.resume_from_checkpoint is None:
            logger.info(
                f"{ForeColor.LIGHTCYAN_EX}Checkpoint detected, resuming training at {last_checkpoint}. To avoid this "
                "behavior, change the '--output_dir' or add '--overwrite_output_dir' to train from scratch."
            )
        else:
            logger.info("No previous checkpoint detected, training from scratch")
    else:
        logger.info("Output directory not found or overwritten")

    checkpoint, resume_from_checkpoint = None, True
    if "pretrained_checkpoint" in cfg and "resume_from_checkpoint" in cfg:
        raise ValueError("Either specify a pretrained checkpoint to start from or a previous checkpoint to resume from")
    if "pretrained_checkpoint" in cfg:
        checkpoint = cfg.pretrained_checkpoint
        resume_from_checkpoint = False
    elif "resume_from_checkpoint" in cfg:
        checkpoint = cfg.resume_from_checkpoint
        resume_from_checkpoint = True
    elif training_cfg.resume_from_checkpoint is not None:
        checkpoint = training_cfg.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    set_seed(training_cfg.seed)

    deepspeed = getattr(training_cfg, 'deepspeed', None) is not None
    lora = getattr(cfg.training, "lora", None) is not None
    if checkpoint is not None:
        logger.info(f"Loading weights from local checkpoint {checkpoint}")
        model = get_model_type().from_pretrained(checkpoint)
        tokenizer = make_bilingual_tokenizer(cfg.model.encoder_model, cfg.model.decoder_model, is_deepspeed=deepspeed)
    else:
        logger.info("Making encoder-decoder model from default pre-trained checkpoint")
        model, tokenizer = make_encoder_decoder(
            cfg.model.encoder_model, cfg.model.decoder_model, is_deepspeed=deepspeed)
    if not resume_from_checkpoint:
        checkpoint = None

    if model.config.decoder.decoder_start_token_id is None:
        raise ValueError("Make sure that 'config.decoder_start_token_id' is correctly defined")

    model.label_smoothing_factor = cfg.training.label_smoothing
    model.fc_layer_reg_lambda = cfg.training.fc_layer_reg_lambda

    # Temporarily set max_target_length for training.
    max_target_length = model.config.decoder.max_position_embeddings
    padding = "max_length" if cfg.training_preprocess.pad_to_max_length else False
    siamese = model.config.encoder.model_type == "siamese-encoder"
    eor_token_id = tokenizer.source_tokenizer.eor_token_id if siamese else -1

    if getattr(cfg.model, "regularize_context", False):
        model.decoder.model.decoder.regularize_context = True
        if hasattr(cfg.model, "regularization_sigma"):
            model.decoder.model.decoder.regularization_sigma = cfg.model.regularization_sigma

    if hasattr(cfg.model, "freeze"):
        if getattr(cfg.model.freeze, "lm_head", False):
            for param in model.decoder.lm_head.parameters():
                param.requires_grad = False
        if getattr(cfg.model.freeze, "encoder", False):
            for param in model.encoder.parameters():
                param.requires_grad = False

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"Model has {num_params:,} trainable parameters.")
    if deepspeed:
        if num_params > 0:
            logger.error("DeepSpeed is switched on but there are detected trainable parameters! This number should "
                        "have been zero.")
            for param_name, param in model.named_parameters():
                param_size = np.prod(param.size())
                if param_size > 0:
                    logger.error(f"Trainable parameter layer {param_name} has {param_size:,} trainable parameters!")
            raise ValueError("DeepSpeed is switched on but there are detected trainable parameters! This number should "
                            "have been zero.")
        else:
            logger.info("You're using DeepSpeed, so zero detected trainable parameters is correct. DeepSpeed will "
                        "print the actual number of trainable parameters separately.")

    if lora:
        if not peft_available:
            raise ImportError("PEFT not installed but LoRA requested")
        lora_cfg = cfg.training.lora
        logger.info("Initializing LoRA via PEFT")
        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            bias=lora_cfg.bias,
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=["key", "query", "value"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Temporarily set max_target_length for training.
    max_target_length = model.config.decoder.max_position_embeddings
    padding = "max_length" if cfg.training_preprocess.pad_to_max_length else False
    siamese = model.config.encoder.model_type == "siamese-encoder"
    context_injection = hasattr(cfg.dataset_construction, "context_injection")
    if context_injection:
        model.force_preparing_model_for_generation = True

    train_datasets, eval_datasets, test_datasets = [], [], []
    for dataset_idx, dataset_name in enumerate(cfg.data):
        logger.info(f"{ForeColor.LIGHTCYAN_EX}Loading dataset \"{dataset_name}\" ({dataset_idx + 1}/{len(cfg.data)})")
        train_dataset, eval_dataset, test_dataset = load_hf_dataset(
            cfg,
            cfg.data[dataset_name],
            training_cfg,
            tokenizer,
            padding,
            siamese,
            skip_eval,
            eor_token_id,
            max_target_length
        )
        if train_dataset is not None:
            train_datasets.append(train_dataset)
        if eval_dataset is not None:
            eval_datasets.append(eval_dataset)
        if test_dataset is not None:
            test_datasets.append(test_dataset)
    for interleave_step_idx, interleave_step in enumerate(cfg.interleave):
        logging.info(f"Interleave step {interleave_step_idx}")
        interleaving_rates = getattr(interleave_step, "rates", None)
        if interleaving_rates is None:
            interleaving_rates = [1 / len(cfg.data)] * (len(cfg.data) - 1)
        interleaving_rates = list(interleaving_rates)
        interleaving_rates = [1 - sum(interleaving_rates)] + interleaving_rates
        if interleaving_rates[0] < 0:
            raise ValueError("Interleaving rate for one of the augmenting datasets is too large.")

        def _interleave_datasets(datasets_, label, interleave_step):
            if len(datasets_) > 1:
                logger.info(f"Interleaving {label} datasets")
                strategy = getattr(interleave_step, "stopping_strategy", "first_exhausted")
                return datasets.interleave_datasets(datasets_, probabilities=interleaving_rates, stopping_strategy=strategy)
            else:
                return datasets_[0]

        cur_train_dataset = _interleave_datasets(train_datasets[:len(interleaving_rates)], "training", interleave_step)
        cur_eval_dataset = _interleave_datasets(eval_datasets[:len(interleaving_rates)], "validation", interleave_step)
        cur_test_dataset = _interleave_datasets(test_datasets[:len(interleaving_rates)], "test", interleave_step)

        train_datasets = [cur_train_dataset] + train_datasets[len(interleaving_rates):]
        eval_datasets = [cur_eval_dataset] + eval_datasets[len(interleaving_rates):]
        test_datasets = [cur_test_dataset] + test_datasets[len(interleaving_rates):]

    if len(train_datasets) > 1:
        raise ValueError("Not enough interleaving steps specified.")

    train_dataset, eval_dataset, test_dataset = train_datasets[0], eval_datasets[0], test_datasets[0]

    logger.info("Final dataset sizes:")
    logger.info(f"    Training size   = {len(train_dataset)}")
    if eval_dataset is not None:
        logger.info(f"    Validation size = {len(eval_dataset)}")
    else:
        logger.info("    No validation set")
    if test_dataset is not None:
        logger.info(f"    Test size       = {len(test_dataset)}")
    else:
        logger.info("    No test set")

    logger.info("Counting training set unks")
    unk_count = sum(
        [sum([t == tokenizer.target_tokenizer.unk_token_id for t in ex["labels"]]) for ex in tqdm(train_dataset)])
    logger.info("Counting training set tokens")
    num_tokens = sum([1 for ex in tqdm(train_dataset) for t in ex["labels"]])
    logger.info(f"There are {unk_count} unks out of {num_tokens} tokens in the training data.")

    if cfg.training_preprocess.shuffle_training_data:
        logger.info("Shuffling training dataset")
        train_dataset = train_dataset.shuffle(seed=training_cfg.seed)

    train_dataset, eval_dataset, test_dataset = [
        KnowledgeInjectionDataset(d) for d in (train_dataset, eval_dataset, test_dataset)
    ]

    if context_injection:
        pci_cfg = cfg.dataset_construction.context_injection
        KnowledgeInjectionDataset.prepare_context_embeddings(
            pci_cfg.context_file,
            pci_cfg.context_encoder,
            raw_contexts=pci_cfg.raw_contexts,
            cuda=getattr(pci_cfg, "cuda", False),
            batch_size=getattr(pci_cfg, "batch_size", 1),
            overwrite=getattr(pci_cfg, "overwrite_existing_embeddings", False)
        )
        logger.info("Testing context alignment for training dataset")
        train_dataset.inject_context(pci_cfg.context_lookup_key, raw_contexts=pci_cfg.raw_contexts)
        if eval_dataset is not None:
            logger.info("Testing context alignment for validation dataset")
            eval_dataset.inject_context(pci_cfg.context_lookup_key, raw_contexts=pci_cfg.raw_contexts)
        if test_dataset is not None:
            logger.info("Testing context alignment for test dataset")
            test_dataset.inject_context(pci_cfg.context_lookup_key, raw_contexts=pci_cfg.raw_contexts)

    glossary_injection = hasattr(cfg, "glossary")
    if glossary_injection:
        logger.info("Injecting glossary into training dataset")
        train_dataset.inject_glossary(
            cfg.glossary.dataset_path,
            source_encoder_name = cfg.glossary.source_encoder_name,
            target_decoder_name = cfg.glossary.target_decoder_name,
            is_deepspeed=deepspeed
        )
        if eval_dataset is not None:
            logger.info("Injecting glossary into validation dataset")
            eval_dataset.inject_glossary(
                cfg.glossary.dataset_path,
                source_encoder_name = cfg.glossary.source_encoder_name,
                target_decoder_name = cfg.glossary.target_decoder_name,
                is_deepspeed=deepspeed
            )
        if test_dataset is not None:
            logger.info("Injecting glossary into test dataset")
            test_dataset.inject_glossary(
                cfg.glossary.dataset_path,
                source_encoder_name = cfg.glossary.source_encoder_name,
                target_decoder_name = cfg.glossary.target_decoder_name,
                is_deepspeed=deepspeed
            )

    # Data collator
    quantize_padding = (training_cfg.fp16 or training_cfg.bf16) and not cfg.training_preprocess.no_padding_quantization
    label_pad_token_id = -100 if cfg.training_preprocess.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if cfg.training_preprocess.pad_to_max_length:
        if siamese:
            raise NotImplementedError("Siamese encoders with pad_to_max_length are not currently implemented")
        logger.debug("Using default_data_collator")
        data_collator = default_data_collator
    else:
        logger.debug("Using DataCollatorForSeq2Seq")
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if quantize_padding else None,
        )
        if context_injection or glossary_injection:
            logger.debug("Wrapping in KnowledgeDataCollatorForSeq2Seq")
            data_collator = KnowledgeDataCollatorForSeq2Seq(data_collator)
        if siamese:
            logger.debug("Wrapping in SiameseDataCollatorForSeq2Seq")
            data_collator = SiameseDataCollatorForSeq2Seq(
                data_collator,
                model.config.encoder.num_registers,
                eor_token_id
            )
        if context_injection:
            logger.debug("Activating context collation")
            with tokenizer.as_target_tokenizer():
                tgt_pad_token_id = tokenizer.pad_token_id
            data_collator.setup_context(
                train_dataset.context_name_key, raw_context=pci_cfg.raw_contexts, raw_pad_token_id=tgt_pad_token_id
            )
        if glossary_injection:
            logger.debug("Activating glossary collation")
            data_collator.setup_glossary(train_dataset.glossary_name_key)

    # Metric
    logger.info("Loading SacreBLEU")
    metric = load_metric("sacrebleu")
    eval_output_idxs = random.choices(range(len(eval_dataset)), k=cfg.training_preprocess.eval_decodings_in_tensorboard)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        with tokenizer.as_target_tokenizer():
            pad_token_id = tokenizer.pad_token_id
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            if cfg.training_preprocess.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        if len(eval_output_idxs) > 0:
            text_logs = {
                f"evaluation_example_{idx}": f"Prediction: {decoded_preds[idx]}\n\nLabel: {decoded_labels[idx][0]}"
                for idx in eval_output_idxs
            }
            trainer.log(text_logs)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    logger.info("Initializing trainer")
    if lora:
        training_cfg.remove_unused_columns = False
        import inspect
        data_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=list(inspect.signature(model.base_model.model.forward).parameters.keys()),
            logger=logger,
            description=None,
            model_name=model.__class__.__name__,
        )
    trainer = CAISeq2SeqTrainer(
        model=model,
        args=training_cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if not skip_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_cfg.predict_with_generate else None,
    )
    CAITensorboardCallback.replace_in_trainer(trainer)
    # if getattr(cfg.model, "decoder_has_forced_bos_token", False):
    #     trainer.generation_kwargs = {
    #         "forced_bos_token_id": model.forced_bos_token_id(tokenizer)
    #     }

    # Training
    logger.info("Kicking off training")
    logger.debug(f"    resume_from_checkpoint={checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    logger.info("Saving final model")
    trainer.save_model()

    metrics = train_result.metrics
    max_train_samples = (
        cfg.training_preprocess.max_train_samples if cfg.training_preprocess.max_train_samples is not None else \
            len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    logger.info("Saving training metrics")
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    logger.info("Saving trainer state")
    trainer.save_state()

    results = {}
    max_length = (
        training_cfg.generation_max_length
        if training_cfg.generation_max_length is not None
        else cfg.training_preprocess.val_max_target_length
    )
    if not skip_eval and test_dataset is not None:
        logger.info(f"{ForeColor.LIGHTCYAN_EX}Evaluating the final checkpoint")

        metrics = trainer.evaluate(
            max_length=max_length, num_beams=training_cfg.generation_num_beams, metric_key_prefix="eval")
        max_eval_samples = \
            cfg.training_preprocess.max_eval_samples if cfg.training_preprocess.max_eval_samples is not None else \
                len(test_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(test_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return results


def _mp_fn(_index):
    # For xla_spawn (TPUs)
    main()      # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    hydra_args = [sys.argv[0]] + [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    hf_args = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg.startswith('--')]

    sys.argv = hydra_args       # Kludge to keep the Hydra decorator working
    main()                      # pylint: disable=no-value-for-parameter
