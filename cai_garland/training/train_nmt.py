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
from colorama import init as init_colorama, Fore as ForeColor

import datasets
import numpy as np
from datasets import load_dataset, load_metric

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, default_data_collator, set_seed
from transformers.trainer_utils import get_last_checkpoint

from cai_common.utils.tensorboard_callback import CAITensorboardCallback
from cai_garland.models.factory import make_encoder_decoder
from cai_garland.data.siamese_collator import SiameseDataCollatorForSeq2Seq


logger = logging.getLogger(__name__)

init_colorama()
cs = ConfigStore()
cs.store(group="training", name="huggingface_training_args", node=HydraSeq2SeqTrainingArguments)


def preprocess_function(
    examples,
    tokenizer=None,
    max_source_length=None,
    max_target_length=None,
    padding=None,
    ignore_pad_token_for_loss=None,
    use_registers=False
):
    if tokenizer is None or \
        max_source_length is None or \
        max_target_length is None or \
        padding is None or \
        ignore_pad_token_for_loss is None:
        raise ValueError("One of the argument to preprocess_function is missing")

    inputs = [ex for ex in examples["tibetan"]]
    targets = [ex for ex in examples["english"]]

    inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    with tokenizer.as_target_tokenizer():
        targets = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and ignore_pad_token_for_loss:
        targets["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in targets["input_ids"]
        ]

    inputs["labels"] = targets["input_ids"]
    return inputs


@hydra.main(version_base="1.2", config_path="./train_nmt.config", config_name="config")
def main(cfg):
    cfg.training.output_dir = HydraConfig.get().run.dir
    training_cfg = HydraSeq2SeqTrainingArguments.as_hf_training_args(cfg.training)
    training_cfg.logging_dir = os.path.join(cfg.training.output_dir, "tb_logs")
    training_cfg.do_eval = not cfg.skip_eval

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

    set_seed(training_cfg.seed)

    logger.info("Loading training dataset")
    train_dataset = load_dataset(cfg.data.dataset_loader, cfg.data.dataset_config, split=datasets.splits.Split.TRAIN)
    logger.info("Loading validation dataset")
    eval_dataset = load_dataset(
        cfg.data.dataset_loader, cfg.data.dataset_config, split=datasets.splits.Split.VALIDATION)
    logger.info("Loading test dataset")
    test_dataset = load_dataset(cfg.data.dataset_loader, cfg.data.dataset_config, split=datasets.splits.Split.TEST)

    if cfg.training_preprocess.shuffle_training_data:
        logger.info("Shuffling training dataset")
        train_dataset = train_dataset.shuffle(seed=training_cfg.seed)

    logger.info("Making encoder-decoder model")
    model, tokenizer = make_encoder_decoder(cfg.model.encoder_model, cfg.model.decoder_model)

    import ipdb; ipdb.set_trace()

    if model.config.decoder.decoder_start_token_id is None:
        raise ValueError("Make sure that 'config.decoder_start_token_id' is correctly defined")

    # Temporarily set max_target_length for training.
    max_target_length = model.config.decoder.max_position_embeddings
    padding = "max_length" if cfg.training_preprocess.pad_to_max_length else False

    logger.info("Preprocessing training dataset")
    with training_cfg.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=cfg.training_preprocess.preprocessing_num_workers,
            load_from_cache_file=not cfg.overwrite_cache,
            desc="Running tokenizer on train dataset",
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_source_length": cfg.model.max_source_length,
                "max_target_length": max_target_length,
                "padding": padding,
                "ignore_pad_token_for_loss": cfg.training_preprocess.ignore_pad_token_for_loss
            }
        )
    logger.info("Preprocessing validation dataset")
    with training_cfg.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=cfg.training_preprocess.preprocessing_num_workers,
            load_from_cache_file=not cfg.overwrite_cache,
            desc="Running tokenizer on train dataset",
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_source_length": cfg.model.max_source_length,
                "max_target_length": max_target_length,
                "padding": padding,
                "ignore_pad_token_for_loss": cfg.training_preprocess.ignore_pad_token_for_loss
            }
        )
    logger.info("Preprocessing test dataset")
    with training_cfg.main_process_first(desc="test dataset map pre-processing"):
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=cfg.training_preprocess.preprocessing_num_workers,
            load_from_cache_file=not cfg.overwrite_cache,
            desc="Running tokenizer on test dataset",
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_source_length": cfg.model.max_source_length,
                "max_target_length": max_target_length,
                "padding": padding,
                "ignore_pad_token_for_loss": cfg.training_preprocess.ignore_pad_token_for_loss
            }
        )

    # Data collator
    label_pad_token_id = -100 if cfg.training_preprocess.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if cfg.training_preprocess.pad_to_max_length:
        logger.debug("Using default_data_collator")
        data_collator = default_data_collator
    else:
        logger.debug("Using DataCollatorForSeq2Seq")
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_cfg.fp16 else None,
        )

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
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if not cfg.skip_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_cfg.predict_with_generate else None,
    )
    CAITensorboardCallback.replace_in_trainer(trainer)

    # Training
    checkpoint = None
    if training_cfg.resume_from_checkpoint is not None:
        checkpoint = training_cfg.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
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
    if not cfg.skip_eval:
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


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
