import os
import sys
import hydra
import torch
import logging
import datasets
import transformers

import numpy as np

from typing import Optional, Any, Union
from dataclasses import dataclass
from transformers import set_seed, Trainer, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from sklearn.metrics import precision_recall_fscore_support
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from cai_common.utils.hydra_training_args import HydraTrainingArguments
from cai_garland.models.factory import make_bilingual_tokenizer
from cai_common.models.utils import get_local_ckpt, get_cai_config
from cai_garland.models.cai_encoder_decoder_seq_class import CAIEncoderDecoderForSequenceClassification


logger = logging.getLogger(__name__)

cs = ConfigStore()
cs.store(group="training", name="huggingface_training_args", node=HydraTrainingArguments)

seed = 42


def preprocess_function(examples, tokenizer):
    if tokenizer is None:
        raise ValueError("Tokenizer argument to preprocess_function is missing")
    inputs = {
        "labels": examples["label"],
    }
    inputs |= tokenizer(examples['source'], padding=False)
    target = [
        f"{pos}{tokenizer.target_tokenizer.mask_token}{tgt}" for pos, tgt in zip(examples["pos"], examples["target"])]
    with tokenizer.as_target_tokenizer():
        target = tokenizer(target, padding=False)
    inputs["decoder_input_ids"] = target["input_ids"]
    inputs["decoder_attention_mask"] = target["attention_mask"]
    return inputs


def compute_metrics(p):
    preds, labels = np.argmax(p.predictions, axis=1), p.label_ids
    precision, recall, fscore, _ = precision_recall_fscore_support(preds, labels, average='weighted')
    return {
        "precision": precision,
        "recall": recall,
        "f1": fscore
    }


@dataclass
class DataCollatorForBilingualSequenceClassification:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        encoder_features = [{
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"]
        } for example in features]
        decoder_features = [{
            "input_ids": example["decoder_input_ids"],
            "attention_mask": example["decoder_attention_mask"]
        } for example in features]

        encoder_features = self.tokenizer.pad(
            encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        decoder_features = self.tokenizer.pad(
            decoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        labels = [example["labels"] for example in features]
        if not return_tensors == "pt":
            raise NotImplementedError("Only PyTorch tensors currently implemented")
        labels = torch.LongTensor(labels)

        return {
            "input_ids": encoder_features["input_ids"],
            "attention_mask": encoder_features["attention_mask"],
            "decoder_input_ids": decoder_features["input_ids"],
            "decoder_attention_mask": decoder_features["attention_mask"],
            "labels": labels
        }


@hydra.main(version_base="1.2", config_path="./aligner.config", config_name="config")
def main(cfg):
    cfg.training.output_dir = HydraConfig.get().run.dir
    training_cfg = HydraTrainingArguments.as_hf_training_args(cfg.training)
    training_cfg.logging_dir = os.path.join(cfg.training.output_dir, "tb_logs")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    set_seed(seed)

    logger.info("Making encoder-decoder model")
    local_ckpt = get_local_ckpt("experiments/aligner/pretrained-translator", model_dir=True)
    model = CAIEncoderDecoderForSequenceClassification.from_pretrained(local_ckpt)

    logger.info("Loading CAI translation model config")
    cai_base_config = get_cai_config(local_ckpt)
    encoder_name = cai_base_config['encoder_model_name']
    encoder_length = cai_base_config['encoder_max_length']
    decoder_name = cai_base_config['decoder_model_name']
    decoder_length = cai_base_config['decoder_max_length']
    logger.info(f"Encoder name={encoder_name}, length={encoder_length}")
    logger.info(f"Decoder name={decoder_name}, length={decoder_length}")

    logger.info("Loading bilingual tokenizer")
    tokenizer = make_bilingual_tokenizer(encoder_name, decoder_name)

    logger.info("Loading datasets")
    data_dir = os.path.join(os.environ["CAI_DATA_BASE_PATH"], "experiments/aligner/")
    train_dataset = datasets.load_dataset("csv", data_files=os.path.join(data_dir, "train.csv"))['train']
    eval_dataset = datasets.load_dataset("csv", data_files=os.path.join(data_dir, "validation.csv"))['train']
    logger.info(f"Training dataset size = {len(train_dataset)}")
    logger.info(f"Validation dataset size = {len(eval_dataset)}")

    logger.info("Shuffling datasets")
    train_dataset = train_dataset.shuffle(seed=cfg.seed)
    eval_dataset = eval_dataset.shuffle(seed=cfg.seed)

    logger.info("Preprocessing training dataset")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=cfg.data.preprocessing_num_workers,
        load_from_cache_file=not cfg.data.overwrite_cache,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer}
    )
    logger.info("Preprocessing validation dataset")
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=cfg.data.preprocessing_num_workers,
        load_from_cache_file=not cfg.data.overwrite_cache,
        desc="Running tokenizer on validation dataset",
        fn_kwargs={"tokenizer": tokenizer}
    )

    logger.info("Filtering long examples from training dataset")
    pre_filter_len = len(train_dataset)
    train_dataset = train_dataset.filter(
        lambda ex: len(ex['input_ids']) < encoder_length and len(ex['decoder_input_ids']) < decoder_length,
        desc="Training filter"
    )
    logger.info(f"Training: {1 - len(train_dataset) / pre_filter_len} has been filtered out, {len(train_dataset)} "
                 "examples left.")
    logger.info("Filtering long examples from validation dataset")
    pre_filter_len = len(eval_dataset)
    eval_dataset = eval_dataset.filter(
        lambda ex: len(ex['input_ids']) < encoder_length and len(ex['decoder_input_ids']) < decoder_length,
        desc="Validation filter"
    )
    logger.info(f"Validation: {1 - len(eval_dataset) / pre_filter_len} has been filtered out, {len(eval_dataset)} "
                 "examples left.")

    logger.info("Kicking off training!")
    trainer = Trainer(
        model=model,
        args=training_cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForBilingualSequenceClassification(
            tokenizer,
            model=model
        ),
        compute_metrics=compute_metrics
    )
    trainer.train(
        ignore_keys_for_eval=[
            "past_key_values",
            "decoder_hidden_states",
            "decoder_attentions",
            "cross_attentions",
            "encoder_last_hidden_state",
            "encoder_hidden_states",
            "encoder_attentions",
        ]
    )

    logger.info("Saving results")
    trainer.save_model()


if __name__ == '__main__':
    main()      # pylint: disable=no-value-for-parameter
