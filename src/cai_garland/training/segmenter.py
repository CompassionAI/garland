import os
import sys
import hydra
import logging
import datasets
import transformers

import numpy as np

from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    set_seed,
    AutoConfig,
    AlbertForSequenceClassification,
    Trainer,
    DataCollatorWithPadding
)
from cai_common.models.utils import get_local_ckpt
from cai_common.utils.hydra_training_args import HydraTrainingArguments
from cai_manas.tokenizer.tokenizer import CAITokenizer


logger = logging.getLogger(__name__)

cs = ConfigStore()
cs.store(group="training", name="huggingface_training_args", node=HydraTrainingArguments)


def preprocess_function(examples, tokenizer):
    if tokenizer is None:
        raise ValueError("Tokenizer argument to preprocess_function is missing")
    inputs = tokenizer(examples['segment'], padding=False)
    inputs['labels'] = examples['label']
    return inputs


def compute_metrics(p):
    preds, labels = np.argmax(p.predictions, axis=1), p.label_ids
    precision, recall, fscore, _ = precision_recall_fscore_support(preds, labels, average='weighted')
    return {
        "precision": precision,
        "recall": recall,
        "f1": fscore
    }


@hydra.main(version_base="1.2", config_path="./segmenter.config", config_name="config")
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

    set_seed(cfg.seed)

    logger.info("Loading tokenizer")
    logger.info(f"Creating tokenizer: {cfg.model.tokenizer_name}")
    logger.debug(f"Tokenizer location: {CAITokenizer.get_local_model_dir(cfg.model.tokenizer_name)}")
    tokenizer = CAITokenizer.from_pretrained(CAITokenizer.get_local_model_dir(cfg.model.tokenizer_name))
    logger.debug(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

    logger.info("Loading model")
    albert_cfg = AutoConfig.from_pretrained(cfg.model.model_hf_name, num_labels=2)
    local_ckpt = get_local_ckpt(cfg.model.cai_checkpoint_name)
    logger.info(f"Local checkpoint resolved to: {local_ckpt}")
    tibert_mdl = AlbertForSequenceClassification.from_pretrained(local_ckpt, config=albert_cfg)
    albert_cfg.vocab_size = tokenizer.vocab_size
    albert_cfg.max_position_embeddings = cfg.model.model_length
    tibert_mdl.resize_token_embeddings(len(tokenizer))

    logger.info("Loading datasets")
    data_dir = os.path.join(os.environ["CAI_DATA_BASE_PATH"], "processed_datasets/segmenter-dataset")
    train_dataset = datasets.load_dataset("csv", data_files=os.path.join(data_dir, "train.csv"))['train']
    eval_dataset = datasets.load_dataset("csv", data_files=os.path.join(data_dir, "validation.csv"))['train']
    logger.info(f"Training dataset size = {len(train_dataset)}")
    logger.info(f"Validation dataset size = {len(eval_dataset)}")

    logger.info("Shuffling training dataset")
    train_dataset = train_dataset.shuffle(seed=cfg.seed)

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
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer}
    )

    logger.info("Filtering long examples from training dataset")
    pre_filter_len = len(train_dataset)
    train_dataset = train_dataset.filter(
        lambda ex: len(ex['input_ids']) < tibert_mdl.config.max_position_embeddings,
        desc="Training filter"
    )
    logger.info(f"Training: {1 - len(train_dataset) / pre_filter_len} has been filtered out, {len(train_dataset)} "
                 "examples left.")
    logger.info("Filtering long examples from validation dataset")
    pre_filter_len = len(eval_dataset)
    eval_dataset = eval_dataset.filter(
        lambda ex: len(ex['input_ids']) < tibert_mdl.config.max_position_embeddings,
        desc="Validation filter"
    )
    logger.info(f"Validation: {1 - len(eval_dataset) / pre_filter_len} has been filtered out, {len(eval_dataset)} "
                 "examples left.")

    logger.info("Kicking off training!")
    trainer = Trainer(
        model=tibert_mdl,
        args=training_cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt'
        ),
        compute_metrics=compute_metrics
    )
    trainer.train()

    logger.info("Saving results")
    trainer.save_model()


if __name__ == '__main__':
    main()      # pylint: disable=no-value-for-parameter
