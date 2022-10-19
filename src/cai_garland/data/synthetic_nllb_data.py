import os
import sys
import hydra
import logging
import unicodedata

from tqdm.auto import tqdm
from datasets import load_dataset
from colorama import init as init_colorama
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from cai_garland.utils.str_processors import ProcessorSymbolCleaningJSON
from .parallel_dataset_prep import _apply_processors_unpacked


init_colorama()
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="./dataset_prep.config", config_name="synthetic_nllb_data")
def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    ProcessorSymbolCleaningJSON.base_dir = os.path.dirname(__file__)

    logger.info("Loading dataset")
    nllb_data = load_dataset("allenai/nllb", cfg.languages.source + '-' + cfg.languages.target)[cfg.dataset.split]
    logger.info("Extracting original data")
    translation_key = cfg.dataset.translation_key
    original_data = [ex[translation_key] for ex in tqdm(nllb_data)]
    logger.info(f"Loaded dataset of length {len(original_data)}")

    logger.info("Filtering")
    filtered_data = []
    for ex in tqdm(original_data):
        if all('TIBETAN' in unicodedata.name(c) or c == ' ' for c in ex[cfg.languages.source]):
            filtered_data.append(ex)
    logger.info(f"Dataset length={len(filtered_data)} after filtering")

    translation_needed = hasattr(cfg.languages, "final") and not cfg.languages.final == cfg.languages.target
    if translation_needed:
        logger.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model, src_lang=cfg.languages.source, tgt_lang=cfg.languages.final)
        logger.info("Loading model")
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model)

        if cfg.compute.cuda:
            model.cuda()

        logger.info("Translating")
    else:
        logger.info("Dumping dataset")

    os.makedirs(os.path.join(os.environ['CAI_TEMP_PATH'], cfg.output.output_dir), exist_ok=True)
    with \
        open(os.path.join(os.environ['CAI_TEMP_PATH'], cfg.output.output_dir, cfg.output.source_fn), 'w') as f_source, \
        open(os.path.join(os.environ['CAI_TEMP_PATH'], cfg.output.output_dir, cfg.output.target_fn), 'w') as f_target \
    :
        for i in tqdm(range(0, len(filtered_data), cfg.compute.batch_size)):
            parallel_batch = filtered_data[i:i+cfg.compute.batch_size]
            intermediate_batch = [ex[cfg.languages.target] for ex in parallel_batch]
            if translation_needed:
                translated_tokens = model.generate(
                    **tokenizer(intermediate_batch, return_tensors="pt", padding=True).to(model.device),
                    forced_bos_token_id=tokenizer.lang_code_to_id[cfg.languages.final],
                    max_length=cfg.generation.max_length,
                    num_beams=cfg.generation.num_beams,
                    repetition_penalty=cfg.generation.repetition_penalty,
                    no_repeat_ngram_size=cfg.generation.no_repeat_ngram_size
                )
                targets_batch = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            else:
                targets_batch = intermediate_batch
            sources_batch = [ex[cfg.languages.source] for ex in parallel_batch]

            sources_batch = _apply_processors_unpacked(cfg.dataset.preprocessing.source_lang, [sources_batch])[0]
            sources_batch = _apply_processors_unpacked(cfg.output.postprocessing.source_lang, [sources_batch])[0]
            targets_batch = _apply_processors_unpacked(cfg.dataset.preprocessing.target_lang, [targets_batch])[0]
            targets_batch = _apply_processors_unpacked(cfg.output.postprocessing.target_lang, [targets_batch])[0]

            for source_l, target_l in zip(sources_batch, targets_batch):
                if target_l == '.' or target_l == '':
                    continue
                f_source.write(source_l + '\n')
                f_target.write(target_l + '\n')
            f_source.flush()
            f_target.flush()


if __name__ == "__main__":
    main()      # pylint: disable=no-value-for-parameter
