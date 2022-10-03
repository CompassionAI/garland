import os
import sys
import glob
import logging
import unicodedata

import hydra
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm


@hydra.main(version_base="1.2", config_path="./translate.config", config_name="retranslate")
def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    logging.info("Loading re-translation model")
    logging.debug("  Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_model_name, src_lang="eng_Latn")
    if cfg.list_language_codes:
        for lang_code in sorted(list(tokenizer.lang_code_to_id.keys())):
            print(lang_code)
        return
    logging.debug("  Loading model")
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.hf_model_name)
    if cfg.model.cuda:
        logging.debug("  Copying model to GPU")
        model.cuda()

    logging.info("Re-translating")
    output_ext = getattr(cfg, "output_extension", cfg.target_language_code)
    in_fns = glob.glob(cfg.input_glob)
    for in_fn in (files_pbar := tqdm(in_fns)):
        files_pbar.set_description(os.path.basename(in_fn))
        with open(in_fn, 'r') as f_in:
            translated = [l.strip() for l in f_in.readlines()]
        out_fn = os.path.join(cfg.output_dir, os.path.splitext(os.path.basename(in_fn))[0] + '.' + output_ext)
        with open(out_fn, 'w') as f_out:
            for line in tqdm(translated, leave=False, desc="Translating"):
                if any('TIBETAN' in unicodedata.name(c) for c in line):
                    f_out.write(line + '\n')
                elif len(line) == 0:
                    f_out.write('\n')
                else:
                    translated_tokens = model.generate(
                        **tokenizer(line, return_tensors="pt").to(model.device),
                        forced_bos_token_id=tokenizer.lang_code_to_id[cfg.target_language_code],
                        max_length=cfg.model.max_length
                    )[0]
                    f_out.write(tokenizer.decode(translated_tokens, skip_special_tokens=True) + '\n')
                f_out.flush()


if __name__ == "__main__":
    main()      # pylint: disable=no-value-for-parameter
