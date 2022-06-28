import os
import sys
import logging
import hydra
from hydra.utils import instantiate
import glob

from tqdm.auto import tqdm

from ..utils.translator import Translator, TokenizationTooLongException


def interactive(translator, cfg):
    print("Interactive Tibetan translation...")
    while True:
        print("===")
        bo_text = input("Tibetan (or type exit): ")
        if bo_text == "exit":
            break
        print(translator.translate(bo_text))


def batch(translator, cfg):
    if cfg.mode.input_glob is None:
        raise ValueError("Specify an input file (or glob) in the mode.input_glob setting. You can do this from the "
                         "command line.")
    os.makedirs(cfg.mode.output_dir, exist_ok=True)
    in_fns = glob.glob(cfg.mode.input_glob)
    for in_fn in tqdm(in_fns, desc="Files"):
        with open(in_fn, encoding=cfg.mode.encoding) as in_f:
            bo_text = in_f.read()

        hard_segments = instantiate(cfg.mode.hard_segmentation, bo_text, **dict(cfg.mode.hard_segmenter_kwargs))

        out_fn = os.path.join(
            cfg.mode.output_dir, os.path.splitext(os.path.basename(in_fn))[0] + '.' + cfg.mode.output_extension)
        with open(out_fn, mode='w') as out_f:
            for segment in tqdm(hard_segments, desc="Segments", leave=False):
                for preproc_func in cfg.mode.preprocessing:
                    segment = instantiate(preproc_func, segment)

                try:
                    tgt_segment = translator.translate(segment)
                except TokenizationTooLongException as err:
                    if not cfg.mode.skip_long_inputs:
                        raise err
                    else:
                        tgt_segment = "SEGMENT TOKENIZATION TOO LONG FOR ENCODER MODEL"

                for postproc_func in cfg.mode.postprocessing:
                    tgt_segment = instantiate(postproc_func, tgt_segment)

                if cfg.mode.output_parallel_translation:
                    out_f.write(segment + '\n')
                out_f.write(tgt_segment + '\n')
                out_f.write('\n')


@hydra.main(version_base="1.2", config_path="./translate.config", config_name="config")
def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    translator = Translator(os.path.join(cfg.model.model_ckpt, cfg.model.model_size))
    translator.num_beams = cfg.generation.num_beams
    if cfg.cuda:
        translator.cuda()

    instantiate(cfg.mode.process_func, translator, cfg)


if __name__ == "__main__":
    main()
