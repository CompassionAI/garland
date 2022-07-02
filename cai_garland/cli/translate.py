import os
import sys
import logging
import hydra
from hydra.utils import instantiate
import glob
from omegaconf import OmegaConf

from tqdm.auto import tqdm

from ..utils.translator import Translator, TokenizationTooLongException


def interactive(translator, mode_cfg, generation_cfg):
    print("Interactive Tibetan translation...")
    while True:
        print("===")
        bo_text = input("Tibetan (or type exit): ")
        if bo_text == "exit":
            break
        print(translator.translate(bo_text))


def batch(translator, mode_cfg, generation_cfg):
    if mode_cfg.input_glob is None:
        raise ValueError("Specify an input file (or glob) in the mode.input_glob setting. You can do this from the "
                         "command line.")
    os.makedirs(mode_cfg.output_dir, exist_ok=True)
    in_fns = glob.glob(mode_cfg.input_glob)
    for in_fn in tqdm(in_fns, desc="Files"):
        with open(in_fn, encoding=mode_cfg.input_encoding) as in_f:
            bo_text = in_f.read()

        in_cfg_fn = os.path.join(os.path.dirname(in_fn), os.path.splitext(os.path.basename(in_fn))[0] + '.config.yaml')
        if os.path.isfile(in_cfg_fn):
            in_cfg = OmegaConf.load(in_cfg_fn)
            generation_cfg = OmegaConf.merge(generation_cfg, in_cfg)
        else:
            generation_cfg = generation_cfg

        hard_segments = instantiate(
            generation_cfg.segmentation.hard_segmentation,
            bo_text,
            **dict(generation_cfg.segmentation.hard_segmenter_kwargs)
        )

        out_fn = os.path.join(
            mode_cfg.output_dir, os.path.splitext(os.path.basename(in_fn))[0] + '.' + mode_cfg.output_extension)
        with open(out_fn, mode='w') as out_f:
            for segment in tqdm(hard_segments, desc="Segments", leave=False):
                for preproc_func in generation_cfg.processing.preprocessing:
                    segment = instantiate(preproc_func, segment)

                try:
                    tgt_segment = translator.translate(segment)
                except TokenizationTooLongException as err:
                    if not mode_cfg.skip_long_inputs:
                        raise err
                    else:
                        tgt_segment = "SEGMENT TOKENIZATION TOO LONG FOR ENCODER MODEL"

                for postproc_func in generation_cfg.processing.postprocessing:
                    tgt_segment = instantiate(postproc_func, tgt_segment)

                if mode_cfg.output_parallel_translation:
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
    translator.num_beams = cfg.generation.generation.num_beams
    if cfg.cuda:
        translator.cuda()

    instantiate(cfg.mode.process_func, translator, cfg.mode, cfg.generation)


if __name__ == "__main__":
    main()
