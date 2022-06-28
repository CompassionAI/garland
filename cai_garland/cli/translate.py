import os
import sys
import logging
import hydra
from hydra.utils import instantiate
import glob
from omegaconf import OmegaConf

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
    if cfg.input_glob is None:
        raise ValueError("Specify an input file (or glob) in the mode.input_glob setting. You can do this from the "
                         "command line.")
    os.makedirs(cfg.output_dir, exist_ok=True)
    in_fns = glob.glob(cfg.input_glob)
    for in_fn in tqdm(in_fns, desc="Files"):
        with open(in_fn, encoding=cfg.encoding) as in_f:
            bo_text = in_f.read()

        in_cfg_fn = os.path.join(os.path.dirname(in_fn), os.path.splitext(os.path.basename(in_fn))[0] + '.config.yaml')
        if os.path.isfile(in_cfg_fn):
            in_cfg = OmegaConf.load(in_cfg_fn)
            cur_cfg = OmegaConf.merge(cfg, in_cfg)
        else:
            cur_cfg = cfg

        hard_segments = instantiate(cur_cfg.hard_segmentation, bo_text, **dict(cur_cfg.hard_segmenter_kwargs))

        out_fn = os.path.join(
            cur_cfg.output_dir, os.path.splitext(os.path.basename(in_fn))[0] + '.' + cur_cfg.output_extension)
        with open(out_fn, mode='w') as out_f:
            for segment in tqdm(hard_segments, desc="Segments", leave=False):
                for preproc_func in cur_cfg.preprocessing:
                    segment = instantiate(preproc_func, segment)

                try:
                    tgt_segment = translator.translate(segment)
                except TokenizationTooLongException as err:
                    if not cur_cfg.skip_long_inputs:
                        raise err
                    else:
                        tgt_segment = "SEGMENT TOKENIZATION TOO LONG FOR ENCODER MODEL"

                for postproc_func in cur_cfg.postprocessing:
                    tgt_segment = instantiate(postproc_func, tgt_segment)

                if cur_cfg.output_parallel_translation:
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

    instantiate(cfg.mode.process_func, translator, cfg.mode)


if __name__ == "__main__":
    main()
