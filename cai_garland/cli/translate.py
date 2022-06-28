import os
import sys
import logging
import hydra
from hydra.utils import instantiate

from ..utils.translator import Translator


def interactive(translator, cfg):
    print("Interactive Tibetan translation...")
    while True:
        print("===")
        bo_text = input("Tibetan (or type exit): ")
        if bo_text == "exit":
            break
        print(translator.translate(bo_text))


def batch(translator, cfg):
    raise NotImplementedError("Batch processing not yet implemented")


@hydra.main(version_base="1.2", config_path="./translate.config", config_name="config")
def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    translator = Translator(os.path.join(cfg.model.model_ckpt, cfg.model.model_size))
    instantiate(cfg.mode.process_func, translator, cfg)


if __name__ == "__main__":
    main()
