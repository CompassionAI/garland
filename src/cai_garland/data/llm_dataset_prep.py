import os
import sys
import hydra
import shutil
import logging

from tqdm.auto import tqdm
from cai_common.data import TeiLoader
from cai_garland.utils.str_processors import ProcessorSymbolCleaningJSON
from dask.distributed import Client, LocalCluster
from colorama import init as init_colorama


init_colorama()
DATA_BASE_PATH = os.environ['CAI_DATA_BASE_PATH']
logger = logging.getLogger(__name__)

dask_logger = logging.getLogger("distributed.utils_perf")
dask_logger.setLevel(logging.ERROR)


def load_parallel_folios(cfg):
    logger.info("Loading English dataframe")
    en_df = TeiLoader('kangyur').dataframe
    en_df['location'] = en_df.location.fillna(0).astype(int)
    en_df = en_df.compute()
    en_texts = en_df \
        .sort_values(['tohoku_number', 'location']) \
        .groupby('tohoku_number') \
            .text \
            .apply(lambda x: ' '.join(x))
    return en_texts


def load_processors():
    ProcessorSymbolCleaningJSON.base_dir = os.path.dirname(__file__)
    ProcessorSymbolCleaningJSON.clean_not_skip = True

    logging.info("Loading English processors")
    processors_1 = [
        hydra.utils.instantiate(proc) 
        for proc in parallel_prep_cfg.stages['naive-concats-sequenced'].dataset.preprocessing.target_lang
    ]
    processors_2 = [hydra.utils.instantiate(proc) for proc in parallel_prep_cfg.output.postprocessing.target_lang]
    processors = processors_1 + processors_2

    return processors


def apply_processors(folios, processors):
    logging.info("Processing folios")
    for processor in tqdm(processors):
        folios = folios.apply(processor)
    return folios


@hydra.main(version_base="1.2", config_path="./dataset_prep.config", config_name="llm_dataset_prep")
def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    logger.info("Spinning up Dask cluster")
    dask_client = Client(LocalCluster(
        n_workers=cfg.compute.dask_n_workers,
        threads_per_worker=1,
        ip="localhost" if cfg.compute.local_dask_dash else "*"))
    logger.info(
        f"Dashboard is at {dask_client.dashboard_link}")

    if os.path.isdir(cfg.output.output_dir):
        if cfg.output.reset_output_dir:
            logger.info("Resetting output dir")
            shutil.rmtree(cfg.output.output_dir)
        else:
            logger.error("Output dir exists and is not being reset!")
    os.makedirs(cfg.output.output_dir, exist_ok=False)

    folios = load_parallel_folios(cfg)
    processors = load_processors()
    folios = apply_processors(folios, processors)

    logger.info("Generating training data")
    with open(os.path.join(cfg.output.output_dir, "segments.txt"), 'w') as f_out:
        for toh, text in tqdm(list(folios.items())):
            sents = list(map(lambda x: x.strip(), text.split('.')))
            for sent_i in tqdm(range(len(sents)), desc=toh, leave=False):
                prev_i = max(0, sent_i - cfg.input.num_sentences - 1)
                datum = ('. '.join(sents[prev_i:sent_i + 1]) + '.\n').replace('..', '.').replace('  ', ' ')
                f_out.write(datum)
            f_out.flush()


if __name__ == "__main__":
    with hydra.initialize(version_base="1.2", config_path="dataset_prep.config"):
        parallel_prep_cfg = hydra.compose(config_name="parallel_dataset_prep")

    main()      # pylint: disable=no-value-for-parameter
