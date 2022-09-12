import os
import sys
import copy
import hydra
import pickle
import random
import logging

from tqdm.auto import tqdm
from dask.distributed import Client, LocalCluster
from multiprocessing import Pool
from cai_garland.utils.str_processors import ProcessorSymbolCleaningJSON

from .parallel_dataset_sequencing import make_sequencer
from .parallel_dataset_prep import _pull_parallel_dataset


DATA_BASE_PATH = os.environ['CAI_DATA_BASE_PATH']
logger = logging.getLogger(__name__)


sequencer = None


def _sequencing_generator(sequencer, start_sent, in_sequence=True):
    if in_sequence:
        for cur_sent in sequencer.generate(start_sent):
            yield cur_sent
    else:
        while True:
            yield start_sent
            while True:
                next_sent = random.choice(sequencer.flat_data)
                if not sequencer.are_in_sequence(start_sent['english'], next_sent['english']):
                    start_sent = next_sent
                    break


def _copy_sequencer(sequencer_copy):
    global sequencer
    sequencer = sequencer_copy


def _make_pos_ang_neg_datum(args):
    start_sent, in_sequence = args
    pos_datum = {
        "label": 1
    }
    for num_concats, cur_sent in enumerate(_sequencing_generator(sequencer, start_sent, in_sequence)):
        if num_concats == 0:
            pos_datum["first_segment"] = cur_sent['tibetan'].strip()
        else:
            pos_datum["second_segment"] = cur_sent['tibetan'].strip()
            break
    if not num_concats == 1:
        return None
    concatted = pos_datum["first_segment"] + '|' + pos_datum["second_segment"]
    neg_datum = None
    if ' ' in concatted:
        space_idx = random.choice([i for i, c in enumerate(concatted) if c == ' '])
        neg_datum = {
            "first_segment": concatted[:space_idx].replace('|', ' ').replace('  ', ' ').strip(),
            "second_segment": concatted[space_idx:].replace('|', ' ').replace('  ', ' ').strip(),
            "label": 0
        }
    return pos_datum, neg_datum


def _make_split_part(starting_sents, sequencer, in_sequence):
    with Pool(20, initializer=_copy_sequencer, initargs=(sequencer,)) as p:
        res = list(tqdm(
            p.imap(
                _make_pos_ang_neg_datum,
                [(sent, in_sequence) for sent in starting_sents],
            ),
            total=len(starting_sents)
        ))
    res = list(zip(*filter(lambda x: x is not None, res)))
    res = res[0] + res[1]
    return list(filter(lambda x: x is not None, res))


def _prep_split(flat_data, cfg, stage_cfg):
    logger.info("Creating sequencer object")
    sequencer = make_sequencer(cfg.compute, stage_cfg.sequencing, flat_data)
    starting_sents = copy.deepcopy(flat_data)

    logger.info("Sequencing related sentences")
    res = _make_split_part(starting_sents, sequencer, in_sequence=True)
    logger.info("Sequencing unrelated sentences")
    res.extend(_make_split_part(starting_sents, sequencer, in_sequence=False))
    return res


@hydra.main(version_base="1.2", config_path="./dataset_prep.config", config_name="segmentation_dataset_prep")
def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    random.seed(cfg.compute.seed)
    ProcessorSymbolCleaningJSON.base_dir = os.path.dirname(__file__)

    logger.info("Spinning up Dask cluster")
    dask_client = Client(LocalCluster(n_workers=10, threads_per_worker=1, ip="localhost"))
    logger.info(f"Dashboard is at {dask_client.dashboard_link}")

    stage_cfg = cfg.stages['naive-concats-sequenced']

    train_flat_data, valid_flat_data, test_flat_data = _pull_parallel_dataset(dask_client, cfg, stage_cfg)
    if len(valid_flat_data) > 0:
        raise Exception("This dataset should have no validation split")
    valid_flat_data = test_flat_data

    logger.info("Preparing training split")
    training_split = _prep_split(train_flat_data, cfg, stage_cfg)
    logger.info("Shuffling training split")
    random.shuffle(training_split)

    logger.info("Preparing validation split")
    valid_split = _prep_split(valid_flat_data, cfg, stage_cfg)
    logger.info("Shuffling validation split")
    random.shuffle(valid_split)

    logger.info(f"Training data balance is {sum([ex['label'] for ex in training_split]) / len(training_split):.2%}")
    logger.info(f"Validation data balance is {sum([ex['label'] for ex in valid_split]) / len(valid_split):.2%}")

    logger.info("Saving to disk")
    output_dir = os.path.join(os.environ["CAI_TEMP_PATH"], "segmentation_data")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), "wb") as f:
        pickle.dump(training_split, f)
    with open(os.path.join(output_dir, "valid.pkl"), "wb") as f:
        pickle.dump(valid_split, f)


if __name__ == "__main__":
    main()      # pylint: disable=no-value-for-parameter
