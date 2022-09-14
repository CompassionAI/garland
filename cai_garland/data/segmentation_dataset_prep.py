import os
import sys
import copy
import hydra
import random
import logging

from tqdm.auto import tqdm
import pandas as pd
from dask.distributed import Client, LocalCluster
from multiprocessing import Pool
from cai_garland.utils.str_processors import ProcessorSymbolCleaningJSON

from .parallel_dataset_sequencing import make_sequencer
from .parallel_dataset_prep import _pull_parallel_dataset


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


def _clean_tibetan(bo_text):
    bo_text = bo_text.strip()
    if not bo_text.endswith("།"):
        bo_text += "།"
    return bo_text.replace('  ', ' ')


def _make_datum(args):
    start_sent, in_sequence, positives_only = args
    pos_datum = {
        "label": 1
    }
    for num_concats, cur_sent in enumerate(_sequencing_generator(sequencer, start_sent, in_sequence)):
        if num_concats == 0:
            pos_datum["first_segment"] = _clean_tibetan(cur_sent['tibetan'])
        else:
            pos_datum["second_segment"] = _clean_tibetan(cur_sent['tibetan'])
            break
    if not num_concats == 1:
        return None
    neg_datum = None
    if not positives_only:
        concatted = pos_datum["first_segment"] + '|' + pos_datum["second_segment"]
        if ' ' in concatted:
            space_idx = random.choice([i for i, c in enumerate(concatted) if c == ' '])
            neg_datum = {
                "first_segment": _clean_tibetan(concatted[:space_idx].replace('|', ' ')),
                "second_segment": _clean_tibetan(concatted[space_idx:].replace('|', ' ')),
                "label": 0
            }
    return pos_datum, neg_datum


def _make_split_part(starting_sents, sequencer, in_sequence, positives_only=False):
    with Pool(20, initializer=_copy_sequencer, initargs=(sequencer,)) as p:
        res = list(tqdm(
            p.imap(
                _make_datum,
                [(sent, in_sequence, positives_only) for sent in starting_sents],
            ),
            total=len(starting_sents)
        ))
    res = list(zip(*filter(lambda x: x is not None, res)))
    res = res[0] + res[1]
    return list(filter(lambda x: x is not None, res))


def _prep_split_twosided(flat_data, cfg, stage_cfg):
    logger.info("Creating sequencer object")
    sequencer = make_sequencer(cfg.compute, stage_cfg.sequencing, flat_data)
    starting_sents = copy.deepcopy(flat_data)

    logger.info("Sequencing related sentences")
    res = _make_split_part(starting_sents, sequencer, in_sequence=True)
    logger.info("Sequencing unrelated sentences")
    res.extend(_make_split_part(starting_sents, sequencer, in_sequence=False))
    return res


def _prep_split_onesided(flat_data, cfg, stage_cfg, positives_end_on_segment_end=False):
    logger.info("Creating sequencer object")
    sequencer = make_sequencer(cfg.compute, stage_cfg.sequencing, flat_data)
    starting_sents = copy.deepcopy(flat_data)

    logger.info("Sequencing related sentences")
    sequences = _make_split_part(starting_sents, sequencer, in_sequence=True, positives_only=True)
    logger.info("Sequencing unrelated sentences")
    sequences.extend(_make_split_part(starting_sents, sequencer, in_sequence=False, positives_only=True))

    logger.info("Making examples")
    res = []
    for sequence in tqdm(sequences):
        assert sequence["label"] == 1
        if positives_end_on_segment_end:
            if random.random() > 0.5:
                cur_res = sequence["first_segment"]
                if random.random() > 0.5:
                    cur_res = (cur_res.strip() + ' ' + sequence["second_segment"].strip()).strip()
                res.append({
                    "segment": cur_res,
                    "label": 1
                })
            else:
                segment = sequence["first_segment"].strip()
                space_idx = random.choice([i for i, c in enumerate(segment) if c == ' '])
                res.append({
                    "segment": segment[:space_idx].strip(),
                    "label": 0
                })
        else:
            # One random positive example
            combined = sequence["first_segment"].strip() + "| " + sequence["second_segment"].strip() + " "
            space_idx = random.choice([i for i, c in enumerate(combined) if c == ' ' and i > combined.index("|")])
            segment = combined[:space_idx].replace("|", "").replace("  ", " ").strip()
            res.append({
                "segment": segment,
                "label": 1
            })

            # All possible negative examples, usually zero
            combined = sequence["first_segment"].strip() + "|" + sequence["second_segment"].strip()
            eligible_spaces = [i for i, c in enumerate(combined) if c == ' ' and i < combined.index("|")]
            for space_idx in eligible_spaces:
                space_idx = random.choice(eligible_spaces)
                segment = combined[:space_idx].replace("  ", " ").strip()
                res.append({
                    "segment": segment,
                    "label": 0
                })
    return res


_prep_split = _prep_split_onesided


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

    logger.info(f"Number of training examples is {len(training_split):,}")
    logger.info(f"Number of validation examples is {len(valid_split):,}")
    logger.info(f"Training data balance is {sum([ex['label'] for ex in training_split]) / len(training_split):.2%}")
    logger.info(f"Validation data balance is {sum([ex['label'] for ex in valid_split]) / len(valid_split):.2%}")

    logger.info("Saving to disk")
    output_dir = os.path.join(os.environ["CAI_TEMP_PATH"], "segmentation_data")
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(training_split).to_csv(os.path.join(output_dir, "train.csv"), index=False)
    pd.DataFrame(valid_split).to_csv(os.path.join(output_dir, "validation.csv"), index=False)


if __name__ == "__main__":
    main()      # pylint: disable=no-value-for-parameter
