import logging
import os
import sys
import math
import random
from itertools import zip_longest
from copy import deepcopy

import hydra
from hydra.utils import instantiate
from tqdm import tqdm
from colorama import init as init_colorama, Fore as ForeColor

from dask.distributed import Client, LocalCluster
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cai_common.data import ParallelTMXLoader, TeiLoader, KangyurLoader
from cai_manas.tokenizer import CAITokenizer
from cai_garland.utils.str_processors import ProcessorSymbolCleaningJSON


init_colorama()
DATA_BASE_PATH = os.environ['CAI_DATA_BASE_PATH']
logger = logging.getLogger(__name__)


dask_logger = logging.getLogger("distributed.utils_perf")
dask_logger.setLevel(logging.ERROR)


class DuplicateFolioException(Exception):
    pass


def _preprocess_flat_data(flat_data, cfg):
    # Preprocess training data and pack it into source-target dictionaries
    if len(flat_data) == 0:
        return []
    src_processors = [instantiate(proc) for proc in cfg.input.preprocessing.source_lang]
    tgt_processors = [instantiate(proc) for proc in cfg.input.preprocessing.target_lang]
    for src_processor, tgt_processor in zip_longest(src_processors, tgt_processors, fillvalue=lambda x: x):
        if isinstance(flat_data[0]['tibetan'], list):
            flat_data = [
                {
                    "tibetan": [src_processor(subdatum) for subdatum in datum["tibetan"]],
                    "english": tgt_processor(datum["english"])}
                for datum in flat_data
            ]
            res = []
            for example in flat_data:
                if len(example['english']) == 0:
                    continue
                example['tibetan'] = [datum for datum in example['tibetan'] if len(datum) > 0]
                if max([len(datum) for datum in example['tibetan']]) == 0:
                    continue
                res.append(example)
        else:
            flat_data = [
                {
                    "tibetan": src_processor(datum["tibetan"]),
                    "english": tgt_processor(datum["english"])}
                for datum in flat_data
            ]
            res = []
            for example in flat_data:
                if len(example['tibetan']) == 0:
                    continue
                if len(example['english']) == 0:
                    continue
                res.append(example)
    return res


def _shuffle_concatted_dataset(flat_data, cfg, stage_cfg):
    # Use a language model to pick random next sentences that are linguistically adequate to augment the dataset
    import torch
    from transformers import BertTokenizer, BertForNextSentencePrediction

    if len(flat_data) == 0:
        return []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    model.eval()
    if cfg.cuda:
        model.cuda()

    def score_valid_next(first_sent, second_sents):
        first_sent, second_sents = first_sent.lower(), [sent.lower() for sent in second_sents]
        encoding = tokenizer([first_sent] * len(second_sents), second_sents, return_tensors='pt', padding=True)
        encoding = {key: val.cuda() for key, val in encoding.items()}
        logits = model(**encoding)[0]
        softmax = torch.nn.functional.softmax(logits)
        return [x[0] for x in softmax.tolist()]

    remaining_data = deepcopy(flat_data)
    total_data_len = len(flat_data)
    flat_data = []
    cur_sent = remaining_data.pop(random.randrange(len(remaining_data)))
    num_fails = 0
    for _ in tqdm(range(math.floor(stage_cfg.num_shuffled_elems_frac * total_data_len))):
        flat_data.append(cur_sent)
        found_next_sent = False
        for _ in range(0, stage_cfg.shuffle_elem_find_tries, cfg.batch_size):
            try_batch_idxs = [random.randrange(len(remaining_data)) for _ in range(cfg.batch_size)]
            try_candidates = [remaining_data[idx]['english'] for idx in try_batch_idxs]
            scores = score_valid_next(cur_sent['english'], try_candidates)
            to_break = False
            for score, batch_idx in zip(scores, try_batch_idxs):
                candidate_idx = batch_idx
                if score > stage_cfg.shuffle_elem_find_threshold:
                    to_break = True
                    break
            if to_break:
                found_next_sent = True
                break
        if not found_next_sent:
            num_fails += 1
        cur_sent = remaining_data.pop(candidate_idx)
    if num_fails > 0:
        logger.warning(f"    Number of times failed to find next sentence: {num_fails}")
    return flat_data


def _pull_parallel_dataset(dask_client, cfg, stage_cfg):
    # Loads flat training and test datasets of parallel sentences into memory from Dask
    logger.info("Loading Dask dataframe")

    ParallelTMXLoader.data_glob = os.path.join(cfg.input.parallel_dataset_location, "*.tmx")
    if cfg.output.sort_by_starting_index:
        logger.info("Preparing indexed join between translations and parallel sentences")
        folio_df = TeiLoader('kangyur').dataframe
        folio_df['locator'] = folio_df \
            .tohoku_number \
                .str.lower() \
                .map(lambda x: x.replace('toh', ''))\
            + '|' \
            + folio_df.location.fillna('').str.lower()
        folio_df = folio_df.set_index('locator')

        parallel_df = ParallelTMXLoader() \
            .apply_markup() \
            .clean_bad_chars() \
            .dataframe
        parallel_df['locator'] = parallel_df.tohoku.str.lower() + '|' + parallel_df.folio.str.lower()

        joined_df = parallel_df.join(folio_df, on='locator', rsuffix="_folio", how='outer')
        joined_df = joined_df[['tohoku', 'volume_number', 'location', 'tibetan', 'english', 'text']]
        joined_df['volume_number'] = joined_df.volume_number.fillna(-1).astype('int64')
        joined_df['text'] = joined_df.text.fillna('')
        joined_df['english'] = joined_df.english.fillna('')
        joined_df['start_idx'] = joined_df.apply(
            lambda row: ' '.join(row.text.split()).find(' '.join(row.english.split())), axis=1, meta=(None, 'int64'))
        joined_df = joined_df[joined_df.start_idx >= 0]
        parallel_df = dask_client.persist(joined_df) \
            [["tohoku", "tibetan", "english", "volume_number", "location", "start_idx"]]
    else:
        logger.info("Preparing parallel sentences")
        parallel_df = ParallelTMXLoader() \
            .apply_markup() \
            .clean_bad_chars() \
            .dataframe
        parallel_df = dask_client.persist(parallel_df)[["tohoku", "tibetan", "english"]]
    parallel_df["tohoku"] = parallel_df.tohoku.fillna(-1).astype(int)

    logger.info("Loading training dataframe")
    train_df = parallel_df[~parallel_df.tohoku.isin(cfg.input.test_tohoku_numbers)].compute()
    if cfg.output.sort_by_starting_index:
        train_df = train_df.sort_values(
            ["tohoku", "volume_number", "location", "start_idx"])[["tibetan", "english"]].dropna()
    train_df = train_df[["tibetan", "english"]]
    logger.info("Loading test dataframe")
    test_df = parallel_df[parallel_df.tohoku.isin(cfg.input.test_tohoku_numbers)].compute()
    if cfg.output.sort_by_starting_index:
        test_df = test_df.sort_values(
            ["tohoku", "volume_number", "location", "start_idx"])[["tibetan", "english"]].dropna()
    test_df = test_df[["tibetan", "english"]]

    if not stage_cfg.get('exclude_from_validation', False):
        logger.info("Splitting out validation data")
        train_df, val_df = train_test_split(train_df, test_size=cfg.output.validation_frac)
    else:
        val_df = pd.DataFrame({"tibetan": [], "english": []})

    logger.debug(f"Number of Tibetan characters in test data: {int(test_df.tibetan.map(len).sum())}")
    logger.debug(f"Number of sentences in test data: {int(test_df.tibetan.count())}")

    logger.info("Pivoting dataframes to dictionaries")
    train_flat_data, val_flat_data, test_flat_data = train_df.to_dict(orient="records"), \
        val_df.to_dict(orient="records"), test_df.to_dict(orient="records")

    logger.info("Pre-processing")
    train_flat_data = _preprocess_flat_data(train_flat_data, cfg)
    val_flat_data = _preprocess_flat_data(val_flat_data, cfg)
    test_flat_data = _preprocess_flat_data(test_flat_data, cfg)

    if stage_cfg.shuffle_concats:
        shuffled_train_data, shuffled_val_data, shuffled_test_data = [], [], []
        for _ in range(stage_cfg.num_shuffling_repetitions):
            shuffled_train_data.extend(_shuffle_concatted_dataset(train_flat_data, cfg, stage_cfg))
            shuffled_val_data.extend(_shuffle_concatted_dataset(val_flat_data, cfg, stage_cfg))
            shuffled_test_data.extend(_shuffle_concatted_dataset(test_flat_data, cfg, stage_cfg))
        train_flat_data, val_flat_data, test_flat_data = shuffled_train_data, shuffled_val_data, shuffled_test_data

    return train_flat_data, val_flat_data, test_flat_data


def _pull_folio_dataset(_dask_client, cfg, stage_cfg):
    # Loads flat training and test datasets of parallel folios into memory from Dask
    english_df = TeiLoader("kangyur").dataframe
    kangyur_df = KangyurLoader().remove_new_lines().dataframe

    logger.info("Loading joined translations and the Kangyur")
    kangyur_df['locator'] = kangyur_df.apply(
        lambda row: str(row.volume_number) + '|' + str(row.location),
        meta=('locator', object),
        axis=1)
    english_df['location'] = english_df.location.fillna(-1).astype(int)
    english_df['locator'] = english_df.apply(
        lambda row: str(row.volume_number) + '|' + str(row.location),
        meta=('locator', object),
        axis=1)
    kangyur_df, english_df = kangyur_df.set_index('locator'), english_df.set_index('locator')
    joined_df = kangyur_df[['filename', 'text']].join(
        english_df[['filename', 'volume_number', 'tohoku_number', 'location', 'text']],
        how='inner',
        lsuffix="_tibetan")
    local_df = joined_df.compute().sort_values('tohoku_number')

    dupes = local_df.groupby(by=local_df.index).count().text.sort_values()
    dupes = dupes[dupes > 1]

    proced_dupes = []
    for dupe_idx in dupes.index:
        dupe_df = local_df.loc[dupe_idx]
        if len(dupe_df) == 2:
            row_1, row_2 = dupe_df.iloc[0], dupe_df.iloc[1]
            if not all([row_1[col] == row_2[col] for col in
                ["filename_tibetan", "text_tibetan", "volume_number", "location"]
            ]):
                logger.warning(f"There is a dupe that doesn't look like two Tohoku numbers spanning a folio, locator="
                               f"{dupe_idx}, equality check failed.")
                continue
            if not all([row_1.tohoku_number[3:].isnumeric(), row_1.tohoku_number[3:].isnumeric()]):
                logger.warning(f"There is a dupe whose Tohoku numbers are not numeric, locator={dupe_idx}.")
                continue
            if row_1.tohoku_number == row_2.tohoku_number:
                logger.warning(f"There is a dupe that doesn't look like two Tohoku numbers spanning a folio, locator="
                               f"{dupe_idx}, Tohoku numbers are the same.")
                continue
            if not int(row_1.tohoku_number[3:]) == int(row_2.tohoku_number[3:]) - 1:
                logger.warning(f"There is a dupe that doesn't look like two Tohoku numbers spanning a folio, locator="
                               f"{dupe_idx}, Tohoku numbers not consecutive.")
                continue
            split_folio = row_1.text_tibetan.split("༄༅")
            if len(split_folio) == 1:
                logger.warning(f"There is a dupe that doesn't look like two Tohoku numbers spanning a folio, locator="
                               f"{dupe_idx}, folio does not contain ༄༅.")
                continue
            if not len(split_folio) == 2:
                logger.warning(f"There is a dupe that doesn't look like two Tohoku numbers spanning a folio, locator="
                               f"{dupe_idx}, folio does not split into 2 parts.")
                continue
            split_folio[1] = "༄༅" + split_folio[1]
            dupe_df.text_tibetan.iat[0], dupe_df.text_tibetan.iat[1] = split_folio[0], split_folio[1]
            # local_df.loc[dupe_idx] = dupe_df    # This shouldn't be needed, as dupe_df is already a view
            proced_dupes.append(dupe_idx)
    dupes = dupes.drop(proced_dupes)

    if len(dupes) > 0:
        logger.warning(f"There are {len(dupes)} duplicates in the translation and Kangyur join!")

    local_df = local_df[~local_df.index.duplicated(keep='first')] \
        .rename(columns={
            "text_tibetan": "tibetan",
            "text": "english"})[["tibetan", "english", "tohoku_number"]]

    logger.info("Splitting training and test data")
    test_tohoku_nums = ['toh' + str(num) if not str(num)[0] == 't' else str(num)
        for num in cfg.input.test_tohoku_numbers]
    train_df = local_df[~local_df.tohoku_number.isin(test_tohoku_nums)]
    test_df = local_df[local_df.tohoku_number.isin(test_tohoku_nums)]

    if not stage_cfg.get('exclude_from_validation', False):
        logger.info("Splitting out validation data")
        train_df, val_df = train_test_split(train_df, test_size=cfg.output.validation_frac)
    else:
        val_df = pd.DataFrame({"tibetan": [], "english": []})

    logger.info("Pivoting dataframes to dictionaries")
    train_flat_data = train_df[["tibetan", "english"]].to_dict(orient="records")
    val_flat_data = val_df[["tibetan", "english"]].to_dict(orient="records")
    test_flat_data = test_df[["tibetan", "english"]].to_dict(orient="records")

    logger.info("Pre-processing")
    train_flat_data = _preprocess_flat_data(train_flat_data, cfg)
    val_flat_data = _preprocess_flat_data(val_flat_data, cfg)
    test_flat_data = _preprocess_flat_data(test_flat_data, cfg)

    return train_flat_data, val_flat_data, test_flat_data


def _pull_dictionary_dataset(_dask_client, cfg, stage_cfg):
    # Loads flat training dataset of dictionary words into memory from Dask. The test dataset is always empty.
    #   Optionally also applies simple length-based heuristics to only pick out well-defined words without long
    #   dictionary entries, and picks out the shortest definition from a comma-delimited list of definitions.
    from cai_common.dict.dict import TibetanDict, TibetanEncoding
    flat_data = []
    dict_ = TibetanDict(
        glob_override=stage_cfg.dictionary_augment_glob,
        default_encoding=TibetanEncoding.UNICODE)
    for bo, ens in dict_.items():
        if not bo[-1] == '་':
            bo += '་'
        if stage_cfg.pick_best_word:
            en_lengths = max(map(len, ens))
            if en_lengths < stage_cfg.well_defined_word_max_en_len:
                ens = [en_split.strip() for en in ens for en_split in en.split(',')]
                en_lengths = list(map(len, ens))
                flat_data.append({
                    "tibetan": bo,
                    "english": ens[en_lengths.index(min(en_lengths))]})
        else:
            for en in ens:
                flat_data.append({
                    "tibetan": bo,
                    "english": en})

    logger.info("Pre-processing")
    flat_data = _preprocess_flat_data(flat_data, cfg)

    return flat_data, [], []


def _prep_linear_dataset(flat_data, _cfg, _stage_cfg, _tokenizer):
    # No preprocessing, direct passthrough of dataset
    return flat_data


def _prep_concatted_dataset(flat_data, cfg, stage_cfg, tokenizer):
    # Prepare a dataset where consecutive sentences are concatenated to form longer training examples
    bo_token_lengths = [
        len(tokenizer.encode(datum['tibetan'], add_special_tokens=False))
        for datum in tqdm(flat_data, desc="Calculating token lengths")]

    concat_window = stage_cfg.concat_window
    concatted_data = []
    for i in tqdm(range(len(flat_data)), total=len(flat_data), desc="Concatenating"):
        cur_datum = {
            "tibetan": "",
            "english": ""}
        bo_token_count = 2
        for j in range(concat_window):
            if i + j < len(flat_data):
                bo_token_count += bo_token_lengths[i + j]
                if bo_token_count > cfg.input.max_source_length:
                    continue
                cur_datum = deepcopy(cur_datum)
                cur_datum['tibetan'] = (cur_datum['tibetan'] + ' ' + flat_data[i + j]['tibetan']).strip()
                cur_datum['english'] = (cur_datum['english'] + ' ' + flat_data[i + j]['english']).strip()
                concatted_data.append(cur_datum)

    return concatted_data


def _prep_concatted_register_dataset(flat_data, cfg, stage_cfg, tokenizer):
    # Prepare a dataset where consecutive sentences are concatenated to form longer training examples and split into
    #   source language registers of a given maximum length
    from transformers import AutoTokenizer

    bo_tokenizer = tokenizer
    en_tokenizer = AutoTokenizer.from_pretrained(cfg.output.tokenizer_name)
    bo_token_lengths = [
        len(bo_tokenizer.encode(datum['tibetan'], add_special_tokens=False))
        for datum in tqdm(flat_data, desc="Calculating source token lengths")]
    en_token_lengths = [
        len(en_tokenizer.encode(datum['english'], add_special_tokens=False))
        for datum in tqdm(flat_data, desc="Calculating target token lengths")]

    concatted_data = []
    for i in tqdm(range(len(flat_data)), total=len(flat_data), desc="Segmenting"):
        # First, append the greedy segmentation
        bo_registers, bo_length, en_length, en_line = [], 0, 0, ""
        cur_idx = i
        for _ in range(stage_cfg.max_num_registers):
            bo_register, register_start = "", cur_idx
            concat_count = 0
            max_concat_count = stage_cfg.get("max_source_sentences_per_register", None)
            while sum(bo_token_lengths[register_start:cur_idx + 1]) <= cfg.input.max_source_length - 2:
                if cur_idx >= len(flat_data) or en_length + en_token_lengths[cur_idx] >= \
                    cfg.output.max_target_length - 2:
                    break
                bo_register += ' ' + flat_data[cur_idx]['tibetan']
                en_line += ' ' + flat_data[cur_idx]['english']
                bo_length += bo_token_lengths[cur_idx]
                en_length += en_token_lengths[cur_idx]
                cur_idx += 1
                concat_count += 1
                if max_concat_count is not None and concat_count >= max_concat_count:
                    break
            bo_register = bo_register.strip()
            if len(bo_register) > 0:
                bo_registers.append(bo_register)
        if bo_length == 0:
            continue
        concatted_data.append({
            'tibetan': bo_registers,
            'english': en_line.strip()})
        # Second, generate intermediate segmentations
        while stage_cfg.get("generate_intermediate_segmentation", True):
            if random.random() > stage_cfg.intermediate_segmentation_probability:
                break
            bo_registers, en_line, en_length = [], "", 0
            cur_idx = i
            for _ in range(stage_cfg.max_num_registers):
                top_idx, top_en_length = cur_idx, en_length
                while sum(bo_token_lengths[cur_idx:top_idx + 1]) <= cfg.input.max_source_length - 2:
                    if top_idx >= len(flat_data) or \
                       top_en_length + en_token_lengths[top_idx] >= cfg.output.max_target_length - 2:
                        break
                    top_en_length += en_token_lengths[top_idx]
                    top_idx += 1
                if top_idx == cur_idx:
                    break
                break_point = math.ceil(random.random() * (top_idx - cur_idx))
                break_data = flat_data[cur_idx:cur_idx + break_point]
                bo_register = ' '.join([datum['tibetan'] for datum in break_data]).strip()
                if len(bo_register) > 0:
                    bo_registers.append(bo_register)
                en_line += ' ' + ' '.join([datum['english'] for datum in break_data])
                en_length += sum(en_token_lengths[cur_idx:cur_idx + break_point])
                cur_idx += break_point
            concatted_data.append({
                'tibetan': bo_registers,
                'english': en_line.strip()})
    return concatted_data


def _prep_folio_register_dataset(flat_data, cfg, stage_cfg, tokenizer):
    # Prepare a dataset of folios that have been split into registers
    from cai_garland.utils.segmenters import SegmenterClosingShad

    concatted_data = []
    for datum in tqdm(flat_data, total=len(flat_data), desc="Segmenting"):
        # First, append the greedy segmentation
        if len(datum['tibetan']) == 0 or len(datum['english']) == 0:
            continue
        bo_segments = SegmenterClosingShad()(datum['tibetan'])
        bo_token_lengths = [len(tokenizer.encode(bo_segment, add_special_tokens=False)) for bo_segment in bo_segments]
        bo_registers, register_start, register_idx = [], 0, 0
        for _ in range(stage_cfg.max_num_registers):
            while sum(bo_token_lengths[register_start:register_idx + 1]) <= cfg.input.max_source_length - 2:
                if register_idx == len(bo_token_lengths):
                    break
                register_idx += 1
            if register_idx == register_start:
                continue
            bo_registers.append(' '.join(bo_segments[register_start:register_idx]).strip())
            register_start = register_idx
        if register_idx < len(bo_token_lengths):
            continue
        concatted_data.append({
            'tibetan': bo_registers,
            'english': datum['english']})
        # Second, generate intermediate segmentations
        while True:
            if random.random() > stage_cfg.intermediate_segmentation_probability:
                break
            for num_tries in range(stage_cfg.num_intermediate_tries):
                bo_registers, register_start, register_idx = [], 0, 0
                for _ in range(stage_cfg.max_num_registers - 1):
                    while sum(bo_token_lengths[register_start:register_idx + 1]) <= cfg.input.max_source_length - 2:
                        if register_idx == len(bo_token_lengths):
                            break
                        register_idx += 1
                    if register_idx == register_start:
                        continue
                    break_point = math.ceil((register_idx - register_start) * random.random()) + register_start
                    bo_registers.append(' '.join(bo_segments[register_start:break_point]).strip())
                    register_start = break_point
                if sum(bo_token_lengths[register_start:]) > cfg.input.max_source_length - 2:
                    continue
                bo_registers.append(' '.join(bo_segments[register_start:]).strip())
                concatted_data.append({
                    'tibetan': bo_registers,
                    'english': datum['english']})
                break
            if num_tries == stage_cfg.num_intermediate_tries:
                print("Intermediate segmentation failed for a folio!")

    return concatted_data


def _filter_skips(bo_lines, en_lines):
    bo_res, en_res = [], []
    for bo_line, en_line in zip(bo_lines, en_lines):
        if len(bo_line) == 0 or len(en_line) == 0:
            continue
        bo_res.append(bo_line)
        en_res.append(en_line)
    return bo_res, en_res


def _write_to_file(f_name, lines, preprocess_location, separator=None):
    # Save dataset lines to a file while cleaning out bad symbols
    logger.info(f"Writing {f_name}")
    if separator is not None:
        separator = separator.strip()
    preprocess_location = os.path.join(DATA_BASE_PATH, preprocess_location)
    with open(os.path.join(preprocess_location, f_name), mode="w", encoding="utf-8") as f:
        for line in lines:
            if isinstance(line, list):
                if separator is None:
                    raise ValueError("Dataset requires a separator but none provided")
                line = separator.join(line)
            f.write(line + '\n')


def _check_for_unks(f_name, cfg):
    # Check if there are any tokens in a file that encode into <unk>
    from transformers import AutoTokenizer

    en_tokenizer = AutoTokenizer.from_pretrained(cfg.output.tokenizer_name)

    decoded, encoded = [], []
    with open(os.path.join(cfg.output_dir, f_name), mode="r", encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc=f_name):
            decoded.append(line)
            encoded.append(en_tokenizer.encode(line, add_special_tokens=False))

    unk_id, unk_lines = en_tokenizer.encode(en_tokenizer.unk_token, add_special_tokens=False)[0], []
    for line, line_text in zip(encoded, decoded):
        if unk_id in line:
            unk_lines.append((line, line_text))

    if len(unk_lines) > 0:
        logger.warning(f"Unknown tokens found in {f_name}!!! Total of {len(unk_lines)} lines contain <unk>.")
        for line in unk_lines[:10]:
            logger.warning("    " + line)
        if len(unk_lines) > 9:
            logger.warning("...")


def _filter_src_lengths(bo_data, en_data, cfg, tokenizer):
    prev_len = len(bo_data)

    if prev_len == 0:
        return [], []

    if isinstance(bo_data[0], list):
        bo_token_lengths = [
            max([len(tokenizer.encode(subdatum)) for subdatum in datum])
                    for datum in tqdm(bo_data, desc="Calculating token lengths")]
        bo_data = [
            datums for datums, len_ in zip(bo_data, bo_token_lengths) if len_ < cfg.input.max_source_length]
        en_data = [
            datums for datums, len_ in zip(en_data, bo_token_lengths) if len_ < cfg.input.max_source_length]
    else:
        bo_token_lengths = [len(tokenizer.encode(datum))
                            for datum in tqdm(bo_data, desc="Calculating token lengths")]
        bo_data = [
            datum for datum, len_ in zip(bo_data, bo_token_lengths) if len_ < cfg.input.max_source_length]
        en_data = [
            datum for datum, len_ in zip(en_data, bo_token_lengths) if len_ < cfg.input.max_source_length]

    logger.info(f"Final data keeps {len(bo_data) / prev_len} of the original")

    return bo_data, en_data


def _check_equal_lengths(bo_file_name, en_file_name, preprocess_location):
    # Check if the given processed files are equal length
    preprocess_location = os.path.join(DATA_BASE_PATH, preprocess_location)
    bo_len = len(open(os.path.join(preprocess_location, bo_file_name), mode="r", encoding="utf-8").readlines())
    en_len = len(open(os.path.join(preprocess_location, en_file_name), mode="r", encoding="utf-8").readlines())
    assert bo_len == en_len


@hydra.main(version_base="1.2", config_path="./parallel_dataset_prep.config", config_name="config")
def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    if cfg.input.test_tohoku_numbers is None:
        cfg.input.test_tohoku_numbers = []

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    ProcessorSymbolCleaningJSON.base_dir = os.path.dirname(__file__)

    logger.info("Loading tokenizer")
    tokenizer = CAITokenizer.from_pretrained(CAITokenizer.get_local_model_dir(cfg.input.tokenizer_name))

    logger.info("Spinning up Dask cluster")
    dask_client = Client(LocalCluster(
        n_workers=cfg.processing.dask_workers,
        threads_per_worker=1,
        ip="localhost" if cfg.processing.local_dask_dash else "*"))
    logger.info(
        f"Dashboard is at {dask_client.dashboard_link}")

    if cfg.skip_duplicate_folio_check:
        logger.warning("Skipping check for duplicate folios!!! This is a bad idea.")
    else:
        logger.info("Checking for duplicate folios")
        folio_df = TeiLoader('kangyur').dataframe
        dupes = folio_df.groupby(
            by=[folio_df.tohoku_number, folio_df.volume_number, folio_df.location]).filename.nunique().compute()
        dupes = dupes[dupes > 1]
        if len(dupes) > 0:
            logger.warning("Duplicate folios found! You likely need to exclude them from the TeiLoader.")
            if cfg.input.error_on_duplicate_folios:
                duped_tohokus = sorted(dupes.reset_index().tohoku_number.unique())
                logger.debug(f"There are {len(duped_tohokus)} duplicates")
                logger.debug(f"Tohoku numbers: {duped_tohokus}")
                logger.debug(f"First few duplicate locations: \n{str(dupes.sort_index().head(10))}")
                raise DuplicateFolioException()
        del folio_df
        del dupes

    logger.info("Processing datasets")
    final_train_bo, final_train_en = [], []
    final_valid_bo, final_valid_en = [], []
    final_test_bo, final_test_en = [], []
    for stage_idx, stage_name in enumerate(cfg.stages):
        logger.info(f"{ForeColor.LIGHTCYAN_EX}Running step \"{stage_name}\" ({stage_idx + 1}/{len(cfg.stages)})")

        stage_cfg = cfg.stages[stage_name]

        skip_validation = stage_cfg.get('exclude_from_validation', False)
        if skip_validation:
            logger.info(f"{ForeColor.CYAN}Skipping generating validation data from stage {stage_name}")

        logger.info("Pulling data")
        train_flat_data, valid_flat_data, test_flat_data = instantiate(stage_cfg.pull_func, dask_client, cfg, stage_cfg)
        if skip_validation and len(valid_flat_data) > 0:
            raise ValueError(
                f"Validation data should have been skipped but {len(valid_flat_data)} records were returned")

        logger.info("Preparing training dataset")
        train_concat_data = instantiate(stage_cfg.prep_func, train_flat_data, cfg, stage_cfg, tokenizer)
        if not skip_validation:
            logger.info("Preparing validation dataset")
        valid_concat_data = instantiate(stage_cfg.prep_func, valid_flat_data, cfg, stage_cfg, tokenizer)
        logger.info("Preparing test dataset")
        test_concat_data = instantiate(stage_cfg.prep_func, test_flat_data, cfg, stage_cfg, tokenizer)

        def _split_data_dicts(data_dicts):
            return [datum['tibetan'] for datum in data_dicts], [datum['english'] for datum in data_dicts]

        train_bo, train_en = _split_data_dicts(train_concat_data)
        valid_bo, valid_en = _split_data_dicts(valid_concat_data)
        test_bo, test_en = _split_data_dicts(test_concat_data)

        logger.info("Deduping")
        split_by_pipe = isinstance(train_bo[0], list)
        if split_by_pipe:
            train_bo = ['|'.join(bo_segments) for bo_segments in train_bo]
        deduped = list(set(zip(train_bo, train_en)))
        train_bo, train_en = [bo for bo, _ in deduped], [en for _, en in deduped]
        if split_by_pipe:
            train_bo = [bo_segments.split('|') for bo_segments in train_bo]

        logger.info("Post-processing Tibetan dataset")

        for processor in cfg.output.postprocessing.source_lang:
            processor = instantiate(processor)
            if isinstance(train_bo[0], list):
                train_bo, valid_bo, test_bo = [
                    [
                        [processor(subdatum) for subdatum in datum]
                        for datum in dataset
                    ]
                    for dataset in (train_bo, valid_bo, test_bo)
                ]
            else:
                train_bo, valid_bo, test_bo = [
                    [processor(datum) for datum in dataset] for dataset in (train_bo, valid_bo, test_bo)]

        if cfg.filter_longer_than_max_source_length:
            logger.info("Filtering out Tibetan examples that tokenize longer than max_source_length")
            train_bo, train_en = _filter_src_lengths(train_bo, train_en, cfg, tokenizer)
            valid_bo, valid_en = _filter_src_lengths(valid_bo, valid_en, cfg, tokenizer)
            test_bo, test_en = _filter_src_lengths(test_bo, test_en, cfg, tokenizer)

        logger.info("Post-processing English dataset")
        for processor in tqdm(cfg.output.postprocessing.target_lang):
            processor = instantiate(processor)
            train_en, valid_en, test_en = [
                [processor(datum) for datum in dataset] for dataset in [train_en, valid_en, test_en]]

        logger.info("Appending results")
        final_train_bo.extend(train_bo)
        final_train_en.extend(train_en)
        final_valid_bo.extend(valid_bo)
        final_valid_en.extend(valid_en)
        final_test_bo.extend(test_bo)
        final_test_en.extend(test_en)

    logger.info("Writing final datasets to disk")

    logger.info("Filtering out skipped lines")
    final_train_bo, final_train_en = _filter_skips(final_train_bo, final_train_en)
    final_valid_bo, final_valid_en = _filter_skips(final_valid_bo, final_valid_en)
    final_test_bo, final_test_en = _filter_skips(final_test_bo, final_test_en)

    os.makedirs(cfg.output_dir, exist_ok=True)
    separator_ = cfg.output.get("separator", None)
    _write_to_file("train.bo", final_train_bo, cfg.output_dir, separator=separator_)
    _write_to_file("train.en", final_train_en, cfg.output_dir)
    _write_to_file("valid.bo", final_valid_bo, cfg.output_dir, separator=separator_)
    _write_to_file("valid.en", final_valid_en, cfg.output_dir)
    _write_to_file("test.bo", final_test_bo, cfg.output_dir, separator=separator_)
    _write_to_file("test.en", final_test_en, cfg.output_dir)

    logger.info("Checking for equal lengths")
    logger.info("    Trainining")
    _check_equal_lengths("train.bo", "train.en", cfg.output_dir)
    logger.info("    Validation")
    _check_equal_lengths("valid.bo", "valid.en", cfg.output_dir)
    logger.info("    Test")
    _check_equal_lengths("test.bo", "test.en", cfg.output_dir)

    if cfg.output.check_for_en_unks:
        logger.info("Checking for unknown tokens")
        _check_for_unks("train.en", cfg)
        _check_for_unks("valid.en", cfg)
        _check_for_unks("test.en", cfg)


if __name__ == "__main__":
    main()      # pylint: disable=no-value-for-parameter
