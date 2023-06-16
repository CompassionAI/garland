import os
import re
import sys
import copy
import hydra
import shutil
import logging

from tqdm.auto import tqdm
from cai_common.data import TeiLoader, KangyurLoader
from cai_garland.utils.str_processors import ProcessorSymbolCleaningJSON
from cai_garland.utils.translator import Translator
from dask.distributed import Client, LocalCluster
from colorama import init as init_colorama
from torch import LongTensor
from torch.nn.functional import log_softmax


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
    en_df = en_df \
        .sort_values(['volume_number', 'location']) \
        .set_index(['volume_number', 'location'])

    logger.info("Loading Tibetan dataframe")
    bo_df = KangyurLoader().remove_new_lines().dataframe
    bo_df['location'] = bo_df.location.astype(int)
    bo_df = bo_df.compute()
    bo_df = bo_df \
        .sort_values(['volume_number', 'location']) \
        .set_index(['volume_number', 'location'])

    if cfg.input.bo.complete_edge_sections:
        logger.info("Completing Tibetan edge sections")
        prev_folio = ""
        for volume_number, location in tqdm(bo_df.index):
            cur_folio = bo_df.text.at[volume_number, location]
            if len(cur_folio) == 0:
                continue

            prefix = ""
            if (volume_number, location - 1) in bo_df.index:
                if prev_folio[-1] == '་':
                    prefix = prev_folio.split(' ')[-1]

            postfix = ""
            if cur_folio[-1] == '་':
                if (volume_number, location + 1) in bo_df.index:
                    postfix = bo_df.text.at[volume_number, location + 1].strip().split(' ')[0]
            
            prev_folio = copy.copy(cur_folio)
            bo_df.text.at[volume_number, location] = prefix + cur_folio + postfix

    joined_df = en_df.join(bo_df, lsuffix="_en", rsuffix="_bo").dropna()

    return joined_df.text_bo.tolist(), joined_df.text_en.tolist()


def load_translator(cfg):
    logger.info("Loading translator")
    translator = Translator(os.path.join(cfg.model.model_name, cfg.model.model_size))

    if cfg.compute.cuda:
        logger.info("Moving translator to GPU")
        translator.cuda()

    logger.info("Loading Lotsawa segmentation config")
    generation_cfg = translator_cfg.generation
    translator.hard_segmenter = hydra.utils.instantiate(generation_cfg.segmentation.hard_segmentation)

    translator.preprocessors = [
        hydra.utils.instantiate(preproc_func)
        for preproc_func in generation_cfg.processing.preprocessing
    ]

    translator.soft_segmenter = hydra.utils.instantiate(
        generation_cfg.segmentation.soft_segmentation, translator=translator)

    translator.soft_segment_combiner_config = getattr(generation_cfg.segmentation, "soft_segment_combiner", None)
    translator.soft_segment_preprocessors = [
        hydra.utils.instantiate(preproc_func)
        for preproc_func in generation_cfg.processing.get("soft_segment_preprocessing", [])
    ]

    translator.postprocessors = [
        hydra.utils.instantiate(preproc_func)
        for preproc_func in generation_cfg.processing.postprocessing
    ]

    return translator


def load_processors(cfg):
    ProcessorSymbolCleaningJSON.base_dir = os.path.dirname(__file__)

    logging.info("Loading Tibetan processors")
    bo_processors_1 = [
        hydra.utils.instantiate(proc) 
        for proc in parallel_prep_cfg.stages['naive-concats-sequenced'].dataset.preprocessing.source_lang
    ]
    bo_processors_2 = [hydra.utils.instantiate(proc) for proc in parallel_prep_cfg.output.postprocessing.source_lang]
    bo_processors = bo_processors_1 + bo_processors_2

    logging.info("Loading English processors")
    en_processors_1 = [
        hydra.utils.instantiate(proc) 
        for proc in parallel_prep_cfg.stages['naive-concats-sequenced'].dataset.preprocessing.target_lang
    ]
    en_processors_2 = [hydra.utils.instantiate(proc) for proc in parallel_prep_cfg.output.postprocessing.target_lang]
    en_processors = en_processors_1 + en_processors_2

    return bo_processors, en_processors


def apply_processors(folios, processors):
    for processor in processors:
        folios = list(map(processor, folios))
    return list(folios)


def segment_folios(folios, translator, cfg):
    segment_fn = os.path.join(cfg.output.output_dir, "segments.bo")
    open(segment_fn, 'a').close()
    with open(segment_fn, 'r') as f:
        segments = [l.strip().split(',') for l in f.readlines()]

    with open(segment_fn, 'a') as f:
        for folio in tqdm(folios[len(segments):]):
            cur_segments = list(
                translator.segment(folio, tqdm=lambda *args, **kwargs: tqdm(*args, disable=True, **kwargs)))
            f.write(','.join(cur_segments) + '\n')
            segments.append(cur_segments)

    return segments


def _get_total_logits(bo_text, en_tokens, translator):
    bo_tokens = translator.tokenizer(bo_text, return_tensors="pt")
    model_inputs = {
        'input_ids': bo_tokens.input_ids.to(translator.model.device),
        'attention_mask': bo_tokens.attention_mask.to(translator.model.device),
        'labels': en_tokens['input_ids'].to(translator.model.device),
    }
    return -float(translator.model(**model_inputs).loss.to("cpu"))


_base_tkns = None


def _combine_tokens(*all_tokens, translator):
    global _base_tkns
    if _base_tkns is None:
        with translator.tokenizer.as_target_tokenizer():
            _base_tkns = translator.tokenizer.encode("")
    return {
        'input_ids': LongTensor([_base_tkns[:2] + sum(all_tokens, start=[]) + _base_tkns[-1:]]),
        'attention_mask': LongTensor([[1] * (len(all_tokens) + 3)]),
        'tokens_fixed': LongTensor([1])
    }


our_punct = '!&,-.:;?'
_punct_re = None


def _punctuation_segment(en_text):
    global _punct_re
    if _punct_re is None:
        _punct_re = re.compile(f"[^(?:{'|'.join(re.escape(our_punct))})]+(?:{'|'.join(re.escape(our_punct))})")
    return _punct_re.findall(en_text)


def _score_segments(
    bo_segments,
    en_text,
    translator,
    location_fudge=5,
    width_fudge=2,
    bottom_words_fudge=0.9,
    top_words_fudge=2.2
    # bottom_words_fudge=-1,
    # top_words_fudge=13
):
    scores = {}
    with translator.tokenizer.as_target_tokenizer():
        split_tokens = [translator.tokenizer.encode(t, add_special_tokens=False) for t in _punctuation_segment(en_text)]
    for seg_idx, bo_segment in tqdm(
        enumerate(bo_segments), total=len(bo_segments), leave=False, desc="Scoring segments"
    ):
        all_logits = []
        num_preceding = sum([len(s.split(' ')) for s in bo_segments[:seg_idx]])
        num_subsegments = len(bo_segment.split(' '))
        for start_idx in range(
            max(0, num_preceding - location_fudge), min(len(split_tokens), num_preceding + location_fudge)
        ):
            for length in range(
                max(1, num_subsegments - width_fudge),
                min(len(split_tokens) + 1 - start_idx, num_subsegments + width_fudge + 1)
            ):
                end_idx = start_idx + length
                cur_en_tokens = _combine_tokens(*split_tokens[start_idx:end_idx], translator=translator)
                with translator.tokenizer.as_target_tokenizer():
                    cur_en_text = translator.tokenizer.decode(cur_en_tokens['input_ids'][0][2:-1])
                bo_wordslike_count, en_words_count = bo_segment.count('་') + 1, cur_en_text.count(' ') + 1
                # words_diff = bo_wordslike_count - en_words_count
                # if words_diff < bottom_words_fudge or words_diff > top_words_fudge:
                #     continue
                words_diff = bo_wordslike_count / en_words_count
                if words_diff < bottom_words_fudge or words_diff > top_words_fudge:
                    continue
                all_logits.append(
                    ((cur_en_tokens, start_idx, end_idx), _get_total_logits(bo_segment, cur_en_tokens, translator)))
        if len(all_logits) == 0:
            scores[bo_segment] = [(({'input_ids': LongTensor([[0, 0, 0]])}, -1, -1), -100.)]
        else:
            scores[bo_segment] = sorted(all_logits, key=lambda x: -x[1])
    return scores


def _overlap(a, b, big_a, big_b):
    assert a <= b and big_a <= big_b
    return b > big_a and a < big_b


def match_segments(segments, translation, translator):
    assert len(translation) > 0 and len(segments) > 0

    segment_scores = _score_segments(segments, translation, translator)

    final_matches = {}
    while len(segment_scores) > 0:
        best_bo, ((best_en_tokens, best_en_start, best_en_end), best_score) = max(
            [(segment, scores[0]) for segment, scores in segment_scores.items()],
            key=lambda x: x[1][1])
        with translator.tokenizer.as_target_tokenizer():
            best_en = translator.tokenizer.decode(best_en_tokens['input_ids'][0,2:-1])
        final_matches[best_bo] = (best_en, best_score)

        del segment_scores[best_bo]

        new_segment_scores = {}
        for segment, scores in segment_scores.items():
            cur_segment_scores = [
                ((en, en_start, en_end), score)
                for (en, en_start, en_end), score in scores
                if not _overlap(en_start, en_end, best_en_start, best_en_end)
            ]
            if len(cur_segment_scores) > 0:
                new_segment_scores[segment] = cur_segment_scores
        segment_scores = new_segment_scores
    return final_matches    

@hydra.main(version_base="1.2", config_path="./dataset_prep.config", config_name="mined_dataset_prep")
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

    if os.path.isdir(cfg.output.output_dir) and not cfg.output.continue_from_previous:
        logger.info("Resetting output dir")
        shutil.rmtree(cfg.output.output_dir)
    os.makedirs(cfg.output.output_dir, exist_ok=True)

    translator = load_translator(cfg)

    bo_processors, en_processors = load_processors(cfg)

    bo_folios, en_folios = load_parallel_folios(cfg)

    logging.info("Applying processors to Tibetan")
    bo_folios = apply_processors(bo_folios, bo_processors)
    logging.info("Applying processors to English")
    en_folios = apply_processors(en_folios, en_processors)

    logging.info("Removing bad folios")
    zipped_folios = [(bo_f, en_f) for bo_f, en_f in zip(bo_folios, en_folios) if len(bo_f) > 0 and len(en_f) > 0]
    bo_folios, en_folios = zip(*zipped_folios)

    logging.info("Segmenting the Tibetan folios")
    bo_segments = segment_folios(bo_folios, translator, cfg)

    logging.info("Matching segments to folio candidates")
    matches_fn = os.path.join(cfg.output.output_dir, "matches.csv")
    open(matches_fn, 'a').close()
    with open(matches_fn, 'r') as f:
        proced_f_idxs = set([int(l.strip().split('|')[0]) for l in f.readlines()])
    with open(matches_fn, 'a') as f:
        for f_idx, (segments, translation) in tqdm(enumerate(list(zip(bo_segments, en_folios))), total=len(en_folios)):
            if f_idx in proced_f_idxs:
                continue
            new_matches = match_segments(segments, translation, translator)
            new_matches = [[c[0], c[1][0], c[1][1]] for c in new_matches.items()]
            for match in new_matches:
                f.write('|'.join(map(str, [f_idx] + match)) + '\n')
            f.flush()


if __name__ == "__main__":
    with hydra.initialize(version_base="1.2", config_path="../../../../lotsawa/src/lotsawa/translation_config/"):
        translator_cfg = hydra.compose(config_name="translate")

    with hydra.initialize(version_base="1.2", config_path="dataset_prep.config"):
        parallel_prep_cfg = hydra.compose(config_name="parallel_dataset_prep")

    main()      # pylint: disable=no-value-for-parameter
