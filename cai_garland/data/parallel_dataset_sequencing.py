import re
import string
import random
import logging
import unicodedata
from dataclasses import dataclass
from tqdm.auto import tqdm

from colorama import init as init_colorama

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM, pipeline
from transformers.tokenization_utils import TruncationStrategy


init_colorama()
logger = logging.getLogger(__name__)


class InOrderSequencer:
    """Sequencer for parallel dataset preparation that just returns sentences in order they appear."""

    def __init__(self, inference_cfg, sequencing_cfg, flat_data):
        self.flat_data = flat_data
        self.inference_cfg = inference_cfg
        self.sequencing_cfg = sequencing_cfg
        self.num_fails = 0

    def generate(self, start_example=None):
        """Generate a sequence of adequate sentence fragments using the NLI model specified in the Hydra config.

        Args:
            start_example (str): Example to start generating the adequate sequence from. If None, pick a starting
                example at random."""
        if len(self.flat_data) == 0:
            return

        if start_example is None:
            start_example = random.choice(self.flat_data)
        cur_idx = self.flat_data.index(start_example)

        while True:
            if cur_idx >= len(self.flat_data):
                return
            yield self.flat_data[cur_idx]
            cur_idx += 1


@dataclass
class NLISequencerInferenceConfig:
    cuda: bool
    batch_size: int


@dataclass
class NLISequencerScoreCutoffs:
    entailment: float
    contradiction: float


@dataclass
class NLISequencerSequencingConfig:
    model: str
    num_candidates: int
    lookback_window: int
    temperature: float
    contradiction_probability: float
    score_cutoffs: NLISequencerScoreCutoffs


class NLISequencer:
    """Sequencer for parallel dataset preparation that uses an NLI model to pick adequate next sentence fragments at
        random from the flat dataset."""

    def __init__(self, inference_cfg, sequencing_cfg, flat_data):
        logger.info("Loading shuffling sequencer model")
        self.tokenizer = AutoTokenizer.from_pretrained(sequencing_cfg.model)
        self.model = AutoModelForSequenceClassification.from_pretrained(sequencing_cfg.model)
        self.pipeline_ = pipeline("zero-shot-classification", tokenizer=self.tokenizer, model=self.model)
        self.pipeline_.model.eval()
        if inference_cfg.cuda:
            self.pipeline_.model.cuda()
        self.flat_data = flat_data
        self.inference_cfg = inference_cfg
        self.sequencing_cfg = sequencing_cfg
        self.num_fails = 0

    def _score_for_concats(
        self, base_sent, candidate_pairs, pipeline_, temperature=1, hypothesis_template="{}", return_array=False
    ):
        # This is ripped out from the zero-shot classification pipeline code, with contradictions added
        model_outputs = []
        model_inputs = pipeline_.tokenizer(
            [
                (base_sent, hypothesis_template.format(candidate_pair['english']))
                for candidate_pair in candidate_pairs
            ],
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=TruncationStrategy.LONGEST_FIRST
        )
        model_inputs = {k: model_inputs[k].to(pipeline_.model.device) for k in pipeline_.tokenizer.model_input_names}
        outputs = pipeline_.model(**model_inputs)
        for candidate_pair, logits in zip(candidate_pairs, outputs['logits']):
            model_outputs.append({
                "candidate_pair": candidate_pair,
                "logits": logits,
            })

        logits = np.vstack([output["logits"].cpu().detach().numpy() for output in model_outputs]) / temperature
        N = logits.shape[0]
        n = len(candidate_pairs)
        num_sequences = N // n
        reshaped_outputs = logits.reshape((num_sequences, n, -1))

        contra_id = pipeline_.model.config.label2id['CONTRADICTION']
        neither_id = list({0, 1, 2} - {pipeline_.entailment_id, contra_id})[0]
        entail_logits = reshaped_outputs[..., pipeline_.entailment_id]
        contra_logits = reshaped_outputs[..., contra_id]
        neither_logits = reshaped_outputs[..., neither_id]

        if return_array:
            return np.hstack([entail_logits, contra_logits, neither_logits])
        return {
            "pairs": candidate_pairs,
            "entailment_logits": entail_logits[0, ...],
            "contradiction_logits": contra_logits[0, ...],
            "neither_logits": neither_logits[0, ...]
        }

    def _apply_score_cutoff(self, scores_pairs, cutoff, renormalize=False):
        idxes = np.argwhere(scores_pairs["scores"] > cutoff)[:,0].tolist()
        res = {
            "pairs": [scores_pairs["pairs"][i] for i in idxes],
            "scores": scores_pairs["scores"][idxes],
            "indices": idxes
        }
        if renormalize:
            res["scores"] = res["scores"] / res["scores"].sum()
        return res

    def generate(self, start_example=None):
        """Generate a sequence of adequate sentence fragments using the NLI model specified in the sequencing config.

        Args:
            start_example (str): Example to start generating the adequate sequence from. If None, pick a starting
                example at random."""
        if len(self.flat_data) == 0:
            return

        if start_example is None:
            start_example = random.choice(self.flat_data)
        cur_sent = start_example

        generated = []
        while True:
            yield cur_sent

            base_sentence = ' '.join(
                [x['english'] for x in generated[-self.sequencing_cfg.lookback_window:]] + [cur_sent['english']])
            generated.append(cur_sent)

            scores = {
                "pairs": [],
                "entailment_logits": [],
                "contradiction_logits": []
            }
            tried_idxs = []
            for _ in range(self.sequencing_cfg.num_candidates // self.inference_cfg.batch_size):
                try_batch_idxs = [random.randrange(len(self.flat_data)) for _ in range(self.inference_cfg.batch_size)]
                try_candidates = [self.flat_data[idx] for idx in try_batch_idxs]

                cur_scores = self._score_for_concats(
                    base_sentence,
                    try_candidates,
                    self.pipeline_,
                    temperature=self.sequencing_cfg.temperature
                )
                for key, val in cur_scores.items():
                    if key in scores:
                        scores[key].extend(val)
                tried_idxs.extend(try_batch_idxs)
            scores = {
                "entailment": {
                    "pairs": scores["pairs"],
                    "logits": np.array(scores["entailment_logits"])
                },
                "contradiction": {
                    "pairs": scores["pairs"],
                    "logits": np.array(scores["contradiction_logits"])
                },
            }
            # This normalizes the scores to represent how much better than average the score is. This is needed because
            #   the average score is 1 / shuffle.num_candidates.
            scores["entailment"]["scores"] = self.sequencing_cfg.num_candidates * \
                np.exp(scores["entailment"]["logits"]) / np.exp(scores["entailment"]["logits"]).sum(-1, keepdims=True)
            scores["contradiction"]["scores"] = self.sequencing_cfg.num_candidates * \
                np.exp(scores["contradiction"]["logits"]) / np.exp(scores["contradiction"]["logits"]).sum(
                    -1, keepdims=True)

            scores["entailment"] = self._apply_score_cutoff(
                scores["entailment"], self.sequencing_cfg.score_cutoffs.entailment)
            scores["contradiction"] = self._apply_score_cutoff(
                scores["contradiction"], self.sequencing_cfg.score_cutoffs.contradiction)

            all_scores = np.concatenate([
                (1 - self.sequencing_cfg.contradiction_probability) * scores["entailment"]["scores"]
                    / self.sequencing_cfg.num_candidates,
                self.sequencing_cfg.contradiction_probability * scores["contradiction"]["scores"]
                    / self.sequencing_cfg.num_candidates
            ])
            all_scores = all_scores / all_scores.sum()
            all_pairs = scores["entailment"]["pairs"]
            all_pairs.extend(scores["contradiction"]["pairs"])

            if len(all_scores) == 0:
                self.num_fails += 1
                logger.debug("Failed to find a next sentence!")
                return
            else:
                candidate_idx = int(np.argwhere(np.random.default_rng().multinomial(1, all_scores) == 1))
                cur_sent = all_pairs[candidate_idx]


class ConsecutiveNLISequencer(NLISequencer):
    """Sequencer for parallel dataset preparation that uses an NLI model to stop in-order walks of the flat data when
        it looks like the sentence has changed."""

    def __init__(self, inference_cfg, sequencing_cfg, flat_data):
        super().__init__(inference_cfg, sequencing_cfg, flat_data)

    def generate(self, start_example=None):
        """Generate a sequence of adequate sentence fragments using the NLI model specified in the sequencing config.

        Args:
            start_example (str): Example to start generating the adequate sequence from. If None, pick a starting
                example at random."""
        if len(self.flat_data) == 0:
            return

        if start_example is None:
            start_example = random.choice(self.flat_data)
        cur_idx = self.flat_data.index(start_example)

        generated = []
        while True:
            if cur_idx == len(self.flat_data):
                return
            cur_sent = self.flat_data[cur_idx]
            yield cur_sent

            base_sentence = ' '.join(
                [x['english'] for x in generated[-self.sequencing_cfg.lookback_window:]] + [cur_sent['english']])
            generated.append(cur_sent)

            next_score = self._score_for_concats(
                base_sentence,
                [self.flat_data[cur_idx + 1]],
                self.pipeline_,
                return_array=True
            )
            next_score = 1 - np.exp(next_score[0][2]) / np.exp(next_score[0]).sum()

            if next_score < self.sequencing_cfg.score_cutoff:
                return
            cur_idx += 1


class ConsecutiveCosineSequencer:
    """Sequencer for parallel dataset preparation that uses cosine similarity in the pre-trained embedding space to
        stop in-order walks of the flat data when it looks like the sentence has changed."""

    def __init__(self, inference_cfg, sequencing_cfg, flat_data):
        logger.info("Loading shuffling sequencer model")
        self.tokenizer = AutoTokenizer.from_pretrained(sequencing_cfg.model)
        self.model = AutoModelForMaskedLM.from_pretrained(sequencing_cfg.model)
        self.model.eval()
        if inference_cfg.cuda:
            self.model.cuda()
        self.flat_data = flat_data
        self.inference_cfg = inference_cfg
        self.sequencing_cfg = sequencing_cfg
        self.num_fails = 0


    def generate(self, start_example=None):
        """Generate a sequence of adequate sentence fragments using the NLI model specified in the sequencing config.

        Args:
            start_example (str): Example to start generating the adequate sequence from. If None, pick a starting
                example at random."""
        from scipy.spatial import distance

        if len(self.flat_data) == 0:
            return

        if start_example is None:
            start_example = random.choice(self.flat_data)
        cur_idx = self.flat_data.index(start_example)

        generated = []
        while True:
            if cur_idx == len(self.flat_data):
                return
            cur_sent = self.flat_data[cur_idx]
            yield cur_sent

            base_sentence = ' '.join(
                [x['english'] for x in generated[-self.sequencing_cfg.lookback_window:]] + [cur_sent['english']])
            generated.append(cur_sent)

            cur_sent_embed = self.model(
                **self.tokenizer(self.flat_data[cur_idx + 1]['english'], return_tensors="pt"
            ).to(self.model.device)).logits[-1].cpu().detach().numpy()
            generated_embed = self.model(
                **self.tokenizer(base_sentence, return_tensors="pt").to(self.model.device)
            ).logits[-1].cpu().detach().numpy()
            similarity = distance.cosine(cur_sent_embed[0], generated_embed[0])

            if similarity < self.sequencing_cfg.score_cutoff:
                return
            cur_idx += 1


class FollowsAnywhereSequencer:
    """Sequencer for parallel dataset preparation that takes a next adequate sentence to be whenever the next in-order
        sentence ever shows up as the next following sentence in any translation of any text.

    The sentences are first stripped of accents, lower-cased, consecutive spaces are removed, and then everything is
        removed except spaces and lowercase English letters. The search is performed in this.
    
    Can also fix the casing while sequencing by comparing against the long-form translations.
    """

    fix_casing = False

    def _preprocess(self, text):
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if c in self.final_allowed])
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def __init__(self, inference_cfg, sequencing_cfg, flat_data):
        logger.info("Loading folios for sequencer")

        # This assumes a Dask cluster is already configured
        from cai_common.data import TeiLoader
        folio_df = TeiLoader('kangyur').dataframe.compute()
        self.all_translations = {}
        for _, row in folio_df.iterrows():
            self.all_translations[row.tohoku_number] = self.all_translations.get(row.tohoku_number, "") + " " + \
                row.text
        self.final_allowed = set(string.ascii_lowercase) | set(string.ascii_uppercase) | set([' '])
        self.all_translations = {
            key: self._preprocess(val)
            for key, val in self.all_translations.items()
        }

        if sequencing_cfg.count_examples_not_in_any_translation:
            concat_translations = ' '.join(self.all_translations.values())
            counts = [(self._preprocess(ex['english']) in concat_translations) for ex in tqdm(flat_data)]
            logger.info(f"There are {len(flat_data) - sum(counts)} parallel sentences that don't match any full "
                        f"length translation text. This works out to {1 - sum(counts) / len(flat_data):.2%} of the "
                        f"data.")

        self.flat_data = flat_data
        self.inference_cfg = inference_cfg
        self.sequencing_cfg = sequencing_cfg
        self.num_fails = 0

    def compare_casing(self, sent):
        normed_sent = unicodedata.normalize('NFKD', sent)
        sent = self._preprocess(normed_sent)
        for translation in self.all_translations.values():
            lc_translation = translation.lower()
            sent_idxs = [m.start() for m in re.finditer(re.escape(sent), lc_translation)]
            if len(sent_idxs) > 0:
                prev_bitext_c, prev_translation_c, bitext_c_idx, translation_c_idx = None, None, 0, sent_idxs[0]
                normed_sent = list(normed_sent)
                while bitext_c_idx < len(normed_sent):
                    if translation_c_idx >= len(translation):
                        break
                    cur_bitext_c, cur_translation_c = normed_sent[bitext_c_idx], translation[translation_c_idx]
                    if (not cur_bitext_c in self.final_allowed) or (prev_bitext_c == ' ' and cur_bitext_c == ' '):
                        prev_bitext_c = cur_bitext_c
                        bitext_c_idx += 1
                        continue
                    if (not cur_translation_c in self.final_allowed) or \
                       (prev_translation_c == ' ' and cur_translation_c == ' ') \
                    :
                        prev_translation_c = cur_translation_c
                        translation_c_idx += 1
                        continue
                    if cur_translation_c in string.ascii_uppercase:
                        normed_sent[bitext_c_idx] = normed_sent[bitext_c_idx].capitalize()
                    prev_bitext_c, prev_translation_c = cur_bitext_c, cur_translation_c
                    bitext_c_idx += 1
                    translation_c_idx += 1
                return ''.join(normed_sent)
        return normed_sent

    def are_in_sequence(self, first, second):
        first, second = self._preprocess(first), self._preprocess(second)
        for translation in self.all_translations.values():
            lc_translation = translation.lower()
            all_cur_sent_idxs = [m.start() for m in re.finditer(re.escape(first), lc_translation)]
            all_next_sent_idxs = [m.start() for m in re.finditer(re.escape(second), lc_translation)]
            if any([(j - i - len(first)) <= 1 for i in all_cur_sent_idxs for j in all_next_sent_idxs]):
                return True
        return False
        
    def generate(self, start_example=None):
        """Generate a sequence of adequate sentence fragments using the follows-anywhere rule described in the
            docstring for this class.

        Args:
            start_example (str): Example to start generating the adequate sequence from. If None, pick a starting
                example at random."""
        if len(self.flat_data) == 0:
            return

        if start_example is None:
            start_example = random.choice(self.flat_data)
        cur_idx = self.flat_data.index(start_example)

        while True:
            cur_sent = self.flat_data[cur_idx]
            if self.fix_casing:
                cur_sent['english'] = self.compare_casing(cur_sent['english'])
            yield cur_sent
            cur_idx += 1
            if cur_idx == len(self.flat_data):
                return

            cur_sent = cur_sent['english'].lower()
            next_sent = self.flat_data[cur_idx]['english']
            if not self.are_in_sequence(cur_sent, next_sent):
                self.num_fails += 1
                return


def make_sequencer(inference_cfg, sequencing_cfg, flat_data, fix_casing=False):
    if sequencing_cfg.type == "nli":
        sequencer = NLISequencer
    elif sequencing_cfg.type == "consecutive-nli":
        sequencer = ConsecutiveNLISequencer
    elif sequencing_cfg.type == "consecutive-cosine":
        sequencer = ConsecutiveCosineSequencer
    elif sequencing_cfg.type == "follows-anywhere":
        sequencer = FollowsAnywhereSequencer
    elif sequencing_cfg.type == "in-order":
        sequencer = InOrderSequencer
    else:
        raise ValueError(f"Unknown sequencer {sequencing_cfg.type}")
    sequencer.fix_casing = fix_casing
    return sequencer(inference_cfg, sequencing_cfg, flat_data)
