import os
import re
import sys
import logging
import unicodedata

from tqdm.auto import tqdm


DATA_BASE_PATH = os.environ['CAI_DATA_BASE_PATH']
logger = logging.getLogger(__name__)


def process(dataset_name, tohoku_designator):
    logging.info("Concatenating raw files in memory...")
    dataset_dir = os.path.join(DATA_BASE_PATH, "raw_datasets", dataset_name)
    full_dataset = []
    for fn in tqdm(sorted(list(os.listdir(dataset_dir)))):
        with open(os.path.join(dataset_dir, fn), encoding='utf-8') as f:
            full_dataset.append(f.read())
    full_dataset = ''.join(full_dataset)

    logging.info("Preparing admissible character set...")
    bad_char_set = filter(
        lambda c: not unicodedata.name(c, 'ERROR').startswith('TIBETAN') and not c == ' ', set(full_dataset))

    logging.info("Splitting by Tohoku numbers...")
    tohoku_re = re.compile(f"{{{tohoku_designator}(?:(?:[0-9]+)|(?:[0-9]+\-[0-9]+))\}}")
    tohoku_nums = [tnum[2:-1] for tnum in tohoku_re.findall(full_dataset)]
    tohoku_texts = tohoku_re.split(full_dataset)[1:]

    logging.info("Filtering inadmissible characters...")
    trans_dict = {ord(c): None for c in bad_char_set}
    tohoku_texts = [toh.translate(trans_dict).strip() for toh in tqdm(tohoku_texts)]
    tohokus = {tnum: t_text for tnum, t_text in zip(tohoku_nums, tohoku_texts) if len(t_text) > 0}

    return tohokus


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    dataset_path = os.path.join(DATA_BASE_PATH, "processed_datasets/tohokus")

    if os.path.exists(dataset_path):
        logging.error("Dataset path already exists, manually delete it to regenerate the dataset")
        # return
    else:
        logging.info("Creating dataset path")
        os.mkdir(dataset_path)

    logging.info("Processing Kangyur...")
    kangyur_tohokus = process("OpenPecha-kangyur", 'T')
    logging.info("Processing Tengyur...")
    tengyur_tohokus = process("Esukhia-derge-tengyur/text", 'D')

    logging.info("Saving")
    all_tohokus = kangyur_tohokus | tengyur_tohokus
    for t_num, t_text in tqdm(all_tohokus.items()):
        with open(os.path.join(dataset_path, "tohoku_" + t_num), 'w', encoding='utf-8') as f:
            f.write(t_text)


if __name__ == "__main__":
    main()
