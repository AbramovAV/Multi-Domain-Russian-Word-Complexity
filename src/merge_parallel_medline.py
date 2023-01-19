from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def merge_parallel_texts(root_folder:str | Path) -> pd.DataFrame:
    if isinstance(root_folder, str):
        root_folder = Path(root_folder)
    corpus = defaultdict(list)
    for ru_file, en_file in tqdm(zip(sorted(root_folder.rglob("*_ru.txt")), sorted(root_folder.rglob("*_en.txt")))):
        corpus["ru"].append(ru_file.read_text())
        corpus["en"].append(en_file.read_text())
    return pd.DataFrame.from_dict(corpus)

def main(root_folder:str, save_path:str):
    parallel_corpus = merge_parallel_texts(root_folder)
    parallel_corpus.to_csv(save_path, sep="\t")

if __name__ == '__main__':
    main("data/initial/ru-en-release/", "data/initial/medline_ru_en.tsv")