"""Composes single dataset of UN ru texts (only from 2014)"""
# Considering only texts from 2014
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree as ET

import pandas as pd
from tqdm import tqdm

MIN_LEN = 15
MAX_LEN = 30


def parse_xml_file(xml_path: str | Path) -> Dict[str, List]:
    """
    Extracts sentences from xml files and saves alognside with
    corresponding ids and pathes to source files.

    Args:
        xml_path: path to xml file with UN recordings

    Returns:
        dictionary with parsed sentences, their ids and pathes
        to source file.
    """

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    dataset_template = {
        'sentence': [],
        'id': [],
        'xml_file': []
    }

    for paragraph in root.findall("text/body/p"):
        for sentence in paragraph.findall("s"):
            if sentence.text and \
                 MIN_LEN <= len(sentence.text.split()) <= MAX_LEN:
                dataset_template['sentence'].append(sentence.text)
                dataset_template['id'].append(sentence.get("id"))
                dataset_template['xml_file'].append(xml_path)
    return dataset_template


def transform_xmls_to_tsv(dataset_root: str | Path) -> pd.DataFrame:
    """
    Converts parsed xmls into tsv tables.

    Args:
        dataset_root: folder with xml files with UN recordings.

    Returns:
        dataframe with sentences, their ids and pathes to source files.
    """
    if isinstance(dataset_root, str):
        dataset_root = Path(dataset_root)

    dataset_template = {
        'sentence': [],
        'id': [],
        'xml_file': []
    }

    for xml_file in tqdm(dataset_root.rglob("*.xml")):
        dataset_from_xml = parse_xml_file(xml_file)
        for key in ['sentence', 'id', 'xml_file']:
            dataset_template[key].extend(dataset_from_xml[key])
    return pd.DataFrame.from_dict(dataset_template)


if __name__ == '__main__':
    UN_dataset = transform_xmls_to_tsv("data/initial/UNv1.0-TEI/2014")
    UN_dataset.to_csv("data/initial/un_2014_ru.tsv", sep='\t')
