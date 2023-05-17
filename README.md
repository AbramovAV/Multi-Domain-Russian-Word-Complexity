# Description
This repository contains code and all necessary data for running experiments on multi-domain lexical complexity estimation for words from Russian language. Currently it covers three main domains, corresponding to those described in paper [CompLex: A New Corpus for Lexical Complexity Prediction from Likert Scale Data](https://arxiv.org/abs/2003.07008):
- Bible (Source: [Russian Synodal Bible](https://github.com/christos-c/bible-corpus))
- Biomedical (Source: [Biomedical parallel corpus from WMT20](https://github.com/biomedical-translation-corpora/corpora))
- Socio-political (Source: [UN Parallel Corpus](https://conferences.unite.un.org/UNCORPUS))

# Contents
- [Data preparation](#data-preparation)
- [Data annotation](#data-annotation)
- [Experiment setups](#experiment-setups)

# Data preparation
Data preparation can be splitted into consecutive 2 steps - creating intermediate represenation of original data sources and sampling data for annotation.

Creation of intermediate data representation consists of 2 independent steps - for biomedical and socio-political data respectively.

- Preparation of biomedical data is performed with `src/tools/data_preparation/merge_parallel_medline.py`. It creates parallel Russian-English unannotated corpus of biomedical records saved into a tsv file with "ru" and "en" columns.
- Preparation of socio-political data is performed with `src/tools/data_preparation/compose_un_dataset.py`. It parses XML files with UN meeting records in Russian language, extract sentences, filters them by minimum (15) and maximum number (30) of tokens in sentence, and creates tsv file with sentences, their ids and pathes to original XML files. The intermediate data does not include any alignment with English sentences. Only data from 2014 is taken into consideration due to the large volume of data.

Sampling of data and creation of input tsv files for Toloka is performed with `src/tools/data_preparation/prepare_data_for_annotation.py`. For each source of data (biomedical/socio-political) it creates a tsv file with word frequencies, surrounding contexts and lemmas, and several tsv files for Toloka each having a single column 'INPUT:text' with contexts and marked target words.

For more technical details and guidance on how to use aforementioned scripts, please, refer to the README in `src/tools/data_preparation`.

# Data annotation

[Yandex.Toloka](https://toloka.ai) is a crowdsourcing platform, that was used to annotate the data. For both data sources we have employed the same instruction and pool settings. Please, refer to the paper [Collection and evaluation of lexical complexity data for Russian language using crowdsourcing](https://journals.rudn.ru/linguistics/article/view/31331) for instruction in English (section **Methodology**). Pool settings are as follows:

- Overlap - 10 Tolokers;
- Region by phone number - Russia, Ukraine, Belarus, Kazakhstan;
- Earnings - If earned in last 24 hours $\ge$ 0.5$, then suspend in pool 1 day;
- Skipped assignments - If task suites skipped in a row $\ge$ 2, then ban on project 3 days;
- Fast responses - Minimum time per task suite — 15. Recent task suites to use — 5. If number of fast responses $\ge$ 2  and number of responses $\ge$ 2, then ban on project 7 days;
- Majority vote - Accept as majority — 5. Recent tasks to use — 10. If correct responses (%) $\le$ 50, then suspend in pool 1 day.

# Experiment setups
