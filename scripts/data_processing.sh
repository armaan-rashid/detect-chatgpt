##! /bin/bash

python data_processing.py merge --orig_file SQuAD/squad_raw.csv --sampled_file SQuAD/squad_responses_00.csv \
    --orig_cols "answers" --sampled_cols "responses" --outfile SQuAD/squad_00.csv
python data_processing.py merge --orig_file SQuAD/squad_raw.csv --sampled_file SQuAD/squad_responses_50.csv \
    --orig_cols "answers" --sampled_cols "responses" --outfile SQuAD/squad_50.csv
python data_processing.py merge --orig_file SQuAD/squad_raw.csv --sampled_file SQuAD/squad_responses_100.csv \
    --orig_cols "answers" --sampled_cols "responses" --outfile SQuAD/squad_100.csv