##! /bin/bash

python data_processing.py strip --strip_file XSum/xsum_responses_150.csv --strip_col prompts --strip_msg "Complete the following text: "
python data_processing.py strip --strip_file XSum/xsum_responses_200.csv --strip_col prompts --strip_msg "Complete the following text: "

python data_processing.py merge --orig_file XSum/xsum_raw.csv --sampled_file XSum/xsum_responses_150.csv \
    --orig_cols "articles" --sampled_cols "prompts, responses" --outfile XSum/xsum_150.csv
python data_processing.py merge --orig_file XSum/xsum_raw.csv --sampled_file XSum/xsum_responses_200.csv \
    --orig_cols "articles" --sampled_cols "prompts, responses" --outfile XSum/xsum_200.csv

python data_processing.py merge --orig_file SQuAD/squad_raw.csv --sampled_file SQuAD/squad_responses_150.csv \
    --orig_cols "answers" --sampled_cols "responses" --outfile SQuAD/squad_150.csv
python data_processing.py merge --orig_file SQuAD/squad_raw.csv --sampled_file SQuAD/squad_responses_200.csv \
    --orig_cols "answers" --sampled_cols "responses" --outfile SQuAD/squad_200.csv