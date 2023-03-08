##! /bin/bash

export OPENAI_API_KEY=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
python data_querying.py xsum -i XSum/xsum_raw.csv --out_chatgpt XSum/xsum_responses_00.csv -m "Complete the following text: " \
        -t 30 -n 500 -w 200 --temperature 0.0
python data_querying.py xsum -i XSum/xsum_raw.csv --out_chatgpt XSum/xsum_responses_25.csv -m "Complete the following text: " \
        -t 30 -n 500 -w 200 --temperature 0.25
python data_querying.py xsum -i XSum/xsum_raw.csv --out_chatgpt XSum/xsum_responses_50.csv -m "Complete the following text: " \
        -t 30 -n 500 -w 200 --temperature 0.5
python data_querying.py xsum -i XSum/xsum_raw.csv --out_chatgpt XSum/xsum_responses_75.csv -m "Complete the following text: " \
        -t 30 -n 500 -w 200 --temperature 0.75
python data_querying.py xsum -i XSum/xsum_raw.csv --out_chatgpt XSum/xsum_responses_100.csv -m "Complete the following text: " \
        -t 30 -n 500 -w 200 --temperature 1.0

python data_querying.py squad -l --out_human SQuAD/squad_raw.csv -n 300

python data_querying.py squad -i SQuAD/squad_raw.csv --out_chatgpt SQuAD/squad_responses_00.csv -m "Complete the following text: " \
        -t 30 -n 300 -w 20 --temperature 0.0
python data_querying.py xsum -i SQuAD/squad_raw.csv --out_chatgpt SQuAD/squad_responses_25.csv -m "Complete the following text: " \
        -t 30 -n 300 -w 20 --temperature 0.25
python data_querying.py xsum -i SQuAD/squad_raw.csv --out_chatgpt SQuAD/squad_responses_50.csv -m "Complete the following text: " \
        -t 30 -n 300 -w 20 --temperature 0.5
python data_querying.py xsum -i SQuAD/squad_raw.csv --out_chatgpt SQuAD/squad_responses_75.csv -m "Complete the following text: " \
        -t 30 -n 300 -w 20 --temperature 0.75
python data_querying.py xsum -i SQuAD/squad_raw.csv --out_chatgpt SQuAD/squad_responses_100.csv -m "Complete the following text: " \
        -t 30 -n 300 -w 20 --temperature 1.0