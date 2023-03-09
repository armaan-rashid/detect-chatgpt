##! /bin/bash

python data_querying.py xsum -q -i XSum/xsum_raw.csv --out_chatgpt XSum/xsum_responses_150.csv -m "Complete the following text: " \
        -k 30 -n 500 -w 200 -t 1.5
python data_querying.py xsum -q -i XSum/xsum_raw.csv --out_chatgpt XSum/xsum_responses_200.csv -m "Complete the following text: " \
        -k 30 -n 500 -w 200 -t 2.0

python data_querying.py squad -q -i SQuAD/squad_raw.csv --out_chatgpt SQuAD/squad_responses_150.csv -n 300 -w 2 -t 1.5
python data_querying.py squad -q -i SQuAD/squad_raw.csv --out_chatgpt SQuAD/squad_responses_200.csv -n 300 -w 2 -t 2.0

python data_querying.py wp -q -i WritingPrompts/wp_raw.csv -m "Write a short story, minimum length 200 words, based on the following prompt: " \
        -w 200 -n 500 -t 0.0 --out_chatgpt WritingPrompts/wp_responses_00.csv
python data_querying.py wp -q -i WritingPrompts/wp_raw.csv -m "Write a short story, minimum length 200 words, based on the following prompt: " \
        -w 200 -n 500 -t 0.5 --out_chatgpt WritingPrompts/wp_responses_50.csv 
python data_querying.py wp -q -i WritingPrompts/wp_raw.csv -m "Write a short story, minimum length 200 words, based on the following prompt: " \
        -w 200 -n 500 -t 1.0 --out_chatgpt WritingPrompts/wp_responses_100.csv 
python data_querying.py wp -q -i WritingPrompts/wp_raw.csv -m "Write a short story, minimum length 200 words, based on the following prompt: " \
        -w 200 -n 500 -t 1.5 --out_chatgpt WritingPrompts/wp_responses_150.csv 
python data_querying.py wp -q -i WritingPrompts/wp_raw.csv -m "Write a short story, minimum length 200 words, based on the following prompt: " \
        -w 200 -n 500 -t 2.0 --out_chatgpt WritingPrompts/wp_responses_200.csv 