##! /bin/bash

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