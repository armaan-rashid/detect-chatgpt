##! /bin/bash

python detect_chatgpt.py xsum XSum/xsum_00.csv davinci -k 500 --no_run \ 
        -n 100 -s 2 -p 0.15 -w Perturbations/xsum_t00_k500_n100_s2_p15.csv
python detect_chatgpt.py xsum XSum/xsum_50.csv davinci -k 500 --no_run \ 
        -n 100 -s 2 -p 0.15 -w Perturbations/xsum_t50_k500_n100_s2_p15.csv
python detect_chatgpt.py xsum XSum/xsum_100.csv davinci -k 500 --no_run \ 
        -n 100 -s 2 -p 0.15 -w Perturbations/xsum_t100_k500_n100_s2_p15.csv

python detect_chatgpt.py xsum SQuAD/squad_00.csv davinci -k 300 --no_run \ 
        -n 100 -s 2 -p 0.15 -w Perturbations/squad_t00_k300_n100_s2_p15.csv
python detect_chatgpt.py xsum SQuAD/squad_50.csv davinci -k 300 --no_run \ 
        -n 100 -s 2 -p 0.15 -w Perturbations/squad_t50_k300_n100_s2_p15.csv
python detect_chatgpt.py xsum SQuAD/squad_100.csv davinci -k 300 --no_run \ 
        -n 100 -s 2 -p 0.15 -w Perturbations/squad_t100_k300_n100_s2_p15.csv