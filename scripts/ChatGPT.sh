##! /bin/bash

python detect_chatgpt.py xsum Perturbations/xsum_t00_k500_n100_s2_p15.csv davinci --perturbed -o -d xsum_davinci -k 500 -n 100
python detect_chatgpt.py squad Perturbations/squad_t00_k300_n100_s2_p15.csv davinci --perturbed -o -d squad_davinci -k 500 -n 100
python detect_chatgpt.py squad Perturbations/xsum_t50_k300_n100_s2_p15.csv davinci --perturbed -o -d squad_davinci -k 500 -n 100