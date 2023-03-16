##! /bin/bash

python perturb.py XSum/xsum_50.csv -c 5 -k 500 -n 100 -s 2 -p 0.15 -w Perturbations/xsum_t50_k500_n100_s2_p15.csv
python perturb.py XSum/xsum_100.csv -c 5 -k 500 -n 100 -s 2 -p 0.15 -w Perturbations/xsum_t100_k500_n100_s2_p15.csv

python perturb.py WritingPrompts/wp_00.csv -c 5 -k 500 -n 100 -s 2 -p 0.15 -w Perturbations/wp_t00_k500_n100_s2_p15.csv
python perturb.py WritingPrompts/wp_50.csv -c 5 -k 500 -n 100 -s 2 -p 0.15 -w Perturbations/wp_t50_k500_n100_s2_p15.csv
python perturb.py WritingPrompts/wp_100.csv -c 5 -k 500 -n 100 -s 2 -p 0.15 -w Perturbations/wp_t100_k500_n100_s2_p15.csv