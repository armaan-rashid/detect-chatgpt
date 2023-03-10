##! /bin/bash

python perturb.py XSum/xsum_00.csv davinci -k 500 -n 100 -s 2 -p 0.15 -w Perturbations/xsum_t00_k500_n100_s2_p15.csv
python perturb.py XSum/xsum_50.csv davinci -k 500 -n 100 -s 2 -p 0.15 -w Perturbations/xsum_t50_k500_n100_s2_p15.csv
python perturb.py XSum/xsum_100.csv davinci -k 500 -n 100 -s 2 -p 0.15 -w Perturbations/xsum_t100_k500_n100_s2_p15.csv
python perturb.py XSum/xsum_150.csv davinci -k 500 -n 100 -s 2 -p 0.15 -w Perturbations/xsum_t150_k500_n100_s2_p15.csv
python perturb.py XSum/xsum_200.csv davinci -k 500 -n 100 -s 2 -p 0.15 -w Perturbations/xsum_t200_k500_n100_s2_p15.csv


python perturb.py SQuAD/squad_00.csv davinci -k 300 -n 100 -s 2 -p 0.15 -w Perturbations/squad_t00_k300_n100_s2_p15.csv
python perturb.py SQuAD/squad_50.csv davinci -k 300 -n 100 -s 2 -p 0.15 -w Perturbations/squad_t50_k300_n100_s2_p15.csv
python perturb.py SQuAD/squad_100.csv davinci -k 300 -n 100 -s 2 -p 0.15 -w Perturbations/squad_t100_k300_n100_s2_p15.csv
python perturb.py SQuAD/squad_150.csv davinci -k 300 -n 100 -s 2 -p 0.15 -w Perturbations/squad_t150_k300_n100_s2_p15.csv
python perturb.py SQuAD/squad_200.csv davinci -k 300 -n 100 -s 2 -p 0.15 -w Perturbations/squad_t200_k300_n100_s2_p15.csv