##! /bin/bash

python detect_chatgpt.py wp --perturbation_file Perturbations/wp_t00_k500_n100_s2_p15.csv --huggingface_query_models gpt2 -n 100 -k 500 -d wp_gpt2
python detect_chatgpt.py wp --perturbation_file Perturbations/wp_t00_k500_n100_s2_p15.csv --huggingface_query_models EleutherAI/gpt-j-6B -n 100 -k 500 -d wp_gptj
python detect_chatgpt.py wp --perturbation_file Perturbations/wp_t00_k500_n100_s2_p15.csv --huggingface_query_models facebook/opt-2.7b -n 100 -k 500 -d wp_opt
python detect_chatgpt.py wp --perturbation_file Perturbations/wp_t00_k500_n100_s2_p15.csv --huggingface_query_models EleutherAI/gpt-neo-2.7B -n 100 -k 500 -d wp_neo
python detect_chatgpt.py wp --perturbation_file Perturbations/wp_t00_k500_n100_s2_p15.csv --openai_query_models babbage -n 100 -k 500 -d wp_babbage
