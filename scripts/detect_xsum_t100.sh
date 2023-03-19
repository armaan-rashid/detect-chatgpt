##! /bin/bash

python detect_chatgpt.py xsum 100 --perturbation_file Perturbations/XSum_t100_k500_n100_s2_p15.csv --huggingface_query_models gpt2
python detect_chatgpt.py xsum 100 --perturbation_file Perturbations/XSum_t100_k500_n100_s2_p15.csv --huggingface_query_models facebook/opt-2.7b 
python detect_chatgpt.py xsum 100 --perturbation_file Perturbations/XSum_t100_k500_n100_s2_p15.csv --huggingface_query_models EleutherAI/gpt-neo-2.7B 
python detect_chatgpt.py xsum 100 --perturbation_file Perturbations/XSum_t100_k500_n100_s2_p15.csv --huggingface_query_models EleutherAI/gpt-j-6B
python detect_chatgpt.py xsum 100 --perturbation_file Perturbations/XSum_t100_k500_n100_s2_p15.csv --openai_query_models babbage 
