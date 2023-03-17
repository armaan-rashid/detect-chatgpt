##! /bin/bash
python detect_chatgpt.py xsum --perturbation_file Perturbations/xsum_t00_k500_n100_s2_p15.csv --huggingface_query_models EleutherAI/gpt-neo-2.7B -n 100 -k 500 -d xsum_neo
python detect_chatgpt.py xsum --perturbation_file Perturbations/xsum_t00_k500_n100_s2_p15.csv --openai_query_models babbage -n 100 -k 500 -d xsum_babbage
