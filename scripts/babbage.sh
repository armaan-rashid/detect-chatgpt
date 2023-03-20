python detect_chatgpt.py wp 00 --perturbation_file Perturbations/WritingPrompts_t00_k500_n100_s2_p15.csv --openai_query_models babbage -e
python detect_chatgpt.py wp 100 --perturbation_file Perturbations/WritingPrompts_t100_k499_n100_s2_p15.csv --openai_query_models babbage -e
python detect_chatgpt.py xsum 50 --perturbation_file Perturbations/XSum_t50_k500_n100_s2_p15.csv --openai_query_models babbage -e
python detect_chatgpt.py xsum 100 --perturbation_file Perturbations/XSum_t100_k500_n100_s2_p15.csv --openai_query_models babbage -e

