# DetectChatGPT
This repo contains the code and experimental results of Armaan Rashid and Julia Park's final project for CS224N winter 2023, DetectChatGPT.

Much of the code here is a heavily adapted and refactored version of Eric Mitchell's original DetectGPT @ https://github.com/eric-mitchell/detect-gpt. If you want 
to use the code and functionality here, we ask that you please also cite the original DetectGPT and include Mitchell's MIT License (in his repo and ours)
in your use of the code. 

The core perturbation and querying functions in our repo, which is the core of the detection method, is the same as DetectGPT's original implementation
with some heavy refactoring, and adapted to the case where we are querying multiple models. The main addition we made to his original code was to break up the perturbation, querying, predicting pipeline into parts such that you can stop after perturbing or querying, save the results, and pick up where you left off later. That said, most of the data processing and gathering code is original.  

The datasets we used are in the respective files, and Perturbations are here as well. detect_chatgpt.py is the main script: run it with -h and it will tell you how to use it to perform experiments. perturb.py, data_processing.py, data_querying.py are also scripts if you want to just perturb data, process data, or get data from ChatGPT, respectively.
