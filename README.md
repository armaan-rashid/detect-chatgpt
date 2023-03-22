# DetectChatGPT
This repo contains the code and experimental results of Armaan Rashid and Julia Park's final project for CS224N winter 2023, DetectChatGPT.

Much of the code here is a heavily adapted and refactored version of Eric Mitchell's original DetectGPT @ https://github.com/eric-mitchell/detect-gpt. If you want 
to use the code and functionality here, we ask that you please also cite the original DetectGPT and include Mitchell's MIT License (in his repo and ours)
in your use of the code. 

The core perturbation and querying functions in our repo, which is the core of the detection method, is the same as DetectGPT's original implementation
with some heavy refactoring, and adapted to the case where we are querying multiple models. That said, most of the data processing and gathering code is
original. 
