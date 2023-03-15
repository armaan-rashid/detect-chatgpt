##! /bin/bash

sudo apt update && sudo apt upgrade
sudo apt install curl
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs=3.3.0
git clone https://github.com/armaan-rashid/CS224N-final-project.git
conda create -n perturb pandas transformers -c conda-forge
conda activate perturb
cd CS224N-final-project