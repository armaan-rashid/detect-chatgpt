##! /bin/bash

sudo apt update -y && sudo apt upgrade -y
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs=3.3.0
