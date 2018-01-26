#!/usr/bin/env bash

ps aux | grep -i '__main__.py *' | awk '{print $2}' | xargs kill -9

sudo apt-get install git htop tmux
sudo apt-get install make build-essential libssl-dev zlib1g-dev libbz2-dev libsqlite3-dev openssl python3.5-venv

cd ~
wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz
sudo tar xzf Python-3.5.2.tgz
cd Python-3.5.2
sudo ./configure --with-ensurepip=install --prefix=/data/
sudo make altinstall
#sudo apt-get install python3.5-venv

cd ~
mkdir Virtualenvs
python3.5 -m venv Virtualenvs/tf-1.3
source Virtualenvs/tf-1.3/bin/activate
pip install --upgrade pip
pip install numpy
pip install pillow
pip install scipy
pip install sklearn
pip install tabulate
pip install xlwt
pip install python-dateutil
pip install matplotlib
pip install pyyaml
pip install networkx
#pip install tensorflow-gpu

sudo apt-get install google-perftools
export PYTHONPATH="../"
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/
export LD_PRELOAD="/usr/lib/libtcmalloc.so.4"


# Check supported whl formats for the machine
# import pip
# print(pip.pep425tags.get_supported())

# Get list of all whls
# curl -s https://storage.googleapis.com/tensorflow |xmllint --format - |grep whl

# Find the relevant 1.3.0 whl and install
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp35-cp35m-linux_x86_64.whl
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp27-none-linux_x86_64.whl

# https://stackoverflow.com/questions/42013316/after-building-tensorflow-from-source-seeing-libcudart-so-and-libcudnn-errors




# pip3 install —upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0rc2-cp27-none-linux_x86_64.whl
