#!/bin/bash

function install {
    echo installing "$1"
    shift
    apt-get -y install "$@" >/dev/null 2>&1
}

function pip_install {
    echo installing "$1"
    shift
    pip3 install "$@" >/dev/null 2>&1
}

function pip_upgrade {
    echo installing "$1"
    shift
    pip3 install --upgrade "$@" >/dev/null 2>&1
}

echo "updating package information"
apt-get -y update >/dev/null 2>&1


# Common
install 'pip' python3-pip
install 'hdf5' libhdf5-7 libhdf5-dev
install 'hdf5' python3-tk

pip_upgrade 'pip' pip

pip_install 'scipy' numpy scipy pandas sympy nose pillow

# Sklearn
pip_install 'sklearn' sklearn

# Keras
pip_install 'keras' keras

# Tensorflow
pip_install 'tensorflow'

# Miscellaneous
pip_install 'required Python libraries' pyyaml cython
pip_install 'h5py' h5py
pip_install 'ipython' ipython
pip_install 'jupyter' jupyter
pip_install 'matplotlib' matplotlib

echo 'All set!'