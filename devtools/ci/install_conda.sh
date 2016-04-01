#!/bin/sh
# based on OPS version

#sudo apt-get update

### Install Miniconda

echo travis_fold:start:install.conda
echo Install conda

MINICONDA=Miniconda2-latest-Linux-x86_64.sh
MINICONDA_MD5=$(curl -s https://repo.continuum.io/miniconda/ | grep -A3 $MINICONDA | sed -n '4p' | sed -n 's/ *<td>\(.*\)<\/td> */\1/p')
wget https://repo.continuum.io/miniconda/$MINICONDA
if [[ $MINICONDA_MD5 != $(md5sum $MINICONDA | cut -d ' ' -f 1) ]]; then
    echo "Miniconda MD5 mismatch"
    echo "Expected: $MINICONDA_MD5"
    echo "Found: $(md5sum $MINICONDA | cut -d ' ' -f 1)"
    exit 1
fi
bash $MINICONDA -b

export PATH=$HOME/miniconda2/bin:$PATH

# add omnia and update
conda config --add channels http://conda.anaconda.org/omnia
conda update --yes conda

echo travis_fold:end:install.conda
