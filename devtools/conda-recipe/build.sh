#!/bin/bash

export PYWIGNER=`pwd`
cd && git clone https://github.com/dynamiq-md/dynamiq_engine
cd dynamiq_engine && python setup.py install
cd $PYWIGNER

$PYTHON setup.py install
