#!/bin/sh
# Run ipython notebook tests

cd examples
testfail=0
#python ipynbtest.py "alanine.ipynb" || testfail=1
cd ../..
if [ $testfail -eq 1 ]
then
    exit 1
fi

