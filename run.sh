#!/usr/bin/env bash

echo "-----------------------------------------"
echo "Test the prediction accuracy of the model"
echo "-----------------------------------------"

python dnnspmv/model/spmv_model.py test

echo "-----------------------------------------"
echo "Test the predict of a particular matrix:"
echo "-----------------------------------------"

python dnnspmv/model/spmv_model.py predict dnnspmv/data/1.mtx