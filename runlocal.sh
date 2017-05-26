#!/bin/bash

venv/bin/python3 src/basic_model.py --train-path ../data/hw3_train.dat --val-path ../data/hw3_val.dat "${@}"
