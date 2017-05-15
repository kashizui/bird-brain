import argparse
import os
import _pickle as pickle
import shutil

import numpy as np
from string import ascii_lowercase
from string import punctuation

num_classes = 28 # 26 characters, space, blank
str_to_index = {}

def make_train(path):
    train_path = os.path.join(path, "train")
    train_output = "data/train/train.dat"
    make_dataset(train_path, train_output)
    
def make_test(path):
    test_path = os.path.join(path, "test")
    test_output = "data/test/test.dat"
    make_dataset(test_path, test_output)

def make_dataset(path, output):
    l1 = []
    l2 = []
    l3 = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename[-4:] == ".npy":
                full_path = os.path.join(dirpath, filename)
                data = np.load(full_path).T
                l1.append(data)
                label_filename = filename.split("_")[0] + ".txt"
                l2.append(get_label(os.path.join(dirpath, label_filename)))
                print(data.shape)
                l3.append(len(data))
    dataset = (l1, l2, l3)
    with open(output, "wb") as f:
        pickle.dump(dataset, f)

def get_label(path):
    with open(path, "r") as f:
        line = f.readline().strip()
        line = line.translate(line.maketrans('','',punctuation))
        split_line = line.split(" ")[2:]
        label_string = " ".join(split_line).lower()
        return string_to_index_mapping(label_string)

def string_to_index_mapping(label_string):
    return [str_to_index[c] for c in label_string]

def construct_string_to_index_mapping():
    str_to_index = {}
    offset = ord("a")
    for c in ascii_lowercase:
        str_to_index[c] = ord(c) - offset
    str_to_index[" "] = ord("z") - offset + 1
    str_to_index["_"] = ord("z") - offset + 2
    return str_to_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts MFCC features from corpus")
    parser.add_argument("corpus_path", type=str, help="Path to corpus")
    args = parser.parse_args()
    str_to_index = construct_string_to_index_mapping()
    make_train(args.corpus_path)
    make_test(args.corpus_path)