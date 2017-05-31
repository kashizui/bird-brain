import argparse
import os
import _pickle as pickle
import shutil

import numpy as np
from string import ascii_lowercase
from string import punctuation

num_classes = 28  # 26 characters, space, blank
str_to_index = {}
output_dir = "data"

def make_train(path, phone):
    train_path = os.path.join(path, "train")
    train_output_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_output_dir, exist_ok=True)
    train_output = os.path.join(train_output_dir, 'train.dat')
    make_dataset(train_path, train_output, phone)
   

def make_test(path, phone):
    test_path = os.path.join(path, "test")
    test_output_dir = os.path.join(output_dir, 'test')
    os.makedirs(test_output_dir, exist_ok=True)
    test_output = os.path.join(test_output_dir, 'test.dat')
    make_dataset(test_path, test_output, phone)


def make_dataset(path, output, phone):
    l1 = []
    l2 = []
    l3 = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename[-4:] == ".npy":
                full_path = os.path.join(dirpath, filename)
                data = np.load(full_path).T
                l1.append(data)
                if phone:
                    label_filename = filename.split("_")[0] + ".phn"
                    l2.append(get_phone_label(os.path.join(dirpath, label_filename)))
                else:
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
        line = line.translate(line.maketrans('', '', punctuation))
        split_line = line.split(" ")[2:]
        label_string = " ".join(split_line).lower()
        return string_to_index_mapping(label_string)

def get_phone_label(path):
    labels = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if '#h' in line or 'h#' in line:
                continue
            line = line.translate(line.maketrans('', '', punctuation))
            word = line.split(" ")[2]
            labels.append(word)
    label_string = " ".join(labels).lower()
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
    parser = argparse.ArgumentParser(
        description="Extracts MFCC features from corpus")
    parser.add_argument("mfcc_path", type=str, help="Path to corpus")
    parser.add_argument('--phone', action='store_true')
    args = parser.parse_args()
    str_to_index = construct_string_to_index_mapping()
    make_train(args.mfcc_path, args.phone)
    make_test(args.mfcc_path, args.phone)
