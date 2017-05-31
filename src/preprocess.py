import argparse
import os

import numpy as np
import librosa

import python_speech_features
import scipy.io.wavfile as wavfile

from sphfile import SPHFile

# Library 2
# from python_speech_features import mfcc
# from python_speech_features import delta
# from python_speech_features import logfbank

# Library 3 scikits.talkbox - doesn't worl in python 3??
#import scipy.io.wavfile
#from scikits.talkbox.features import mfcc

def process_switchboard(path, output_path):
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename[-4:] == ".sph":
                full_path = sph_to_wav(dirpath, filename, output_path)
                wave, sr = librosa.load(full_path, mono=True, sr=16000)
                mfcc_features = librosa.feature.mfcc(wave, sr=sr, n_mfcc=12)
                print(wave.shape)
                delta_features = librosa.feature.delta(mfcc_features, order=1)
                output_filename = os.path.join(output_path, filename[:-4] + "_mfcc")
                concat_features = np.concatenate((mfcc_features, delta_features), axis=0)
                np.save(output_filename, concat_features, allow_pickle=False)

def sph_to_wav(dirpath, filename, output_path):
    full_path = os.path.join(dirpath, filename)
    new_path = os.path.join(output_path, filename[:-3] + ".wav")
    sph = SPHFile(full_path)
    sph.write_wav(new_path)
    return new_path
    # command = "sox -t sph %s -t wav %s " % (full_path, new_path)

def process_timit(path, output_path):
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename[-4:] == ".wav":
                full_path = os.path.join(dirpath, filename)
                wave, sr = librosa.load(full_path, mono=True, sr=16000)
                mfcc_features = librosa.feature.mfcc(wave, sr=sr, n_mfcc=12,hop_length=256)
                delta_features = librosa.feature.delta(mfcc_features, order=1)
                output_filename = os.path.join(dirpath, filename[:-4] + "_mfcc")
                concat_features = np.concatenate((mfcc_features, delta_features), axis=0)
                print(concat_features.shape)
                np.save(output_filename, concat_features, allow_pickle=False)
                
def process_timit_psf(path, output_path):
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename[-4:] == ".wav":
                full_path = os.path.join(dirpath, filename)
                wave, sr = librosa.load(full_path, mono=True, sr=16000)
                mfcc_features = python_speech_features.mfcc(wave, samplerate=sr, numcep=13, nfilt=26, appendEnergy=True, winlen=0.025, winstep=0.01)
                delta_features = python_speech_features.delta(mfcc_features, 9)
                
                output_filename = os.path.join(dirpath, filename[:-4] + "_mfcc")
                print(output_filename)
                concat_features = np.concatenate((mfcc_features, delta_features), axis=1).T
                print(concat_features.shape)
                np.save(output_filename, concat_features, allow_pickle=False)
                
def write_output():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts MFCC features from corpus")
    parser.add_argument("corpus_path", type=str, help="Path to corpus")
    requiredCorpus = parser.add_argument_group('Required corpus name argument')
    requiredCorpus.add_argument('-c', choices = ["timit", "switchboard"], type=str, dest="dataset", required=True, help='Corpus name')
    #parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    output_path = args.corpus_path # args.output_dir if args.output_dir else args.corpus_path
    #os.makedirs(output_path, exist_ok=True)
    #os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
    #os.makedirs(os.path.join(output_path, "test"), exist_ok=True)
    if args.dataset == "timit":
        process_timit_psf(args.corpus_path, output_path)
    else:
        process_switchboard(args.corpus_path, output_path)