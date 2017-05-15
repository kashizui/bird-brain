import argparse
import os

import numpy as np
import librosa

# Library 2
# from python_speech_features import mfcc
# from python_speech_features import delta
# from python_speech_features import logfbank

# Library 3 scikits.talkbox - doesn't worl in python 3??
#import scipy.io.wavfile
#from scikits.talkbox.features import mfcc

def process_switchboard(path):
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename[-4:] == ".sph":
                full_path = os.path.join(dirpath, filename)
                full_path = sph_to_wav(full_path)
                wave, sr = librosa.load(full_path, mono=True, sr=16000)
                mfcc_features = librosa.feature.mfcc(wave, sr=sr)
                output_filename = os.path.join(dirpath, filename[:-4] + "_mfcc")
                print(output_filename)

def sph_to_wav(full_path):
    new_path = full_path[:-3] + "wav"
    command = "sox -t sph %s -t wav %s " % (full_path, new_path)
    return new_path

def process_timit(path):
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename[-4:] == ".wav":
                full_path = os.path.join(dirpath, filename)
                wave, sr = librosa.load(full_path, mono=True, sr=16000)
                mfcc_features = librosa.feature.mfcc(wave, sr=sr, n_mfcc=12)
                delta_features = librosa.feature.delta(mfcc_features, order=1)
                output_filename = os.path.join(dirpath, filename[:-4] + "_mfcc")
                
                concat_features = np.concatenate((mfcc_features, delta_features), axis=0)
                #print(concat_features.shape)
                np.save(output_filename, concat_features, allow_pickle=False)
                
def write_output():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts MFCC features from corpus")
    parser.add_argument("corpus_path", type=str, help="Path to corpus")
    args = parser.parse_args()
    process_timit(args.corpus_path)
    #process_timit("LDC93S1_TIMIT-Acoustic-Phonetic-Continuous-Speech-Corpus/timit")