# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:39:44 2018

@author: DrLC
"""

import zipfile
import json
import pickle, gzip
import os, sys
import numpy

trump_dir_path_default = os.path.split(os.path.realpath(__file__))[0]
trump_raw_path_default = os.path.join(trump_dir_path_default,
                                      "raw_trump")
trump_ex_path_default = os.path.join(trump_dir_path_default,
                                     "ex_trump")
trump_dataset_path_default = os.path.join(trump_dir_path_default,
                                          "trump.pkl.gz")
trump_letter_dict_path_default = os.path.join(trump_dir_path_default,
                                              "letter_dict.pkl.gz")

def extract_zip_all(raw_path=trump_raw_path_default,
                    ex_path=trump_ex_path_default):
    
    for root, dirs, files in os.walk(raw_path):
        for file in files:
            tmp_path = os.path.join(root, file)
            if not zipfile.is_zipfile(tmp_path):
                continue
            if "condensed" in tmp_path:
                continue
            zf = zipfile.ZipFile(tmp_path, mode="r")
            zf.extractall(ex_path)
    
def read_json_text_all(dir_path=trump_ex_path_default):
    
    data = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            tmp_path = os.path.join(root, file)
            with open(tmp_path, "r") as f:
                tmp_data = json.load(f)
            for d in tmp_data:
                if "text" in d.keys():
                    data.append(d["text"])
    return data
    
def get_letter_dict(data, threshold=89):
    
    cnt = {}
    for s in data:
        for l in s:
            if l in cnt.keys():
                cnt[l] += 1
            else:
                cnt[l] = 1
    cnt = sorted(cnt.items(), key=lambda d: d[1], reverse=True)
    letter_dict = {}
    letter_cnt = 0
    for key, val in cnt[:threshold]:
        letter_dict[key] = letter_cnt
        letter_cnt += 1
    return letter_dict
   
def save_dataset(data, path=trump_dataset_path_default):
    
    with gzip.open(path, "wb") as f:
        pickle.dump(data, f)
        
def load_dataset(path=trump_dataset_path_default):
    
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    return data
    
def save_letter_dict(letter_dict, path=trump_letter_dict_path_default):
    
    with gzip.open(path, "wb") as f:
        pickle.dump(letter_dict, f)
        
def load_letter_dict(path=trump_letter_dict_path_default):
    
    with gzip.open(path, "rb") as f:
        letter_dict = pickle.load(f)
    return letter_dict
    
if __name__ == "__main__":
    
    #extract_zip_all("raw_trump", "ex_trump")
    #data = read_json_text_all("ex_trump")
    #save_dataset(data, "trump.pkl.gz")
    
    data = load_dataset(trump_dataset_path_default)
    
    #letter = get_letter_dict(data)
    #save_letter_dict(letter)
    
    letter = load_letter_dict(trump_letter_dict_path_default)