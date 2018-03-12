# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 20:55:54 2018

@author: DrLC
"""

import pickle, gzip
import random
import numpy

import os
trump_dir_path_default = os.path.split(os.path.realpath(__file__))[0]
trump_dataset_path_default = os.path.join(trump_dir_path_default,
                                          "trump.pkl.gz")
trump_letter_dict_path_default = os.path.join(trump_dir_path_default,
                                              "letter_dict.pkl.gz")
trump_tokenized_dataset_path_default = os.path.join(trump_dir_path_default,
                                                    "trump_tokenized.pkl.gz")
trump_token_dict_path_default = os.path.join(trump_dir_path_default,
                                             "token_dict.pkl.gz")

class TRUMP():
    
    def __init__(self,
                 data_path=trump_dataset_path_default,
                 dict_path=trump_letter_dict_path_default,
                 valid_ratio=0.2):
        
        with gzip.open(data_path, "rb") as f:
            data = pickle.load(f)
        valid_num = int(len(data) * valid_ratio)
        shuffle_idx = random.sample(range(len(data)), len(data))
        self.__valid_data = [data[i] for i in shuffle_idx[:valid_num]]
        self.__data = [data[i] for i in shuffle_idx[valid_num:]]
        self.__available = random.sample(range(len(self.__data)),
                                         len(self.__data))

        with gzip.open(dict_path, "rb") as f:
            self.__dict = pickle.load(f)
        self.__dict["__START__"] = max(list(self.__dict.values())) + 1
        self.__dict["__END__"] = max(list(self.__dict.values())) + 1
        self.__dict_size = len(self.__dict)
        self.__rev_dict = {}
        for key in self.__dict.keys():
            self.__rev_dict[self.__dict[key]] = key
        self.__max_len = 0
        for d in data:
            if len(d) + 2 > self.__max_len:
                self.__max_len= len(d) + 2
        
        assert (len(self.__dict) == len(self.__rev_dict))
        assert (len(self.__rev_dict) == self.__dict_size)
        
    def minibatch(self, batch_size, valid=False):
        
        if valid:
            if batch_size > len(self.__valid_data):
                print ("Warning! batch_size > valid_set_size !")
                batch_size = len(self.__valid_data)
            data_idx = random.sample(range(len(self.__valid_data)),
                                     batch_size)
            data = [self.__valid_data[i] for i in data_idx]
        else:
            if batch_size > len(self.__data):
                print ("Warning! batch_size > train_set_size !")
                batch_size = len(self.__data)
            if batch_size > len(self.__available):
                self.__available = random.sample(range(len(self.__data)),
                                         len(self.__data))
            data = [self.__data[i] for i in self.__available[:batch_size]]
            self.__available = self.__available[batch_size:]
        
        batch_x = []
        batch_y = []
        batch_l = []
        for d in data:
            batch_l.append(len(d)+1)
            batch_x.append([self.__dict["__START__"]])
            batch_y.append([])
            tmp_pad = [0 for i in range(self.__max_len-batch_l[-1]+1)]
            for i in range(len(d)):
                if d[i] in self.__dict.keys():
                    tmp_idx = self.__dict[d[i]]
                else:
                    tmp_idx = self.__dict_size
                batch_y[-1].append(tmp_idx)
                batch_x[-1].append(tmp_idx)
            batch_y[-1].append(self.__dict["__END__"])
            batch_x[-1] += tmp_pad
            batch_y[-1] += tmp_pad
        batch_x = numpy.asarray(batch_x, dtype=numpy.int32)
        batch_y = numpy.asarray(batch_y, dtype=numpy.int32)
        batch_l = numpy.asarray(batch_l, dtype=numpy.int32)
        return batch_x, batch_y, batch_l
        
    def get_valid_data(self, idx=None):
        
        if idx is None:
            return self.__valid_data
        elif idx >= len(self.__valid_data):
            print ("Out of index!")
            return None
        else:
            return self.__valid_data[idx]
        
    def get_train_data(self, idx=None):
        
        if idx is None:
            return self.__data
        elif idx >= len(self.__data):
            print ("Out of index!")
            return None
        else:
            return self.__data[idx]
        
    def get_dict(self):
        
        return self.__dict
        
    def get_rev_dict(self):
        
        return self.__rev_dict
        
    def get_dict_size(self):
        
        return self.__dict_size
        
    def get_max_len(self):
        
        return self.__max_len
        
        
if __name__ == "__main__":
    
    trump = TRUMP(trump_tokenized_dataset_path_default,
                  trump_token_dict_path_default)
    #x, y, l = trump.minibatch(100000)