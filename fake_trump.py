# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 20:48:26 2018

@author: DrLC
"""

from lstm import LSTM
import trump_data

import tensorflow as tf
import numpy

nn = None
sess = None
trump = None
trump_dict = None
trump_rev_dict = None

def load_model():
    
    global nn, sess, trump
    
    if nn is None and sess is None and trump is None:
        nn = LSTM(max_lr=0.01,
                  min_lr=0.0005,
                  seq_len=57,
                  vocab_size=2402,
                  n_embedding=300,
                  n_out=2402,
                  n_cell=300,
                  rand_seed=1234)
        trump = trump_data.TRUMP(trump_data.trump_tokenized_dataset_path_default,
                                 trump_data.trump_token_dict_path_default)
        tf_config = tf.ConfigProto()  
        tf_config.gpu_options.allow_growth = True
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession(config=tf_config)
        sess.run(init)
        nn.build_arch()
        nn.restore_arch(sess)
    
    if nn is None or sess is None or trump is None:
        nn = None
        sess = None
        trump = None
        return False
    return True

def trump_tweet():
    
    global nn, sess, trump, trump_dict, trump_rev_dict
    
    trump_dict = trump.get_dict()
    trump_rev_dict = trump.get_rev_dict()
    
    seq = [trump_dict["__START__"]]
    padding = [0 for i in range(nn.seq_len-1)]
    curr_seq_len = 1
    curr_idx_generated = 0
    
    x = numpy.asarray([seq+padding], dtype=numpy.int32)
    curr_seq = sess.run(nn.pred_prob,
                        feed_dict={nn.x: x,
                                   nn.l: [curr_seq_len],
                                   nn.bs: 1})[0]
    curr_letter_prob = curr_seq[curr_idx_generated]
    curr_letter = trump_dict["__END__"]
    tmp_prob = numpy.random.uniform()
    for i in range(len(curr_letter_prob)):
        tmp_prob -= curr_letter_prob[i]
        if tmp_prob <= 0:
            curr_letter = i
            break
    seq.append(curr_letter)
    padding = padding[1:]
    curr_seq_len += 1
    curr_idx_generated += 1
    
    while curr_seq_len < nn.seq_len:
        
        if seq[curr_seq_len-1] in trump_rev_dict.keys():
            if (trump_rev_dict[seq[curr_seq_len-1]] == "__END__"
                 or trump_rev_dict[seq[curr_seq_len-1]] == "__START__"):
                break
        
        x = numpy.asarray([seq+padding], dtype=numpy.int32)
        curr_seq = sess.run(nn.pred_prob,
                            feed_dict={nn.x: x,
                                       nn.l: [curr_seq_len],
                                       nn.bs: 1})[0]
        curr_letter_prob = curr_seq[curr_idx_generated]
        tmp_prob = numpy.random.uniform()
        for i in range(len(curr_letter_prob)):
            tmp_prob -= curr_letter_prob[i]
            if tmp_prob <= 0:
                curr_letter = i
                break
        seq.append(curr_letter)
        padding = padding[1:]
        curr_seq_len += 1
        curr_idx_generated += 1
        
    return seq
    
def trump_translate(seq):
    
    global trump_dict, trump_rev_dict
    
    sentence = ""
    for i in seq[1:]:
        if i not in trump_rev_dict.keys():
            sentence += " "
        elif trump_rev_dict[i] == "__END__":
            break
        else:
            sentence += " " + trump_rev_dict[i]
    return sentence
    
    
if __name__ == "__main__":
    
    load_model()
    
    print ("\n\n\nNUT TRUMP TWEET:")
    for i in range(5):
        seq = trump_tweet()
        sent = trump_translate(seq)
        print ("")
        print (sent)