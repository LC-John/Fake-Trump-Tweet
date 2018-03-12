# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 21:50:57 2018

@author: DrLC
"""

import tensorflow as tf
import trump_data

import numpy
import os
import pickle, gzip

#import sas_dataset

class LSTM(object):
    
    def __init__(self, max_lr=0.01,
                 min_lr=0.0005,
                 seq_len=158,
                 vocab_size=92,
                 n_embedding=100,
                 n_out=92,
                 n_cell=300,
                 rand_seed=1234):
        
        # set random seed
        tf.set_random_seed(rand_seed)
        
        # learning speed upper/lower bound
        self.max_lr = max_lr
        self.min_lr = min_lr
        # max sequence length
        self.seq_len = seq_len
        # vocabulary size / dictionary size (including "UNKNOWN")
        self.vocab_size = vocab_size
        # embedding width
        self.n_embedding = n_embedding
        # class number
        self.n_out = n_out
        # cell number / hidden unit number
        self.n_cell = n_cell
        
    def build_arch(self):
        
        self.x = tf.placeholder(tf.int32,
                                [None, self.seq_len],
                                name="x_raw_input")
        self.y = tf.placeholder(tf.int32, [None, self.seq_len],
                                name="y_target_label")
        self.l = tf.placeholder(tf.int32, [None, ],
                                name="length_of_each_sequence")
        self.decay = tf.placeholder(tf.float32, [],
                                    name="decay_iter_divide_decay_speed")
        self.bs = tf.placeholder(tf.int32, [],
                                 name="batch_size")
        
        # LSTM input/output parameters
        self.W = {'in': tf.Variable(tf.random_normal([self.n_embedding,
                                                      self.n_cell]),
                                    name="W_in"),
                  'out': tf.Variable(tf.random_normal([self.n_cell,
                                                       self.n_out]),
                                     name="W_out")}
        self.b = {'in': tf.Variable(tf.constant(0.1,shape=[self.n_cell,]),
                                    name="b_in"),
                  'out': tf.Variable(tf.constant(0.1,shape=[self.n_out,]),
                                     name="b_out")}
        # embedding matrix
        self.embedding = tf.Variable(tf.random_uniform([self.vocab_size,
                                                        self.n_embedding]),
                                     name="embedding_matrix")
                  
        # extract the embedded sequences
        self.embedded_x = tf.nn.embedding_lookup(self.embedding,
                                                 self.x,
                                                 name="x_embedded")
        # reshape the 3d tensor into 2d tensor (flatten) in order to perform
        # a linear transformation
        self.x_flatten = tf.reshape(self.embedded_x, [-1,
                                                      self.n_embedding],
                                    name="x_flatten")
        # perform a linear transformation to fit the cell number
        self.cell_in_ = tf.matmul(self.x_flatten,
                                  self.W["in"]) + self.b["in"]
        # reshape the 2d tensor (flatten) back to 3d tensor
        # the shape is [batch_size, max sequence length, cell number]
        self.cell_in = tf.reshape(self.cell_in_,
                                  [-1, self.seq_len, self.n_cell],
                                  name="lstm_cell_input")
        
        # create the lstm cell
        with tf.variable_scope('LSTM_cell', reuse=True):
            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_cell)
            
        # initial state is zero-state 
        self.init_state = self.lstm_cell.zero_state(self.bs,
                                                    tf.float32)
        # dynamic rnn
        self.cell_out, self.final_state = tf.nn.dynamic_rnn(self.lstm_cell,
                                                            self.cell_in,
                                                            self.l,
                                                            self.init_state)
        
        # calculate learnig speed for this specific iteration
        self.lr = self.min_lr + (self.max_lr - self.min_lr) \
                * tf.pow(numpy.e, -self.decay)
        
        self.cell_out_flatten = tf.reshape(self.cell_out, [-1,
                                                           self.n_cell],
                                           name="cell_out_flatten")
        self.logits_flatten = tf.matmul(self.cell_out_flatten,
                                        self.W["out"]) + self.b["out"]
        self.logits = tf.reshape(self.logits_flatten,
                                 [-1, self.seq_len, self.n_out],
                                 name="logits")
        self.pred_prob = tf.nn.softmax(self.logits, name="prediction_probability")
        
        self.mask = tf.sequence_mask(self.l, self.seq_len,
                                     dtype=tf.float32, name="sequence_mask")
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                     targets=self.y,
                                                     weights=self.mask,
                                                     average_across_batch=True,
                                                     average_across_timesteps=True,
                                                     name="sequence_loss")
        self.final_loss = self.loss
        
        # use GD optimizer to optimize the loss
        self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.final_loss)
        
        # final output prediction
        self.pred = tf.argmax(self.pred_prob,
                              axis=2,
                              output_type=tf.int32,
                              name="prediction")
        # if each prediction is correct
        self.equal = tf.equal(self.pred, self.y,
                              name="correct_prediction")
        self.boolean_mask = tf.cast(self.mask, tf.bool,
                                    name="boolean_mask")
        self.correct = tf.logical_and(self.equal, self.boolean_mask,
                                      name="correct_pred")
        self.correct_ = tf.cast(self.correct, tf.float32,
                                name="correct_pred_cast_float32")
        # caculate accuracy
        self.acc = tf.reduce_mean(self.correct_,
                                  name="accuracy")
            
    def save_arch(self, sess, path="./model/model.ckpt"):
        
        saver = tf.train.Saver()
        save_path = saver.save(sess, path)
        return save_path
        
    def restore_arch(self, sess, path="./model/model.ckpt"):
        
        saver = tf.train.Saver()
        saver.restore(sess, path)
        

if __name__ == "__main__":
    
    epoch_num = 10
    
    batch_size = 16
    valid_size = 5000
    
    train_print_iter = 3
    valid_print_iter = 60
    save_epoch = 2
    
    nn = LSTM(max_lr=0.01,
              min_lr=0.0005,
              seq_len=57,
              vocab_size=2402,
              n_embedding=1000,
              n_out=2402,
              n_cell=1000,
              rand_seed=1234)
    nn.build_arch()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf_config = tf.ConfigProto()  
    tf_config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession(config=tf_config)
    sess.run(init)
    
    trump = trump_data.TRUMP(trump_data.trump_tokenized_dataset_path_default,
                             trump_data.trump_token_dict_path_default)
    
    iter_per_epoch = int(len(trump.get_train_data()) / batch_size)
    lr_decay = iter_per_epoch * 3
    
    tr_acc, tr_loss, va_acc = 0, 0, 0
    best_va_acc = 0
    bar_num = 30
    
    for iter in range(iter_per_epoch * epoch_num):
        
        if (iter % iter_per_epoch == 0):
            if (int(iter / iter_per_epoch) % save_epoch == 0):
                save_path = nn.save_arch(sess)
                print ("\n\tModel saved at " + save_path)
            else:
                print ("")
        
        x, y, l = trump.minibatch(batch_size, False)
        sess.run(nn.train_op, feed_dict={nn.x: x,
                                         nn.y: y,
                                         nn.l: l,
                                         nn.bs: batch_size,
                                         nn.decay: lr_decay})
        
        if iter % train_print_iter == 0:
            tr_acc, tr_loss = sess.run((nn.acc,
                                        nn.final_loss), feed_dict={nn.x: x,
                                                                   nn.y: y,
                                                                   nn.l: l,
                                                                   nn.bs: batch_size,
                                                                   nn.decay: lr_decay})
            
        if iter % valid_print_iter == 0:
            va_x, va_y, va_l = trump.minibatch(batch_size, True)
            va_acc = sess.run((nn.acc), feed_dict={nn.x: va_x,
                                                   nn.y: va_y,
                                                   nn.l: va_l,
                                                   nn.bs: batch_size,
                                                   nn.decay: lr_decay})
            if va_acc > best_va_acc:
                best_va_acc = va_acc
            
        print (("\rEp %d/%d |" %
                (iter/iter_per_epoch+1, epoch_num)), end="")
        proc = int((iter % iter_per_epoch + 1) / iter_per_epoch * bar_num)
        print ('>'*proc + '='*(bar_num-proc) + "|", end="")
        print ("[%.2f%%] (%.1f %.2f %.2f, %.2f)"
                % (100*(iter%iter_per_epoch+1)/iter_per_epoch,
                   tr_loss, tr_acc, va_acc, best_va_acc), end="")
            
    print ("\nTraining complete!")
    save_path = nn.save_arch(sess)
    print ("\n\tModel saved at " + save_path)