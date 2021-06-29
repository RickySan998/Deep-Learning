import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import zipfile
import os
import gensim
import datetime
import utils
with tf.device('/device:GPU:0'):


  class Model(object):
      def __init__(self, dropout, train_emb_bool, vocabulary_matrix, node_per_MLP_layer, num_classes, input_size, sentence_embedding_size, cell_type, num_RNN_layers,learning_rate, regular_coeff, optimizer_option):

          self.graph = tf.Graph()
          self.cell_type=cell_type #Basic, LSTM of GRU
          self.input_size=input_size #this is gonna be 300 for size of emb of a word 
          self.sentence_embedding_size=sentence_embedding_size # the size of the emb output vector of RNN
          self.num_classes=num_classes
          self.node_per_MLP_layer=node_per_MLP_layer
          self.train_emb_bool=train_emb_bool
          self.lambd=regular_coeff
          self.learning_rate=learning_rate
          self.num_RNN_layers=num_RNN_layers
          self.dropout=dropout




          #Create input placeholders
          with self.graph.as_default():

              #Variable to train vocabulary of embeddings
              self.vocabulary_matrix = tf.Variable(vocabulary_matrix , name = 'embedding_vocabulary', trainable = self.train_emb_bool)
                  #[num of words,300]
                  #pad with zero index and [0,0,0] vector is 0th word in vocabulary
          
              self.input = tf.placeholder(
                  dtype=tf.int32, shape=[None, None], name='inputs')#[batch_size,num_of_words_in_this_category]
          
              self.label = tf.placeholder(
                  dtype=tf.int32,  shape=[None], name='labels') # 0 to 4
              
              # A placeholder for indicating each sequence length
              self.seq_len = tf.placeholder(
                  dtype=tf.int32, shape= [None], name='lenghts')

              # A placeholder for validaton vocabulary matrix , for training this is fed [0]
              self.vocabulary_matrix_val=tf.placeholder(
                  dtype=tf.float32, shape=[None, self.vocabulary_matrix.shape[1]], name='validation_vocabulary')# [num of words,300]

              # Num of sentences in a train/val/pred set
              #self.num_total = tf.placeholder(
              #   dtype=tf.float32, shape= (), name='num_sentences_in_set')


              # Add hidden layers
              with tf.name_scope('layers'):
                  with tf.variable_scope('network'):

                      # Checking are we training or validating
                      self.is_training = tf.reduce_all(tf.equal(self.vocabulary_matrix_val, tf.constant(0.0, dtype=tf.float32, shape=[1,1])))
                                        

                      # Choosing the right vocabulary
                      self.vocabulary_matrix_chosen = tf.cond(self.is_training,
                                          lambda :  self.vocabulary_matrix, lambda : tf.concat( [ self.vocabulary_matrix, self.vocabulary_matrix_val], 0))

          
                      #Embedding lookup on trained vocabulary
                      self.embed = tf.nn.embedding_lookup(self.vocabulary_matrix_chosen, self.input) #[batch, num_words, 300]
                      
                      #RNN network
                      self.embedded_sentence=self.RNN_layers(self.embed, self.sentence_embedding_size, self.cell_type)
                      
                      #MLP network
    
                      layer1 = self.fully(input=self.embedded_sentence, num_of_units=self.node_per_MLP_layer[0], activation=tf.nn.tanh, name='fully1')
                      layer2 = self.fully(input=layer1, num_of_units=node_per_MLP_layer[1], activation=tf.nn.tanh, name='fully2')
                      pred = self.fully(input=layer2, num_of_units=self.num_classes, activation=None, name='fully3')

                      # shape [batch_size , num_classes]
                      
                      





              # Fetch a list of our network's trainable parameters.
              self.trainable_vars = tf.trainable_variables()


              # Create output
              with tf.name_scope('output'):
              
                  
                  self.output=tf.nn.softmax(pred) #output in probabilities for every class  
                  #output is [batch_size,num_classes]
              

              # Create predictions
              with tf.name_scope('prediction'): #prediction is [batch_size]
                  self.prediction = tf.cast(tf.argmax(self.output,axis= 1),dtype=tf.int32) #choose the max probable class
              
              # Create loss
              with tf.name_scope('loss'):
    
              #    self.cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels=tf.cast(self.label,dtype=tf.float32), logits=self.prediction, pos_weight=1.65) #jer je neizbalansiran set

                  self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=pred) #use the one without sigmoid, this function has sigmoid inside 
                  
                  self.loss_basic = tf.reduce_mean(
                      self.cross_entropy, name='cross_entropy_loss')

                  #lambd = 0.000001
              # Apply L2 regularization (only to training)
                  l2_norms = [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
                  l2_norm = tf.reduce_mean(l2_norms)
                  self.loss = tf.add(self.loss_basic, self.lambd*l2_norm)

              # Choose the type of the optimiser and create training op
              with tf.name_scope('train_op'):
                  # Choose the type of the optimiser

                  # Select optimizer
                  if optimizer_option == "Adam":
                      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                  elif optimizer_option == "Adadelta":
                      self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
                  elif optimizer_option == "RMSProp":
                      self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                  elif optimizer_option == "Adagrad":
                      self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)

                      
                  # Gradient clipping
                # self.gradients_variables = self.optimizer.compute_gradients(self.loss)
                # clipped_gradients_variables = [(tf.clip_by_value(gradients, -1.0, 1.0), variables) for gradients, variables in gradients_variables]
                #  self.train_op = self.optimizer.apply_gradients(self.gradients_variables)

                  
                  self.train_op = self.optimizer.minimize(self.loss)



              # Create accuracy node
              with tf.name_scope('accuracy'):
                  self.num_correct=self.tf_count(self.label, self.prediction)
                  b=tf.cast(tf.shape(self.input)[0],tf.float32) #batch_size
                  self.accuracy=tf.divide(self.num_correct,b)


              # Create summary for monitoring training progress
              with tf.name_scope('summary'):
                  tf_loss_summary=tf.summary.scalar("loss", self.loss)
                  tf_acc_summary=tf.summary.scalar("accuracy", self.accuracy)

                    
                  self.summary = tf.summary.merge([tf_loss_summary,tf_acc_summary])  
      


      


      # Layers
                    
      def fully(self,input,num_of_units,activation,name):
          return  tf.layers.dense(inputs=input, units=num_of_units, activation=activation)

      # RNN cells
      def BasicRNNCell(self, num_of_units,act=tf.tanh):
          return tf.nn.rnn_cell.BasicRNNCell( num_of_units, activation=act)
      def LSTMCell(self, num_of_units,act=tf.tanh):
          return tf.nn.rnn_cell.LSTMCell(num_of_units, activation=act)
      
      def GRUCell(self, num_of_units,act=tf.tanh):
          return tf.nn.rnn_cell.GRUCell(num_of_units, activation=act)

      def RNN_layers(self,input_data, num_units, cell_type): #num_units is the dimension of the h output,
          # so the size of the vector that is out embedding of sentence
          cells = []
          for _ in range(self.num_RNN_layers):
              if cell_type=='Basic':
                  cell = self.BasicRNNCell(num_units) 
              else:
                  if cell_type=='LSTM':
                      cell = self.LSTMCell(num_units) 
                  else:
                      cell = self.GRUCell(num_units) 
              keep_prob = tf.cond(self.is_training, lambda:tf.constant(self.dropout), lambda:tf.constant(1.0))

              cell = tf.nn.rnn_cell.DropoutWrapper( cell, output_keep_prob = keep_prob)
              cells.append(cell)
          cell = tf.nn.rnn_cell.MultiRNNCell(cells)
          

          # Holding just the last output
          output, _ = tf.nn.dynamic_rnn(cell, input_data, sequence_length=self.seq_len, dtype=tf.float32)
          last = self.last_relevant(output, self.seq_len)
          
          #When running the model later, TensorFlow will return zero vectors for states and outputs 
          # after these sequence lengths. Therefore, weights will not affect those outputs and don’t get trained on them.
          
          
          return last #this is the new embedding of the sentence, vector size num_units
      
          # Holding just the last output
      def last_relevant(self, output, length):
          index = tf.range(0, tf.shape(self.input)[0]) * tf.shape(self.input)[1] + (length - 1)
          flat = tf.reshape(output, [-1,  self.sentence_embedding_size])
          relevant = tf.gather(flat, index)
          return relevant


      

      def save_graph_summary(self, summary_file):
          with self.graph.as_default():
              utils.ensure_parent_exists(summary_file)
              summary_writer = tf.summary.FileWriter(summary_file)
              summary_writer.add_graph(tf.get_default_graph())
              summary_writer.flush()


      def tf_count(self, t, val): #num of correct predictions

          elements_equal_to_value = tf.equal(t, val)
          as_ints = tf.cast(elements_equal_to_value, tf.float32)
          count = tf.reduce_sum(as_ints)
          
          return count
