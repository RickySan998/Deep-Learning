import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import random
import utils


with tf.device('/device:GPU:0'):


  class Trainer(object):
      def __init__(self, vocabulary_matrix_val, vocabulary_matrix_test, input_train, output_train, seq_len_train, input_val, output_val, seq_len_val, input_test, output_test, seq_len_test, model, epochs, sum_path=None, checkpoints_path=None):

          self.model = model
          self.sum_path=sum_path

          if not self.sum_path is None:
              self.model.save_graph_summary(os.path.join(self.sum_path, 'summary'))
              self.train_summary_writer = \
                  tf.summary.FileWriter(os.path.join(self.sum_path, 'train'))
              self.epoch_train_summary_writer = \
                  tf.summary.FileWriter(os.path.join(self.sum_path, 'epoch_train'))
              self.valid_summary_writer = \
                  tf.summary.FileWriter(os.path.join(self.sum_path, 'valid'))
              self.epoch_valid_summary_writer = \
                  tf.summary.FileWriter(os.path.join(self.sum_path, 'epoch_valid'))

          self.train_count=0
          self.val_count=0

          self.input_train=input_train
          self.output_train=output_train
          self.seq_len_train=seq_len_train
          self.vocabulary_matrix=0
          

          self.num_total_train=0 #total number of sentences in train set

          for i in range(len(self.input_train)):
              self.num_total_train+=self.input_train[i].shape[0]

          self.input_val=input_val
          self.output_val=output_val
          self.seq_len_val=seq_len_val
          self.vocabulary_matrix_val = vocabulary_matrix_val
          
          self.num_total_val=0 #total number of sentences in val set

          for i in range(len(self.input_val)):
              self.num_total_val+=self.input_val[i].shape[0]

          self.input_test=input_test
          self.output_test=output_test
          self.seq_len_test=seq_len_test
          self.vocabulary_matrix_test=vocabulary_matrix_test
          
          self.num_total_test=0 #total number of sentences in test set

          for i in range(len(self.input_test)):
              self.num_total_test+=self.input_test[i].shape[0]
          
          
          self._epochs_training = 0 #num  of current epoch 
          self.epochs = epochs # total epoch to train
        
          
          
          self.loss=[]  #train loss
          self.val_loss=[]  #val loss
          self.accuracy=[]  #train accuracy
          self.val_accuracy=[] #val accuracy
      
          self.checkpoints_path=checkpoints_path #where we write checkpoints
        
          # Define session for this Graph
          with self.model.graph.as_default():
              self.session = tf.Session()

          # Initializing variables and tables in the Graph and start running operations
          with self.model.graph.as_default():
              self.session.run(tf.global_variables_initializer())
      

          # Create a saver for checkpoints
          if not checkpoints_path is None:
              with self.model.graph.as_default():
                  self.saver = tf.train.Saver(self.model.trainable_vars,max_to_keep=50)


      def train(self):


          #Make folders for checkpoints
          if not self.checkpoints_path is None:   
              os.makedirs(self.checkpoints_path, exist_ok=True)
              print("New training , no weights")
          
          
          print("{} Start training...")
          self._epochs_training = 0
          self.train_count=0
          self.val_count=0
          self.epoch_train_count=0
          self.epoch_val_count=0
          
          # Train in epoch
          while self._epochs_training < self.epochs:

             # print('Training')
              self._train_epoch() #do one epoch
            
              #print("Validation")
              self.validate() 

              if self._epochs_training%10==0:
                print('Epoch num: '+str(self._epochs_training))
              self._epochs_training += 1


          return self.accuracy, self.loss, self.val_accuracy, self.val_loss


      def _train_epoch(self):

          with self.model.graph.as_default():
             # print("Epoch num "+str(self._epochs_training))

              
              # Shuffle categories
              self.input_train, self.output_train, self.seq_len_train= utils.shuffle(self.input_train, self.output_train,self.seq_len_train)
      
              # Going through categories aS mini batch
              for i in range(len(self.input_train)):

                  # # Shuffle inside the category
                  permute = np.random.permutation(self.input_train[i].shape[0])
                  inputs = self.input_train[i][permute,:]
                  labels = self.output_train[i][permute]
                  seq_len = self.seq_len_train[i][permute]
                      
          
                  # Train on this batch (category)

                  _, self.vocabulary_matrix, basic_loss, acc, loss, prediction= self.session.run(
                          (self.model.train_op, self.model.vocabulary_matrix,self.model.loss_basic, self.model.accuracy, self.model.loss, self.model.prediction), 
                          # train operation to do training and all nodes of graph i want to get out of model
                                      feed_dict={self.model.input: inputs ,
                                              self.model.label: labels,
                                              self.model.seq_len: seq_len,
                                              self.model.vocabulary_matrix_val: np.zeros((1,300))
                                      
                                              }
                          )  



              # Evaluate on the whole set to get the total loss after this epoch
              epoch_loss=0
              epoch_acc=0
            # self.session.run(self.model.zero_ops)
              for i in range(len(self.input_train)):

                  
                  inputs = self.input_train[i]
                  labels = self.output_train[i]
                  seq_len = self.seq_len_train[i]

                  

                  summary, acc, num_correct, loss, basic_loss, prediction = self.session.run(
                              (self.model.summary, self.model.accuracy, self.model.num_correct,self.model.loss, self.model.loss_basic, self.model.prediction), 
                              #  all nodes of graph i want to get out of model and accumulate op for loss and acc
                                          feed_dict={self.model.input: inputs ,
                                                  self.model.label: labels,
                                                  self.model.seq_len: seq_len,
                                                  self.model.vocabulary_matrix_val: np.zeros((1,300))
                                          
                                                  }
                              )  
                  if not (self.sum_path is None): 
                    self.train_summary_writer.add_summary( summary, self.train_count)  #loss for every batch
                    self.train_count+=1

                  epoch_loss+=basic_loss
                  epoch_acc+=num_correct



              epoch_loss/=len(self.input_train)
              epoch_acc/=self.num_total_train

              if not (self.sum_path is None): 
                summary_train=tf.Summary()
                summary_train.value.add(tag='epoch_loss', simple_value=epoch_loss)
                summary_train.value.add(tag='epoch_acc', simple_value=epoch_acc)
                #summary_epoch_train = tf.summary.merge([tf_loss_summary_epoch,tf_acc_summary_epoch])  
                self.epoch_train_summary_writer.add_summary( summary_train, self.epoch_train_count)
                self.epoch_train_count+=1
                self.epoch_train_summary_writer.flush()

              #print('Train loss: '+'{:.3f}'.format(epoch_loss)+'    Train accuracy: '+'{:.3f}'.format(epoch_acc))
              self.loss.append(epoch_loss)
              self.accuracy.append(epoch_acc)
                          

              # Save the model checkpoint after every epoch
              if not self.checkpoints_path is None:
          
                  checkpoint_name = 'checkpoint_'+str(self._epochs_training)

                  with self.model.graph.as_default():
                          self.saver.save(self.session, os.path.join(self.checkpoints_path, checkpoint_name))
                          print('saved '+checkpoint_name)


          


      def validate(self):
    
          with self.model.graph.as_default():
              total_loss=0
              total_acc=0
            #  self.session.run(self.model.zero_ops)
              
              #Go through the categories
              for i in range(len(self.input_val)):

                  inputs = self.input_val[i]
                  labels = self.output_val[i]
                  seq_len = self.seq_len_val[i]




                  summary, acc, num_correct, loss, basic_loss, prediction = self.session.run(
                              (self.model.summary, self.model.accuracy, self.model.num_correct, self.model.loss, self.model.loss_basic, self.model.prediction), 
                              # all nodes of graph i want to get out of model, no train op!
                                          feed_dict={self.model.input: inputs ,
                                                  self.model.label: labels,
                                                  self.model.seq_len: seq_len,
                                                  self.model.vocabulary_matrix_val: self.vocabulary_matrix_val
                                                
                                                  }
                                  ) 

                  
                  if not (self.sum_path is None): 
                    self.valid_summary_writer.add_summary( summary, self.val_count)
                    self.val_count+=1

                  total_loss+=basic_loss
                  total_acc+=num_correct
              

              total_loss/=len(self.input_val)
              total_acc/=self.num_total_val

              if not (self.sum_path is None): 
                  summary_val=tf.Summary()
                  summary_val.value.add(tag='epoch_loss', simple_value=total_loss)
                  summary_val.value.add(tag='epoch_acc', simple_value=total_acc)
                # summary_total_val = tf.summary.merge([tf_loss_summary_total,tf_acc_summary_total])  
                  self.epoch_valid_summary_writer.add_summary( summary_val, self.epoch_val_count)
                  self.epoch_val_count+=1
                  self.epoch_valid_summary_writer.flush()

              #print('Validation loss: '+'{:.3f}'.format(total_loss)+'    Validation accuracy: '+'{:.3f}'.format(total_acc))
      
              self.val_loss.append(total_loss)
              self.val_accuracy.append(total_acc)
                  

                  


      def predict(self,test_checkpoint=None):

       # print(self.checkpoints_path+"/"+"checkpoint_"+test_checkpoint)
        #Loading weights
        #if not test_checkpoint is None: #go to some training
         #   assert os.path.isdir(self.checkpoints_path+"/"+"checkpoint_"+test_checkpoint) , "There is no checkpoint_"+str(test_checkpoint) #we want to continue some training
         #   self.latest=self.checkpoints_path+"/"+"checkpoint_"+test_checkpoint
 
        array_pred=[] # array of predictions


        if not test_checkpoint is None:
          self.latest=os.path.join(self.checkpoints_path,test_checkpoint)
          print(self.latest)

          with self.model.graph.as_default():
              # assert self.load(self.checkpoints_path, self.saver), " Loading weights failed! " #dobije path do poslednjeg checkpointa
              self.saver.restore(self.session,self.latest) 
              print("Loading weights success")  


        # Vocabulary

        vocab_val_test=np.concatenate((self.vocabulary_matrix_val,self.vocabulary_matrix_test),axis=0)

        with self.model.graph.as_default():
            total_loss=0
            total_acc=0

            #Go through the categories
            for i in range(len(self.input_test)):

                inputs = self.input_test[i]
                labels = self.output_test[i]
                seq_len = self.seq_len_test[i]

            

                acc, num_correct, loss_basic, prediction = self.session.run(
                    (self.model.accuracy, self.model.num_correct,self.model.loss_basic, self.model.prediction), 
                    # all nodes of graph i want to get out of model
                                feed_dict={self.model.input: inputs ,
                                        self.model.label: labels,
                                        self.model.seq_len: seq_len,
                                        self.model.vocabulary_matrix_val:  vocab_val_test
                                          }
                        ) 


                total_loss+=loss_basic
                total_acc+=num_correct


                for i in range(len(prediction)):
                    array_pred.append(prediction[i])
                
            

        total_loss/=len(self.input_test)
        total_acc/=self.num_total_test
                
               
        return array_pred, total_loss, total_acc





