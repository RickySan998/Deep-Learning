# -*- coding: utf-8 -*-
"""Question5

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IRYKYVjXQoBuTQyZjnYpGQkJ0GjzvEhT
"""



"""# Instalations"""

from google.colab import drive
drive.mount('/content/gdrive/')

!ls

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

from google.colab import files
from model import Model
from trainer import Trainer
import utils

import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import zipfile
import os
import pickle

# Commented out IPython magic to ensure Python compatibility.
# Load the TensorBoard notebook extension
# %load_ext tensorboard

"""# Load data"""

# LOad data
path='gdrive/MyDrive/DL_cw2/data/'

with open (path+'input_train_cat', 'rb') as fp:
    emb_train_padded = pickle.load(fp)
with open (path+'input_val_cat', 'rb') as fp:
    emb_val_padded = pickle.load(fp)
with open (path+'input_test_cat', 'rb') as fp:
    emb_test_padded = pickle.load(fp)

with open (path+'output_train_cat', 'rb') as fp:
    labels_train = pickle.load(fp)
with open (path+'output_val_cat', 'rb') as fp:
    labels_val = pickle.load(fp)     
with open (path+'output_test_cat', 'rb') as fp:
    labels_test = pickle.load(fp)

with open (path+'seq_len_train', 'rb') as fp:
    seq_len_train = pickle.load(fp)
with open (path+'seq_len_val', 'rb') as fp:
    seq_len_val = pickle.load(fp)
with open (path+'seq_len_test', 'rb') as fp:
    seq_len_test = pickle.load(fp)
with open (path+'vocabulary_matrix', 'rb') as fp:
    vocabulary_matrix = pickle.load(fp)    
with open (path+'vocabulary_matrix_val', 'rb') as fp:
    vocabulary_matrix_val = pickle.load(fp)
with open (path+'vocabulary_matrix_test', 'rb') as fp:
    vocabulary_matrix_test = pickle.load(fp)


#New sentences!
with open (path+'input_test_cat_NEW', 'rb') as fp:
    emb_test_padded_new = pickle.load(fp)
with open (path+'seq_len_test_NEW', 'rb') as fp:
    seq_len_test_new = pickle.load(fp)
with open (path+'vocabulary_matrix_test_NEW', 'rb') as fp:
    vocabulary_matrix_test_new = pickle.load(fp)

input_test_new=emb_test_padded_new[1:2]
output_test_new=[np.asarray([0,0,0,0,0])]
seq_test_new=seq_len_test_new[1:2]
print(seq_test_new[0].shape)
print(output_test_new[0].shape)





end=10
input_train=emb_train_padded[0:end]
output_train=labels_train[0:end]
seq_train=seq_len_train[0:end]


end1=10
input_val=emb_val_padded[0:end1]
output_val=labels_val[0:end1]
seq_val=seq_len_val[0:end1]


end2=10
input_test=emb_test_padded[0:end2]
output_test=labels_test[0:end2]
seq_test=seq_len_test[0:end2]

"""# Tensorboard"""

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')

# Commented out IPython magic to ensure Python compatibility.
# Open tensorboard
# %tensorboard --logdir ./outputs

"""# Functions:"""

def combination(train_emb_bool, cell_type, regular_coeff_val, epochs, 
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,emb_test_padded,labels_test,seq_len_test):



    num_classes=5
    input_size=300

    # Fill with data
    end=10
    input_train=emb_train_padded[0:end]
    output_train=labels_train[0:end]
    seq_train=seq_len_train[0:end]


    end1=10
    input_val=emb_val_padded[0:end1]
    output_val=labels_val[0:end1]
    seq_val=seq_len_val[0:end1]


    end2=10
    input_test=emb_test_padded[0:end2]
    output_test=labels_test[0:end2]
    seq_test=seq_len_test[0:end2]




    # Hyperparameters!
    #train_emb_bool=False
    #cell_type='Basic'
    keep_dropout=0.9

    optimiser='Adam' 

    #regular_coeff_val=[0.0]

   # epochs=100

    max_acc_all=-1
    for regular_coeff in regular_coeff_val:
      for num_RNN_layers in [1,2]:
        for sentence_embedding_size in [600, 900]: #Same as qn 4 or less
          for node_per_MLP_layer in [[32,16],[50,25]]: #same as qn 4 or less
              for learning_rate in [0.1, 0.001]:

                # Create model
                model = Model(keep_dropout, train_emb_bool,vocabulary_matrix, node_per_MLP_layer, num_classes, input_size, sentence_embedding_size, cell_type, num_RNN_layers,learning_rate,regular_coeff,optimiser)

                # Lets train
                trainer = Trainer(vocabulary_matrix_val, vocabulary_matrix_test, input_train, output_train, seq_train, input_val, output_val, seq_val, input_test, output_test, seq_test, 
                                model, epochs)

                train_acc, train_loss, val_acc,val_loss = trainer.train()

                # Ploting the results
                print('Parameters:')
                print('RNN_layers: '+str(num_RNN_layers))
                print('Sentence emb size: '+str(sentence_embedding_size))
                print('Node per MLP layer: '+str(node_per_MLP_layer))
                print('Learning rate: '+str(learning_rate))
                print('Regularization rate: '+str(regular_coeff))
                print('Max validation accuracy is '+ '{:.4f}'.format(np.max(val_acc))+' on epoch: '+str(np.argmax(val_acc)+1))
                print('Training accuracy is '+ '{:.4f}'.format(np.max(train_acc)))
                utils.plot_results(train_acc, train_loss, val_acc, val_loss, 'Training','Validation')

                max_acc=np.max(val_acc)
                if max_acc_all<max_acc:
                  max_acc_all=max_acc
                  best_num_RNN=num_RNN_layers
                  best_emb_size=sentence_embedding_size
                  best_node_MLP=node_per_MLP_layer
                  best_learn=learning_rate
                  best_reg=regular_coeff
                  best_epoch=np.argmax(val_acc)

    print('BEST PARAMETERS:')
    print('RNN_layers: '+str(best_num_RNN))
    print('Sentence emb size: '+str(best_emb_size))
    print('Node per MLP layer: '+str(best_node_MLP))
    print('Learning rate: '+str(best_learn))
    print('Regularization rate: '+str(best_reg))
    print('Max validation accuracy is '+ '{:.4f}'.format(max_acc_all)+' for epoch: '+str(best_epoch))

            




              
    return best_num_RNN,best_emb_size,best_node_MLP,best_learn,best_reg,max_acc_all,best_epoch

# Retrain again for the best parameters

def retrain( num_plot, train_emb_bool, cell_type, 
            best_num_RNN1,best_emb_size1,best_node_MLP1,best_learn1,best_epoch1,best_reg1,
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test):

    num_classes=5
    input_size=300

    # Fill with data
    end=10
    input_train=emb_train_padded[0:end]
    output_train=labels_train[0:end]
    seq_train=seq_len_train[0:end]


    end1=10
    input_val=emb_val_padded[0:end1]
    output_val=labels_val[0:end1]
    seq_val=seq_len_val[0:end1]


    end2=10
    input_test=emb_test_padded[0:end2]
    output_test=labels_test[0:end2]
    seq_test=seq_len_test[0:end2]




    # Hyperparameters!
    keep_dropout=0.9
    optimiser='Adam' 

    epochs=best_epoch1+20

    num_RNN_layers=best_num_RNN1
    sentence_embedding_size=best_emb_size1
    node_per_MLP_layer=best_node_MLP1
    learning_rate = best_learn1
    regular_coeff=best_reg1

            




    # Create model
    model = Model(keep_dropout, train_emb_bool,vocabulary_matrix, node_per_MLP_layer, num_classes, input_size, sentence_embedding_size, cell_type, num_RNN_layers,learning_rate,regular_coeff,optimiser)

    # Lets train
    trainer = Trainer(vocabulary_matrix_val, vocabulary_matrix_test, input_train, output_train, seq_train, input_val, output_val, seq_val, input_test, output_test, seq_test, 
                    model, epochs)

    train_acc, train_loss, val_acc,val_loss = trainer.train()

    # Ploting the results
    print('Parameters:')
    print('RNN_layers: '+str(num_RNN_layers))
    print('Sentence emb size: '+str(sentence_embedding_size))
    print('Node per MLP layer: '+str(node_per_MLP_layer))
    print('Learning rate: '+str(learning_rate))
    print('Regularization rate: '+str(regular_coeff))
    print('Max validation accuracy is '+ '{:.4f}'.format(np.max(val_acc))+' on epoch: '+str(np.argmax(val_acc)+1))
    print('Training accuracy is '+ '{:.4f}'.format(train_acc[np.argmax(val_acc)]))
    utils.plot_results(train_acc, train_loss, val_acc, val_loss, 'Training','Validation')
    plt.savefig('plot'+str(num_plot)+'.png')

    return np.max(val_acc), np.argmax(val_acc), train_acc[np.argmax(val_acc)]

"""# Combinations

## with fixed embeddings, without regularization

with fixed embeddings, with Basic cell, without regularization
"""

# Search combinations
train_emb_bool=False
cell_type='Basic' 
regular_coeff_val=[0.0]
epochs=100

best_num_RNN,best_emb_size,best_node_MLP,best_learn,best_reg,max_acc_all,best_epoch =combination(
   train_emb_bool, cell_type, regular_coeff_val, epochs, 
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

# Retrain for best parameters
train_emb_bool=False
cell_type='Basic' 
num_plot=1

max_val_acc,max_epochs,max_train_acc=retrain( num_plot,
    train_emb_bool, cell_type,
    best_num_RNN,best_emb_size,best_node_MLP,best_learn,best_epoch,best_reg,
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

"""with fixed embeddings, with GRU cell, without regularization"""

# Search combinations
train_emb_bool=False
cell_type='GRU' 
regular_coeff_val=[0.0]
epochs=100

best_num_RNN1,best_emb_size1,best_node_MLP1,best_learn1,best_reg1,max_acc_all1,best_epoch1 =combination(
   train_emb_bool, cell_type, regular_coeff_val, epochs, 
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

# Retrain for best parameters
train_emb_bool=False
cell_type='GRU' 
num_plot=2

max_val_acc,max_epochs,max_train_acc=retrain( num_plot,
    train_emb_bool, cell_type,
    best_num_RNN1,best_emb_size1,best_node_MLP1,best_learn1,best_epoch1,best_reg1,
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

"""with fixed embeddings, with LSTM cell, without regularization"""

# Search combinations
train_emb_bool=False
cell_type='LSTM' 
regular_coeff_val=[0.0]
epochs=100

best_num_RNN2,best_emb_size2,best_node_MLP2,best_learn2,best_reg2,max_acc_all2,best_epoch2, =combination(
   train_emb_bool, cell_type, regular_coeff_val, epochs, 
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

# Retrain for best parameters
train_emb_bool=False
cell_type='LSTM' 
num_plot=3

max_val_acc,max_epochs,max_train_acc=retrain( num_plot,
    train_emb_bool, cell_type,
    best_num_RNN2,best_emb_size2,best_node_MLP2,best_learn2,best_epoch2,best_reg2,
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

"""## with fixed embeddings, with regularization

with fixed embeddings, with Vanilla cell, **with regularization**
"""

# Search combinations
train_emb_bool=False
cell_type='Basic' 
regular_coeff_val=[0.01,0.1]
epochs=100

best_num_RNN3,best_emb_size3,best_node_MLP3,best_learn3,best_reg3,max_acc_all3,best_epoch3 =combination(
   train_emb_bool, cell_type, regular_coeff_val, epochs, 
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

# Retrain for best parameters
train_emb_bool=False
cell_type='Basic' 
num_plot=4

max_val_acc,max_epochs,max_train_acc=retrain( num_plot,
    train_emb_bool, cell_type,
    best_num_RNN3,best_emb_size3,best_node_MLP3,best_learn3,best_epoch3,best_reg3,
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

"""with fixed embeddings, with GRU cell, with regularization"""

# Search combinations
train_emb_bool=False
cell_type='GRU' 
regular_coeff_val=[0.01,0.1]
epochs=100

best_num_RNN4,best_emb_size4,best_node_MLP4,best_learn4,best_reg4,max_acc_all4,best_epoch4 =combination(
   train_emb_bool, cell_type, regular_coeff_val, epochs, 
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

# Retrain for best parameters
train_emb_bool=False
cell_type='GRU' 
num_plot=5

max_val_acc,max_epochs,max_train_acc=retrain( num_plot,
    train_emb_bool, cell_type,
    best_num_RNN4,best_emb_size4,best_node_MLP4,best_learn4,best_epoch4,best_reg4,
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

"""with fixed embeddings, with LSTM cell, with regularization"""

# Search combinations
train_emb_bool=False
cell_type='LSTM' 
regular_coeff_val=[0.01,0.1]
epochs=100

best_num_RNN5,best_emb_size5,best_node_MLP5,best_learn5,best_reg5,max_acc_all5,best_epoch5 =combination(
   train_emb_bool, cell_type, regular_coeff_val, epochs, 
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

# Retrain for best parameters
train_emb_bool=False
cell_type='LSTM' 
num_plot=6

max_val_acc,max_epochs,max_train_acc=retrain( num_plot,
    train_emb_bool, cell_type,
    best_num_RNN5,best_emb_size5,best_node_MLP5,best_learn5,best_epoch5,best_reg5,
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

"""## with trained embeddings, with regularization

**with trained embeddings**, with Vanilla cell, with regularization
"""

# Search combinations
train_emb_bool=True
cell_type='Basic' 
regular_coeff_val=[0.01,0.1]
epochs=100

best_num_RNN6,best_emb_size6,best_node_MLP6,best_learn6,best_reg6,max_acc_all6,best_epoch6 =combination(
   train_emb_bool, cell_type, regular_coeff_val, epochs, 
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

# Retrain for best parameters
train_emb_bool=True
cell_type='Basic' 
num_plot=7

max_val_acc,max_epochs,max_train_acc=retrain( num_plot,
    train_emb_bool, cell_type,
    best_num_RNN6,best_emb_size6,best_node_MLP6,best_learn6,best_epoch6,best_reg6,
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

"""with trained embeddings, with GRU cell, with regularization"""

# Search combinations
train_emb_bool=True
cell_type='GRU' 
regular_coeff_val=[0.01,0.1]
epochs=100

best_num_RNN7,best_emb_size7,best_node_MLP7,best_learn7,best_reg7,max_acc_all7,best_epoch7 =combination(
   train_emb_bool, cell_type, regular_coeff_val, epochs, 
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

# Retrain for best parameters
train_emb_bool=True
cell_type='GRU' 
num_plot=8

max_val_acc,max_epochs,max_train_acc=retrain( num_plot,
    train_emb_bool, cell_type,
    best_num_RNN7,best_emb_size7,best_node_MLP7,best_learn7,best_epoch7,best_reg7,
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

"""with trained embeddings, with LSTM cell, with regularization"""

# Search combinations
train_emb_bool=True
cell_type='LSTM' 
regular_coeff_val=[0.01,0.1]
epochs=100

best_num_RNN8,best_emb_size8,best_node_MLP8,best_learn8,best_reg8,max_acc_all8,best_epoch8 =combination(
   train_emb_bool, cell_type, regular_coeff_val, epochs, 
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

# Retrain for best parameters
train_emb_bool=True
cell_type='LSTM' 
num_plot=9

max_val_acc,max_epochs,max_train_acc=retrain( num_plot,
    train_emb_bool, cell_type,
    best_num_RNN8,best_emb_size8,best_node_MLP8,best_learn8,best_epoch8,best_reg8,
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)

# Retrain for best parameters
train_emb_bool=True
cell_type='LSTM' 
num_plot=10

max_val_acc,max_epochs,max_train_acc=retrain( num_plot,
    train_emb_bool, cell_type,
    best_num_RNN8,best_emb_size8,best_node_MLP8,best_learn8,best_epoch8,0.05,
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test)



"""# Changing dropout and optimiser"""

# Retrain function with choosing all hyperparameters and saving checkpoints

def retrain1( train_emb_bool, cell_type, optimiser, keep_dropout,
            best_num_RNN1,best_emb_size1,best_node_MLP1,best_learn1,best_epoch1,best_reg1,
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test,
    vocabulary_matrix, vocabulary_matrix_val, vocabulary_matrix_test):

    num_classes=5
    input_size=300

    # Fill with data
    end=10
    input_train=emb_train_padded[0:end]
    output_train=labels_train[0:end]
    seq_train=seq_len_train[0:end]


    end1=10
    input_val=emb_val_padded[0:end1]
    output_val=labels_val[0:end1]
    seq_val=seq_len_val[0:end1]


    end2=10
    input_test=emb_test_padded[0:end2]
    output_test=labels_test[0:end2]
    seq_test=seq_len_test[0:end2]


    epochs=best_epoch1+20

    num_RNN_layers=best_num_RNN1
    sentence_embedding_size=best_emb_size1
    node_per_MLP_layer=best_node_MLP1
    learning_rate = best_learn1
    regular_coeff=best_reg1

            




    # Create model
    model = Model(keep_dropout, train_emb_bool,vocabulary_matrix, node_per_MLP_layer, num_classes, input_size, sentence_embedding_size, cell_type, num_RNN_layers,learning_rate,regular_coeff,optimiser)

    # Lets train
    trainer = Trainer(vocabulary_matrix_val, vocabulary_matrix_test, input_train, output_train, seq_train, input_val, output_val, seq_val, input_test, output_test, seq_test, 
                    model, epochs, sum_path=None, checkpoints_path='./training')

    train_acc, train_loss, val_acc,val_loss = trainer.train()

    # Ploting the results
    print('Parameters:')
    print('RNN_layers: '+str(num_RNN_layers))
    print('Sentence emb size: '+str(sentence_embedding_size))
    print('Node per MLP layer: '+str(node_per_MLP_layer))
    print('Learning rate: '+str(learning_rate))
    print('Regularization rate: '+str(regular_coeff))
    print('Max validation accuracy is '+ '{:.4f}'.format(np.max(val_acc))+' on epoch: '+str(np.argmax(val_acc)+1))
    print('Training accuracy is '+ '{:.4f}'.format(train_acc[np.argmax(val_acc)]))
    utils.plot_results(train_acc, train_loss, val_acc, val_loss, 'Training','Validation')
  





    return np.max(val_acc), np.argmax(val_acc), train_acc[np.argmax(val_acc)], train_acc, train_loss, val_acc, val_loss, model, trainer

# Predicting from the chosen checkpoint
def prediction(  checkpoint_num,
    train_emb_bool, cell_type, optimiser, keep_dropout,
            best_num_RNN1,best_emb_size1,best_node_MLP1,best_learn1,best_epoch1,best_reg1,
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test,vocabulary_matrix_test):

    num_classes=5
    input_size=300

    # Fill with data
    end=10
    input_train=emb_train_padded[0:end]
    output_train=labels_train[0:end]
    seq_train=seq_len_train[0:end]


    end1=10
    input_val=emb_val_padded[0:end1]
    output_val=labels_val[0:end1]
    seq_val=seq_len_val[0:end1]


    #Parameters
    epochs=best_epoch1+20

    num_RNN_layers=best_num_RNN1
    sentence_embedding_size=best_emb_size1
    node_per_MLP_layer=best_node_MLP1
    learning_rate = best_learn1
    regular_coeff=best_reg1



    # Create model
    model = Model(keep_dropout, train_emb_bool,vocabulary_matrix, node_per_MLP_layer, num_classes, input_size, sentence_embedding_size, cell_type, num_RNN_layers,learning_rate,regular_coeff,optimiser)

    # Create trainer
    trainer = Trainer(vocabulary_matrix_val, vocabulary_matrix_test, input_train, output_train, seq_train, input_val, output_val, seq_val, input_test, output_test, seq_test, 
                    model, 0, sum_path=None, checkpoints_path='./training')

    # Predict using chosen checkpoint
    predictions, test_loss, test_acc = trainer.predict(checkpoint_num)  



    return predictions, test_loss, test_acc

"""**Fine-tuning the last hyperparameters**"""

# Retrain for 
train_emb_bool=True
cell_type='LSTM' 
optimiser='Adam'
keep_dropout=0.7

max_val_acc,max_epochs,max_train_acc, train_acc, train_loss, val_acc,val_loss, model, trainer =retrain1( 
    train_emb_bool, cell_type, optimiser, keep_dropout,
    2,600,[50,25],0.001,30, 0.01,
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test,
    vocabulary_matrix, vocabulary_matrix_val, vocabulary_matrix_test)

# Tick the best epoch
utils.plot_results1(train_acc, train_loss, val_acc,val_loss,'Training', 'Validation', max_epochs)

# Predict for the test set

from trainer import Trainer

#Hyperparameters
train_emb_bool=True
cell_type='LSTM' 
optimiser='Adam'
keep_dropout=0.7

checkpoint_num="checkpoint_24"


predictions, test_loss, test_acc = prediction( checkpoint_num,
    train_emb_bool, cell_type, optimiser, keep_dropout,
    2,600,[50,25],0.001,30, 0.01,
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test,output_test,seq_test,vocabulary_matrix_test)

print('Test accuracy: '+'{:.4f}'.format(test_acc))
print('Test loss: '+'{:.4f}'.format(test_loss))


# Predicting with +-1 margin

output_labels=[]
for i in range(len(output_test)):
  for label in output_test[i]:
        output_labels.append(label)

num_correct=0
for i in range(len(predictions)):
  if predictions[i]==output_labels[i]+1 or predictions[i]==output_labels[i]-1 or predictions[i]==output_labels[i]:
    num_correct+=1

print('Test accuracy with +-1 margin: '+'{:.4f}'.format(float(num_correct)/float(len(predictions))))

# Predict for new sentences

from trainer import Trainer

#Hyperparameters
train_emb_bool=True
cell_type='LSTM' 
optimiser='Adam'
keep_dropout=0.7

checkpoint_num="checkpoint_24"


predictions, _, _ = prediction( checkpoint_num,
    train_emb_bool, cell_type, optimiser, keep_dropout,
    2,600,[50,25],0.001,30, 0.01,
    emb_train_padded,labels_train,seq_len_train,emb_val_padded,labels_val,seq_len_val,input_test_new,output_test_new,seq_test_new,vocabulary_matrix_test_new)

print(predictions)