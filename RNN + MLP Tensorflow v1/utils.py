import os
import datetime
import matplotlib.pyplot as plt
import io
import numpy as np
import random



def get_output_dir(output_root='.'):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return os.path.join(output_root, 'outputs', timestamp)


def ensure_parent_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
def shuffle(input_train, output_train,seq_len_train):
    c = list (zip(list(zip(input_train, output_train)),seq_len_train))
    random.shuffle(c)
    ab, seq_len = zip(*c)
    input, output = zip(*ab)

    return input, output, seq_len

def plot_results(acc_arr,loss_arr,acc_val,loss_val,title1,title2):
    """
    epoch: Scalar quantity of number of batches over epoch
    acc_arr: Acc on each batch
    loss_arr: Loss on each epoch
    title: Train/Val/or Test
    """
  
    t = np.arange(len(acc_arr))
    t1= np.arange(len(acc_val))
    
    # Plot loss
    #plt.figure()
    fig, ax = plt.subplots(2,2,figsize=(20,20))
    ax[0,0].plot(t,loss_arr)
    ax[0,0].set_title(title1 +  ' loss')
    ax[0,0].set_xlabel('Epoch')
    ax[0,0].set_ylabel('Loss')
    ax[0,0].grid()
    
    # Plot acc
    ax[0,1].plot(t,acc_arr)
    ax[0,1].set_title(title1 + ' accuracy')
    ax[0,1].set_xlabel('Epoch')
    #ax[0,1].set_xticks(np.linspace(0,t1[-1],11))
    ax[0,1].set_ylabel('Accuracy')
    ax[0,1].grid()

    # Plot val loss
    ax[1,0].plot(t1,loss_val)
    ax[1,0].set_title(title2 +  ' loss')
    ax[1,0].set_xlabel('Epoch')
    ax[1,0].set_ylabel('Loss')
    ax[1,0].grid()
    
    # Plot  val acc
    ax[1,1].plot(t1,acc_val)
    ax[1,1].set_title(title2 + ' accuracy')
    ax[1,1].set_xlabel('Epoch')
    #ax[1,1].set_xticks(np.linspace(0,t1[-1],11))
    ax[1,1].set_ylabel('Accuracy')
    ax[1,1].grid()
   # plt.show()

def plot_results1(acc_arr,loss_arr,acc_val,loss_val,title1,title2,max_epoch):
    """
    epoch: Scalar quantity of number of batches over epoch
    acc_arr: Acc on each batch
    loss_arr: Loss on each epoch
    title: Train/Val/or Test
    """
  
    t = np.arange(len(acc_arr))
    t1= np.arange(len(acc_val))
    
    # Plot loss
    #plt.figure()
    fig, ax = plt.subplots(2,2,figsize=(20,20))
    ax[0,0].plot(t,loss_arr)
    ax[0,0].plot([max_epoch],[loss_arr[max_epoch]],'ro')
    ax[0,0].set_title(title1 +  ' loss')
    ax[0,0].set_xlabel('Epoch')
    ax[0,0].set_ylabel('Loss')
    ax[0,0].grid()
    
    # Plot acc
    ax[0,1].plot(t,acc_arr)
    ax[0,1].plot([max_epoch],[acc_arr[max_epoch]],'ro')
    ax[0,1].set_title(title1 + ' accuracy')
    ax[0,1].set_xlabel('Epoch')
    #ax[0,1].set_xticks(np.linspace(0,t1[-1],11))
    ax[0,1].set_ylabel('Accuracy')
    ax[0,1].grid()

    # Plot val loss
    ax[1,0].plot(t1,loss_val)
    ax[1,0].plot([max_epoch],[loss_val[max_epoch]],'ro')
    ax[1,0].set_title(title2 +  ' loss')
    ax[1,0].set_xlabel('Epoch')
    ax[1,0].set_ylabel('Loss')
    ax[1,0].grid()
    
    # Plot  val acc
    ax[1,1].plot(t1,acc_val)
    ax[1,1].plot([max_epoch],[acc_val[max_epoch]],'ro')
    ax[1,1].set_title(title2 + ' accuracy')
    ax[1,1].set_xlabel('Epoch')
    #ax[1,1].set_xticks(np.linspace(0,t1[-1],11))
    ax[1,1].set_ylabel('Accuracy')
    ax[1,1].grid()
   # plt.show()
  
  