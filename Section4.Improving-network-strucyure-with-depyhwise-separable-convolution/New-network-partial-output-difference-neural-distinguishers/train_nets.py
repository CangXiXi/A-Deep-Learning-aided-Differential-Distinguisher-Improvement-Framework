from pickle import dump

import numpy as np
import speck as sp
from tensorflow.keras.backend import concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Dense, concatenate, Conv1D, Input, Reshape, Permute, Add, Flatten, \
    BatchNormalization, Activation,SeparableConv1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
bs = 10000;

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);

def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
  return(res);



#make residual tower of convolutional blocks
def make_resnet(num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3,depth=5, reg_param=0.0001,num=1,final_activation='sigmoid'):
  #Input and preprocessing layers
  inp = Input(shape=(num*2,));
  rs = Reshape(( 2, num))(inp);
  perm = Permute((2,1))(rs);
  #add a single residual layer that will expand the data to num_filters channels
  #this is a bit-sliced layer
  conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm);
  conv0 = BatchNormalization()(conv0);
  conv0 = Activation('relu')(conv0);
  #add residual blocks
  shortcut = conv0;
  for i in range(depth):
    conv1 = SeparableConv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut);
    conv1 = BatchNormalization()(conv1);
    conv1 = Activation('relu')(conv1);
    conv2 = SeparableConv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1);
    conv2 = BatchNormalization()(conv2);
    conv2 = Activation('relu')(conv2);
    shortcut = Add()([shortcut, conv2]);
  #add prediction head
  flat1 = Flatten()(shortcut);
  dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1);
  dense1 = BatchNormalization()(dense1);
  dense1 = Activation('relu')(dense1);
  dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
  dense2 = BatchNormalization()(dense2);
  dense2 = Activation('relu')(dense2);
  out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2);
  model = Model(inputs=inp, outputs=out);
  return(model);


def train_speck_distinguisher(num_epochs, num_rounds, depth,bit,bit_num,wdir):
    bit = np.concatenate((bit, [i + 16 for i in bit]))
    #create the network
    net = make_resnet(depth=depth, reg_param=10**-5,num=bit_num);
    net.compile(optimizer='adam',loss='mse',metrics=['acc']);
    #generate training and validation data
    X, Y = sp.make_train_data(10**7,bit=bit, nr=num_rounds,diff = (0x0040, 0))\
        ;
    print(np.shape(X))
    X_eval, Y_eval = sp.make_train_data(10**6, bit=bit, nr=num_rounds,diff = (0x0040, 0));
    #set up model checkpoint
    check = make_checkpoint(wdir+'best'+str(num_rounds)+'depth'+str(depth)+'.h5');
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));
    #train and evaluate
    h = net.fit(X,Y,epochs=num_epochs,batch_size=bs,validation_data=(X_eval, Y_eval), callbacks=[lr,check]);
    np.save(wdir+'bit'+str(bit)+"r"+str(num_rounds)+'val_acc'+'.npy', h.history['val_acc']);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'val_loss'+'.npy', h.history['val_loss']);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'loss'+'.npy', h.history['loss']);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'acc'+'.npy', h.history['acc']);
    dump(h.history,open(wdir+'bit'+str(bit)+'.p','wb'));
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    return(np.max(h.history['val_acc']));

if __name__ =="__main__":
    train_speck_distinguisher(num_epochs=30, num_rounds=6, depth=5, bit=[12, 13], bit_num=2, wdir='./[12,13]/')
    #0.6674500107765198
    '''
    train_speck_distinguisher(num_epochs=30, num_rounds=6, depth=5, bit=[12, 13, 14], bit_num = 3,wdir='./[12,13,14]/')
    #0.6726250052452087
    #23,969
    train_speck_distinguisher(num_epochs=30, num_rounds=6, depth=5,bit=[5,12,13,14], bit_num = 4,wdir='./[5,12,13,14]/')
    #0.6720860004425049
    #26,017
    train_speck_distinguisher(num_epochs=30, num_rounds=6, depth=5, bit=[5,12, 13, 14,15], bit_num = 5, wdir='./[5,12,13,14,15]/')
    #0.6879990100860596
    #28,065
    '''
