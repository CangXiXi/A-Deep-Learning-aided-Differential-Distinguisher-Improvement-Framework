import numpy as np
import speck as sp
from tensorflow.keras.backend import concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Dense, concatenate, Conv1D, Input, Reshape, Permute, Add, Flatten, \
    BatchNormalization, Activation,SeparableConv1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
bs = 10000;


wdir = './freshly_trained_nets/'

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);

def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
  return(res);

#make residual tower of convolutional blocks
def make_resnet(num_blocks=2, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3,depth=5, reg_param=0.0001, final_activation='sigmoid'):
  inp = Input(shape=(num_blocks * word_size * 2,));
  rs = Reshape((2 * num_blocks, word_size))(inp);
  perm = Permute((2,1))(rs);
  conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm);
  conv0 = BatchNormalization()(conv0);
  conv0 = Activation('relu')(conv0);
  shortcut = conv0;

  for i in range(depth):
    conv1 = SeparableConv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut);
    conv1 = BatchNormalization()(conv1);
    conv1 = Activation('relu')(conv1);
    conv2 = SeparableConv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1);
    conv2 = BatchNormalization()(conv2);
    conv2 = Activation('relu')(conv2);
    shortcut = Add()([shortcut, conv2]);
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

def train_speck_distinguisher(num_epochs, num_rounds=7, depth=1):
    net = make_resnet(depth=depth, reg_param=10**-5);
    net.compile(optimizer='adam',loss='mse',metrics=['acc']);
    X, Y = sp.make_train_data(10**7,num_rounds);
    X_eval, Y_eval = sp.make_train_data(10**6, num_rounds);
    check = make_checkpoint(wdir+'best'+str(num_rounds)+'depth'+str(depth)+'.h5');
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));
    h = net.fit(X,Y,epochs=num_epochs,batch_size=bs,validation_data=(X_eval, Y_eval),
                callbacks=[lr,check]);

    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_acc']);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_loss']);
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    return(net, h);
