import os
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Activation,BatchNormalization,Concatenate,Dense,Embedding,Flatten,Input,Multiply,Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.models import Model,Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt

class CGAN(object):
    def __init__(self,run_folder,image_size,n_channel,z_dim,n_classes):
        self.run_folder=run_folder
        self.img_shape = (image_size,image_size,n_channel)
        self.z_dim=z_dim
        self.n_classes=n_classes
        self.d_losses = []
        self.g_losses = []
        self.epoch = 0
        
        self.compile()
        
    def build_generator(self,z_dim):
        g_input=Input(shape=(z_dim,))
        x=Dense(256*7*7,input_dim=z_dim)(g_input)
        x=Reshape((7,7,256))(x)

        x=Conv2DTranspose(128,kernel_size=3,strides=2,padding='same')(x)
        x=BatchNormalization()(x)
        x=LeakyReLU(alpha=0.01)(x)

        x=Conv2DTranspose(64,kernel_size=3,strides=1,padding='same')(x)
        x=BatchNormalization()(x)
        x=LeakyReLU(alpha=0.01)(x)

        x=Conv2DTranspose(1,kernel_size=3,strides=2,padding='same')(x)
        g_output=Activation('tanh')(x)

        return Model(g_input,g_output)
    def build_cgan_generator(self):
        z=Input(shape=(self.z_dim,))
        label=Input(shape=(1,),dtype='int32')

        label_embedding=Embedding(self.n_classes,self.z_dim,input_length=1)(label)
        label_embedding=Flatten()(label_embedding)
        joined_representation=Multiply()([z,label_embedding])
        g_model=self.build_generator(self.z_dim)
        conditioned_img=g_model(joined_representation)

        return Model([z,label],conditioned_img,name="g")

    def build_discriminator(self,img_shape):
        d_input=Input(shape=(img_shape[0],img_shape[1],img_shape[2]+1))
        x=Conv2D(64,kernel_size=3,strides=2,padding='same')(d_input)
        x=LeakyReLU(alpha=0.01)(x)

        x=Conv2D(64,kernel_size=3,strides=2,padding='same')(x)
        x=BatchNormalization()(x)
        x=LeakyReLU(alpha=0.01)(x)

        x=Conv2D(128,kernel_size=3,strides=2,padding='same')(x)
        x=BatchNormalization()(x)
        x=LeakyReLU(alpha=0.01)(x)

        x=Flatten()(x)
        d_output=Dense(1,activation='sigmoid')(x)

        return Model(d_input,d_output)
    def build_cgan_discriminator(self):
        img=Input(shape=self.img_shape)    
        label=Input(shape=(1,),dtype="int32")

        label_embedding=Embedding(self.n_classes,np.prod(self.img_shape),input_length=1)(label)
        label_embedding=Flatten()(label_embedding)
        label_embedding=Reshape(self.img_shape)(label_embedding)

        concatenated=Concatenate(axis=-1)([img,label_embedding])
        d_model=self.build_discriminator(self.img_shape)
        classification=d_model(concatenated)

        return Model([img,label],classification,name="d")

    def compile(self):
        self.d_model=self.build_cgan_discriminator()
        self.d_model.compile(loss='binary_crossentropy',
                             optimizer=Adam(),
                             metrics=['accuracy'])
        self.d_model.trainable= False
        
        self.g_model=self.build_cgan_generator()
        z=Input(shape=(self.z_dim,))
        label=Input(shape=(1,))
        img=self.g_model([z,label])
        classification=self.d_model([img,label])
        
        self.cgan_model=Model([z,label],classification,name="cgan")
        self.cgan_model.compile(loss='binary_crossentropy',optimizer=Adam())
        
        
    
    def train(self,epochs,batch_size,sample_interval):
        (X_train,y_train),(_,_)=mnist.load_data()
        X_train=X_train/127.5-1
        X_train=np.expand_dims(X_train,axis=3)
        real=np.ones((batch_size,1))
        fake=np.zeros((batch_size,1))
        
        for epoch in range(self.epoch,self.epoch + epochs):
            idx=np.random.randint(0,X_train.shape[0],batch_size)
            real_imgs,labels=X_train[idx],y_train[idx]
            z=np.random.normal(0,1,(batch_size,self.z_dim))
            fake_imgs=self.g_model.predict([z,labels])
            
            d_loss_real=self.d_model.train_on_batch([real_imgs,labels],real)
            d_loss_fake=self.d_model.train_on_batch([fake_imgs,labels],fake)
            d_loss=0.5*np.add(d_loss_real,d_loss_fake)
            
            z=np.random.normal(0,1,(batch_size,self.z_dim))
            labels=np.random.randint(0,self.n_classes,batch_size).reshape(-1,1)
            g_loss=self.cgan_model.train_on_batch([z,labels],real)
            
            if(epoch%sample_interval==0):
                print("%d [D loss: %f,acc:%.2f%%][G loss:%f]"%(epoch,d_loss[0],100*d_loss[1],g_loss))
                self.d_losses.append(d_loss[0])
                self.g_losses.append(g_loss)
                
                self.save_weights(os.path.join(self.run_folder, 'weights/weights.h5'))
                self.sample_images()
                
            self.epoch+=1
    
    def sample_images(self,img_row=2,img_col=5):
        print("====")
        z=np.random.normal(0,1,(img_row*img_col,self.z_dim))
        labels=np.arange(0,10).reshape(-1,1)  
        fake_imgs=self.g_model.predict([z,labels])
        fake_imgs=fake_imgs*0.5+0.5
        fig,axs=plt.subplots(img_row,img_col,figsize=(10,4),sharey=True,sharex=True)
        
        cnt=0
        for i in range(img_row):
            for j in range(img_col):
                axs[i,j].imshow(fake_imgs[cnt,:,:,0],cmap='gray')
                axs[i,j].axis('off')
                axs[i,j].set_title("Digit:%d"%labels[cnt])
                cnt+=1
        fig.savefig(os.path.join(self.run_folder, "images/sample_%d.png" % self.epoch))
        plt.close()
        
    def save_weights(self, filepath):
        self.cgan_model.save_weights(filepath)
        
    def load_weights(self, filepath):
        self.cgan_model.load_weights(filepath)
        
    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val        
