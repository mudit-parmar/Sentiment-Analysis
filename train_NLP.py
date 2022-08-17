#!/usr/bin/env python
# coding: utf-8

# In[2]:


# libraries for removing tensorflow verbose 
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.FATAL)
# importing libraries required
import glob
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, Dense, Dropout, GlobalAveragePooling1D, LSTM
from keras import Sequential
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import pickle

# function to create dataset to work on
def dataset_creation():
    # forming empty dataframe and initializing path
    df_train = pd.DataFrame(columns=['comments','pos/neg'])
    for f in os.listdir('./data/aclImdb/train'):
        # appending values to dataframe depending upon if they are positive or negative reviews
        if f == 'neg' or f =='pos':
            folder_reviews = os.listdir('./data/aclImdb/train/'+f)
            for review in folder_reviews:
                data = open('./data/aclImdb/train/'+f+'/'+review, encoding="utf8")                 
                if f == 'neg':
                    df_train = df_train.append({'comments': data.read(), 'pos/neg': 0}, ignore_index=True)
                else:
                    df_train = df_train.append({'comments': data.read(), 'pos/neg': 1}, ignore_index=True)
    return df_train['comments'],df_train['pos/neg']

if __name__ == "__main__": 
    # calling the dataset creation function
    x_train, y_train = dataset_creation()
    # converting dataframe columns to list and numpy arrays for labels
    x_train=x_train.to_list()
    y_train = np.asarray(y_train).astype('int64')
    
    # performing tokenization through tokenizer
    create_tokens= Tokenizer(num_words=18000, oov_token="<OOV>") 
    create_tokens.fit_on_texts(x_train)  
    # initializing word index
    w_ind = create_tokens.word_index   
    # converion to word index
    sentences = create_tokens.texts_to_sequences(x_train)   
    # performing padding 
    xtrain_padded = pad_sequences(sentences,maxlen=450,truncating='post') 
    # saving tookenized file to be used later for testing
    pickle.dump(create_tokens, open("./data/tokenizer.pkl", 'wb')) 
    # splitting data
    X_train, X_val, y_train, y_val = train_test_split(xtrain_padded, y_train, test_size=0.3, random_state=42)

    '''
    model_1 = Sequential()
    model_1.add(Embedding(18000, 16)) 
    model_1.add(Dropout(0.2))
    model_1.add(LSTM(100))
    model_1.add(Dense(units=256, activation='relu'))
    model_1.add(Dropout(0.2))
    model_1.add(Dense(units=1, activation='sigmoid'))
    model_1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    '''
    # initializing model for nlp
    model_2 = Sequential()
    model_2.add(Embedding(18000, 16)) 
    model_2.add(Dropout(0.1))
    model_2.add(Conv1D(filters=16, kernel_size=2, activation='relu'))
    model_2.add(GlobalAveragePooling1D())
    model_2.add(Dropout(0.1))
    model_2.add(Dense(32, activation='relu'))
    model_2.add(Dropout(0.1))
    model_2.add(Dense(1, activation='sigmoid'))
    
    model_2.summary()
    
    # compiling model
    model_2.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

    history_2 = model_2.fit(X_train, y_train, epochs=3, validation_data=(X_val, y_val))
    # printing training and validation accuracy
    print('Training accuracy :', history_2.history['accuracy'][-1])
    print('Validation Accuracy :', history_2.history['val_accuracy'][-1])

    # saving the model
    model_2.save("./models/Group_24_NLP_model.h5")


# In[ ]:




