#!/usr/bin/env python
# coding: utf-8

# In[2]:


# libraries for removing tensorflow verbose 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# importing libraries required
import glob
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import pickle
import warnings
warnings.filterwarnings('ignore')

# function to create dataset to work on
def dataset_creation():
    # forming empty dataframe and initializing path
    df_test = pd.DataFrame(columns=['comments','pos/neg'])
    for f in os.listdir('./data/aclImdb/test'):
       # appending values to dataframe depending upon if they are positive or negative reviews
        if f == 'neg' or f =='pos':
            folder_reviews = os.listdir('./data/aclImdb/test/'+f)
            for review in folder_reviews:
                data = open('./data/aclImdb/test/'+f+'/'+review, encoding="utf8")                 
                if f == 'neg':
                    df_test = df_test.append({'comments': data.read(), 'pos/neg': 0}, ignore_index=True)
                else:
                    df_test = df_test.append({'comments': data.read(), 'pos/neg': 1}, ignore_index=True)
    return df_test['comments'],df_test['pos/neg']

if __name__ == "__main__": 
    
    # calling the dataset creation function
    x_test, y_test = dataset_creation()
    x_test=x_test.to_list()
    y_test = np.asarray(y_test).astype('int64')


    # Loading the model saved 
    model_2 = keras.models.load_model("./models/Group_24_NLP_model.h5")

    # performing pre-processing and preparing the saved data
    create_tokens = pickle.load(open("./data/tokenizer.pkl", 'rb'))
    # tokenization for test data
    sentence_test = create_tokens.texts_to_sequences(x_test) 
    # padding the data
    padding_test = pad_sequences(sentence_test, maxlen=400) 

    # Predicting the values and calculating accuracy of the predicted values
    predict_result = model_2.predict(padding_test)
    testing_loss, testing_accuracy = model_2.evaluate(padding_test,  y_test)
    print('Test accuracy is :', testing_accuracy)


# In[ ]:




