# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:36:21 2019

@author: KXue
"""

import pandas as pd
import numpy as np
import os, sys

from nltk.tokenize import RegexpTokenizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS 

def make_token(x):
    tokenizer = RegexpTokenizer("\w+\'?\w+|\w+")
    return tokenizer.tokenize(str(x))

def remove_stopwords(x, stop_words):
    return [token for token in x if token not in stop_words]

def lemmatization(x, nlp):
    lemma_result = []
    
    for words in x:
        doc = nlp(words)
        for token in doc:
            lemma_result.append(token.lemma_)
    
    return lemma_result

def pipeline(x, stop_words, nlp):
    x = make_token(x)
    x = remove_stopwords(x, stop_words)
    return ' '.join(lemmatization(x, nlp))



file = os.path.join('I:/', "CAA_TD_NLP_IA", "Data from TD")
data = pd.read_csv(os.path.join(file, "Step2Pretest2010To2015_All.csv")) ## replace df_wide.csv by the correct data

data = data.dropna(subset = ['Dur_Mean', 'Dur_SD']) # discard the observation that Dur_Mean or Dur_SD is NA
data_new = data.copy()
text_columns = ['ItemText', 'KeyText', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 
                'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'D13','D14', 'D15', 
                'D16', 'D17', 'D18', 'D19']

nlp = spacy.load('en_core_web_sm', 
                 disable = ['parser', 'tagger', 'ner'])

for i, row in data.iterrows():
    # doing preprocessing for each content element
    sys.stdout.write('\r' + str(i))
    for text_column in text_columns:
        if type(row.loc[text_column]) is not str:
            continue
        data_new.loc[i, text_column] = pipeline(row[text_column], STOP_WORDS, nlp)

Distractrs = []
for i, row in data_new.iterrows():
    sys.stdout.write('\r' + str(i))
    Distractrs.append('. '.join(list((row['D01':'D19'][~row['D01':'D19'].isna()]).values)))

data_new['Distractors'] = Distractrs

data_new['Key_Distractors'] = data_new.loc[:, 'KeyText']+' '+data_new['Distractors']


saving_files = ['train.csv', 
                'test.csv', 
                'val.csv']

# get all the item content
data_new['Text'] =  data_new.loc[:, 'ItemText'] +" "+ data_new.loc[:, 'Key_Distractors']

data_new['Dur_Mean'] = np.log(data_new['Dur_Mean'])
data_new['Dur_SD'] = np.log(data_new['Dur_SD'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_new,
                                                    data_new,
                                                    test_size=0.2, 
                                                    random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.2,
                                                  random_state = 1)


data_new.to_csv('preprocessed_data.csv', index = False)
X_train.to_csv(saving_files[0], index=False)
X_test.to_csv(saving_files[1], index=False)
X_val.to_csv(saving_files[2], index=False)

    
