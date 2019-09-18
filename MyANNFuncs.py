# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:21:03 2019

@author: Kang Xue
"""
import torch
import numpy as np
import pandas as pd

from allennlp.nn import util as nn_util
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer

from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from spacy.lang.en.stop_words import STOP_WORDS



class Config(dict):
    # set the configuration for ANNs
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


def tonp(tsr): 
    # used to convert the data in GPU memery to np.array()
    return tsr.detach().cpu().numpy()


#def tokenizer(x):
#    tokens = [w.text for w in 
#             SpacyWordSplitter(language='en_core_web_sm', 
#                               pos_tags=False).split_words(x)[:400]]
#    return [token for token in tokens]

def make_token(x):
    return x.split()


from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ArrayField
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data import Instance
from allennlp.data.tokenizers import Token

class MyDatasetReader(DatasetReader):
    # this class is used to load raw data and convert it to instance for allennlp
    def __init__(self, tokenizer,
                 token_indexers,
                 max_seq_len,
                 label_cols,
                 feature_type,
                 testing):
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_seq_len = max_seq_len
        self.label_cols = label_cols
        self.feature_type = feature_type
        self.testing = testing

    def text_to_instance(self, tokens, id, labels):
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}
        
        id_field = MetadataField(id)
        fields["id"] = id_field
        
        if labels is None:
            labels = np.zeros(len(self.label_cols))
        label_field = ArrayField(array=labels)
        fields["label"] = label_field
        # field is a dict
        return Instance(fields)
    
    def _read(self, df):
        #df = pd.read_csv(file_path)
        if self.testing: df = df.head(50)
        for i, row in df.iterrows():
            yield self.text_to_instance(
                [Token(x) for x in self.tokenizer(row[self.feature_type])],
                row["Object ID"], row[self.label_cols].values,
            )
            
            

import torch.nn as nn         
from allennlp.models import Model  
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.text_field_embedders import TextFieldEmbedder                         
class SequentialModel(Model):
    # build up the sequential models
    # this class could be moved to MyANNFuncs.py
    def __init__(self, word_embeddings,
                 encoder,
                 out_sz,
                 vocab):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.predict = nn.Linear(self.encoder.get_output_dim(), # input is output of encode layer 
                                 out_sz) # out_sz is the dimenstion of target
        self.relu = nn.ReLU() # relu
        self.loss = nn.MSELoss() # regression using MSE as loss function
       # self.softmax = nn.Softmax()
        
    def forward(self, tokens,
                id, label):
        mask = get_text_field_mask(tokens) # where the tokens are padding
        embeddings = self.word_embeddings(tokens) # embedding outputs
        encoding = self.encoder(embeddings, mask) # encoding layer outputs
        #state = self.relu(state)
        output = self.predict(encoding) # single dense layer for regression
                                      
        output = {"output": output}
        output["embeddings"] = embeddings
        output['feature'] = encoding
        output["loss"] = self.loss(output['output'], label)
    
        return output

class Predictor:
    # the predictor is to output:
    # 1- embedding feature
    # 2- encoding feature
    # 3- the prediction of the ANNs
    def __init__(self, model, iterator,
                 cuda_device: int=-1):
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device
        
    def _extract_data(self, batch):
        out_dict = self.model(**batch)
        outputs = {'embeddings': tonp(out_dict['embeddings'])} # embedding feature
        outputs['feature'] = tonp(out_dict['feature']) # encoding feature
        outputs['output'] = tonp(out_dict['output']) # prediction
        #return expit(tonp(out_dict["output"])) # if return a probability
        return  outputs
    
    def predict(self, ds):
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        
        embedding_feature, encoding_feature, prediction = [], [], []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                
                outputs = self._extract_data(batch)
                #print(outputs)
                
                for i in range(outputs['embeddings'].shape[0]):
                    embedding = outputs['embeddings'][i]
                    # applying the average pooling
                    # the embedding feature of the input is the mean
                    # of the embeddings of all single token
                    embedding_avg = np.mean(embedding[~np.apply_along_axis(all, 1, embedding == 0.), :], 
                                                      axis = 0) 
                    embedding_feature.append(embedding_avg)
                     
                    encoding_feature.append(outputs['feature'][i])
                    prediction.append(outputs['output'][i])
                     
        return np.array(embedding_feature), np.array(encoding_feature), np.array(prediction)