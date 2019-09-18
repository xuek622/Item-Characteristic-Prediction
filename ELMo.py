# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 05:22:48 2019

@author: Kang Xue
"""

import numpy as np
import os
import pandas as pd
#from allennlp.data import Instance
#from allennlp.data.token_indexers import TokenIndexer
#from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
#from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.nn import util as nn_util

from allennlp.data.vocabulary import Vocabulary
#from allennlp.data.dataset_readers import DatasetReader



from allennlp.data.iterators import BucketIterator


import torch
import torch.nn as nn
import torch.optim as optim

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder



from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder

from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BasicIterator


from MyANNFuncs import Config, Predictor, make_token, MyDatasetReader, SequentialModel

# empty the GPU memory for further application
torch.cuda.empty_cache()

config = Config(
        # setting some hyperparameters for ANN
        testing=False, # if true, just loading a small piece of Data
        seed=1, # random seed
        batch_size = 16, # the bathc_size, not over 16
        lr=5e-3, # learning rate of the optimiztion
        epochs=5, # number of epochs
        hidden_sz = 100, # the output of the LSTM
        max_seq_len = 400, # the maximum number of text length
        max_vocab_size = 100000, # the maximum number of vocabulary
        n_layer = 2, # the number of LSTM layer (encoding layer)
        momentum = 0.8, # the momentum if using SGD
#        training = False,
        )

if config.testing:
    config.epochs = 1
    
USE_GPU = torch.cuda.is_available() # to chech the GPU is available
torch.manual_seed(config.seed) # initialize the torch random seed 
torch.cuda.empty_cache() #clear the GPU memeory


# the type of inputs 
# 
feature_types = ['Text', # item content
                 'ItemText', # item stem
                 'Key_Distractors'] # item options

targets = ['P', 'Time']
targets = ['Time']

#RNN_types = ['LSTM', 'GRU']
RNN_types = ['LSTM']


# three kinds of ELMo (small, middle, original)
OPTION_FILES = ['elmo_2x1024_128_2048cnn_1xhighway_options.json', # small
                 'elmo_2x2048_256_2048cnn_1xhighway_options.json', # middle
                 'elmo_2x4096_512_2048cnn_2xhighway_options.json'] # original
WEIGHT_FILES = ['elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5', # small
                'elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5', # middle
                'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'] # original
ELMo_sizes = ['small', 'middle', 'original']


#data_files = ['train.csv', 'test.csv', 'val.csv'] 

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_val = pd.read_csv('val.csv')

df_all = pd.concat([df_train, df_test, df_val], axis = 0, ignore_index = True)


for ELMo_size in ELMo_sizes:
    if ELMo_size == 'small':
        options_file, weight_file = OPTION_FILES[0], WEIGHT_FILES[0]
        Tuning_status = ['Tuning']
    elif ELMo_size == 'middle':
        options_file, weight_file = OPTION_FILES[1], WEIGHT_FILES[1]
        Tuning_status = ['Tuning']
    else:
        options_file, weight_file = OPTION_FILES[2], WEIGHT_FILES[2]
        Tuning_status = ['Freezing']
 
    for Tuning_state in Tuning_status:
        requires_grad_status = True if Tuning_state == 'Tuning' else False                          
        for target in targets:
            for feature_type in feature_types:
                for RNN_type in RNN_types:
                    
                    label_cols = ['P'] if target == 'P' else ['Dur_Mean', 'Dur_SD']   
                    
                    #directory = feature_type+'_EMLo_'+ ELMo_size+'_'+target+'_'+RNN_type+'_'+Tuning_state+'_'+'.csv'
                    directory = feature_type+'_EMLo_'+ ELMo_size+'_'+Tuning_state+'_'+target+'_'+RNN_type
                    
                    print(directory)
                    
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    
                    
                    # some objects initialization for both training and prediction
                    vocab = Vocabulary()
                    token_indexer = ELMoTokenCharactersIndexer()
                    
                    if not os.path.exists(os.path.join(directory, 'model.th')):
   
                        # initialize the reader which convert dataframe to instance for allennlp
                        reader = MyDatasetReader(
                                tokenizer=make_token, # tokenizer function
                                token_indexers={"tokens": token_indexer}, 
                                max_seq_len = config.max_seq_len, # the max tokens for each input
                                label_cols = label_cols, # variable names of targeting
                                feature_type = feature_type, # the type of raw input
                                testing = config.testing, # if testing the code
                                )
                        
                        # convert dataframe to instance
                        train_ds = reader.read(pd.concat([df_train[feature_type], df_train[label_cols], df_train["Object ID"]], axis = 1))
                        test_ds = reader.read(pd.concat([df_test[feature_type], df_test[label_cols], df_test["Object ID"]], axis = 1))
                        val_ds = reader.read(pd.concat([df_val[feature_type], df_val[label_cols], df_val["Object ID"]], axis = 1))
                        
                        # create vocabulary
    #                        vocab = Vocabulary.from_instances(train_ds + val_ds + test_ds, 
    #                                                          max_vocab_size = config.max_vocab_size)
                        
                        
                        iterator = BucketIterator(batch_size = config.batch_size,
                                                  biggest_batch_first = True,
                                                  sorting_keys=[("tokens", 'num_tokens')])
                
                        iterator.index_with(vocab) # don't forget this step
        
                        # read samples
                        batch = next(iter(iterator(train_ds)))
                        # move data to the GPU memory
                        
                        batch = nn_util.move_to_device(batch, 0 if USE_GPU else -1)
                    
                    
                    #batch['tokens']['tokens'].shape
                    #batch['label'].shape
                    
                    # model initialization
                    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file, 
                                                      requires_grad = requires_grad_status)
                    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})

                    if RNN_type == 'LSTM':
                        # setting encoder be the LSTM
                        encoder: Seq2VecEncoder = PytorchSeq2VecWrapper(nn.LSTM(word_embeddings.get_output_dim(), 
                                                                                config.hidden_sz,
                                                                                num_layers = config.n_layer,
                                                                                bidirectional=True, 
                                                                                batch_first=True))
                    else:
                        # setting encoder be the GRU
                        encoder: Seq2VecEncoder = PytorchSeq2VecWrapper(nn.GRU(word_embeddings.get_output_dim(), 
                                                                               config.hidden_sz,
                                                                               num_layers = config.n_layer,
                                                                               bidirectional=True, 
                                                                               batch_first=True))         
                    
                    
#################################################### Model training ####################################
                    if not os.path.exists(os.path.join(directory, 'model.th')): # if the model is not trained
 
                        model = SequentialModel(
                                    word_embeddings, 
                                    encoder, 
                                    out_sz = len(label_cols),
                                    vocab = vocab
                                    )
                        
                        # initialize the ANN structure
                    
                        if USE_GPU: model.cuda() # using GPU
                        else: model # using CPU
                        
                        # initialize the optimizer
                        #optimizer = optim.Adam(model.parameters(), lr=config.lr)
                        optimizer = optim.SGD(model.parameters(), 
                                              lr=config.lr, 
                                              momentum = config.momentum)
    
                        trainer = Trainer(
                            model=model, # ANN models
                            optimizer=optimizer,
                            iterator=iterator,
                            train_dataset=train_ds,
                            validation_dataset = val_ds,
                            cuda_device=0 if USE_GPU else -1,
                            num_epochs=config.epochs,
                            patience = 1,
                        )
    
                        metrics = trainer.train()
                        print(torch.cuda.memory_cached())
                        print(torch.cuda.memory_allocated())
    
                        #  save the model.
                        with open(os.path.join(directory, 'model.th'), 'wb') as f:
                            torch.save(model.state_dict(), f)
                    
                                

 #############################################  Feature Extraction ####################################### 
                    
                    #predictor = Predictor(model, seq_iterator, cuda_device=0 if USE_GPU else -1) # object to get the features
             
                    # Item content feature
                    
                    raw_inputs = ['ItemText', 'KeyText', 'Text',
                                  'Distractors', 'Key_Distractors', 
                                  'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 
                                  'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 
                                  'D13','D14', 'D15', 'D16', 'D17', 'D18', 'D19']

                    seq_iterator = BasicIterator(batch_size=32)
                    seq_iterator.index_with(vocab)                    
                    
                    for raw_input in raw_inputs:
                        df_all_copy = df_all.copy()
                        print('extract features of '+raw_input+' ...')
                        memory_cached = torch.cuda.memory_cached()
                        memory_allocated = torch.cuda.memory_allocated()
                        print(memory_cached)
                        print(memory_allocated)
                        
                        if 'train_ds' in locals():
                            del train_ds, test_ds, val_ds, model
                            print('deleting the template files to release GPU memory...')
                            torch.cuda.empty_cache()
                            print(memory_cached - torch.cuda.memory_cached())
                            print(memory_allocated - torch.cuda.memory_allocated())

                        if os.path.isfile(os.path.join(directory, raw_input+'_feature.csv')):
                            continue
                        
                        model = SequentialModel(
                                word_embeddings, 
                                encoder, 
                                out_sz = len(label_cols),
                                vocab = vocab
                                )
                        
                          
                        with open(os.path.join(directory, 'model.th'), 'rb') as f:
                            model.load_state_dict(torch.load(f))

                        model.cuda()
                        
                        predictor = Predictor(model, seq_iterator, cuda_device=0 if USE_GPU else -1)
                        
                        reader = MyDatasetReader(
                                tokenizer=make_token, # tokenizer function
                                token_indexers={"tokens": token_indexer}, 
                                max_seq_len = config.max_seq_len, # the max tokens for each input
                                label_cols = label_cols, # variable names of targeting
                                feature_type = raw_input, # the type of raw input
                                testing = config.testing, # if testing the code
                                )
                        
                        raw_input_index = ~df_all.loc[:, raw_input].isna()
                        if any(raw_input_index):
                            all_ds = reader.read(df_all.loc[raw_input_index, :])
                            embedding, encoding, prediction = predictor.predict(all_ds)
            
                            # initialize the embedding and enoding feature matrix
                            embedding_feature = np.ones((len(raw_input_index), embedding.shape[1]))*np.nan
                            encoding_feature = np.ones((len(raw_input_index), encoding.shape[1]))*np.nan
                            
                            # for some items the feature is NA if it doesn't have such distractor
                            embedding_feature[raw_input_index, :] = embedding
                            encoding_feature[raw_input_index, :] = encoding
                            
                            # saving  
                            embedding_feature = pd.DataFrame(embedding_feature, columns = [raw_input+'_embedding_'+str(x+1) for x in range(embedding_feature.shape[1])])
                            encoding_feature = pd.DataFrame(encoding_feature, columns = [raw_input+'_emcoding_'+str(x+1) for x in range(encoding_feature.shape[1])])
                            
                            
                            if(raw_input == 'Text'):
                                prediction = pd.DataFrame(prediction, columns = label_cols)
                                prediction.to_csv(os.path.join(directory, 'Prediction.csv'))
                            
                            df_feature = pd.concat([embedding_feature, encoding_feature], axis = 1)
                            
                            file = os.path.join(directory, raw_input+'_feature.csv')
                            
                            pd.concat([df_all, df_feature], axis=1).to_csv(file, index = False)
                            
                            del all_ds
                            torch.cuda.empty_cache()
#                    train_embedding_feature, train_encoding_feature, train_prediction = predictor.predict(train_ds) # get the feature of training set 
#                    test_embedding_feature, test_encoding_feature, test_prediction = predictor.predict(test_ds) # get the feature of testing set
#                    val_embedding_feature, val_encoding_feature, val_prediction = predictor.predict(val_ds)
#                    
#                    
#                    np.savetxt(fname=saving_files[0], X = train_embedding_feature, delimiter=',')
#                    np.savetxt(fname=saving_files[1], X = train_encoding_feature, delimiter=',')
#                    np.savetxt(fname=saving_files[2], X = train_prediction, delimiter=',')
#                    
#                    np.savetxt(fname=saving_files[3], X = test_embedding_feature, delimiter=',')
#                    np.savetxt(fname=saving_files[4], X = test_encoding_feature, delimiter=',')
#                    np.savetxt(fname=saving_files[5], X = test_prediction, delimiter=',')

                    torch.cuda.empty_cache() # empty the GPU memory for next activation

