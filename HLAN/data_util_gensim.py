# -*- coding: utf-8 -*-
import codecs
import numpy as np
#load data of zhihu
#import word2vec
from gensim.models import Word2Vec
import os
import pickle
PAD_ID = 0
from tflearn.data_utils import pad_sequences
_GO="_GO"
_END="_END"
_PAD="_PAD"
import random
from tqdm import tqdm

def get_label_sub_matrix(vocabulary_word2index_label,kb_path,name_scope=''):
    cache_path ='../cache_vocabulary_label_pik/'+ name_scope + "_label_sub.pik"
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            result=pickle.load(data_f)
            return result
    else:
        # load label embedding
        m = len(vocabulary_word2index_label)
        result=np.zeros((m,m))
        print('vocabulary_word2index_label length:',m)
        with open(kb_path, 'r') as label_pairs:
            lps = label_pairs.readlines() # lps: label pairs
        lps = [x.strip() for x in lps]
        for lp in lps:
            labels = lp.split(',')
            if len(labels) == 3 and labels[-1] == 'true' or len(labels) == 2:
                index_j = vocabulary_word2index_label.get(labels[0].lower(),-1)
                index_k = vocabulary_word2index_label.get(labels[1].lower(),-1)
                if index_j != -1 and index_k != -1 and index_j != index_k: # if both of the two labels are in the training data, and they are different from each other (diagonal as 0).
                    result[index_j,index_k] = 1.
                    print('matched:', labels[0], str(index_j), labels[1], str(index_k))
        #save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f:
                pickle.dump(result, data_f)
    return result  
    
# a weighted
def get_label_sim_matrix(vocabulary_index2word_label,word2vec_model_label_path='../tag-all.bin-300',name_scope='',threshold=0):
    '''here the word2vec_model should have embedding for all the labels
        otherwise there will be a KeyError from Gensim'''
    cache_path ='../cache_vocabulary_label_pik/'+ name_scope + "_label_sim_" + str(threshold) + ".pik"
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            result=pickle.load(data_f)
            return result
    else:
        #model=word2vec.load(word2vec_model_label_path,kind='bin') # for danielfrg's word2vec models
        model = Word2Vec.load(word2vec_model_label_path) # for gensim word2vec models

        #m = model.vectors.shape[0]-1 #length # the first one is </s>, to be eliminated
        m = len(vocabulary_index2word_label)
        result=np.zeros((m,m))
        count_less_th = 0.0 # count the sim less than the threshold
        for i in tqdm(range(0,m)):
            for j in range(0,m):
                #vector_i=model.get_vector(vocabulary_index2word_label[i]) # for danielfrg's word2vec models
                #vector_j=model.get_vector(vocabulary_index2word_label[j]) # for danielfrg's word2vec models
                #print(vocabulary_index2word_label[i],vocabulary_index2word_label[j])
                vector_i=model.wv[vocabulary_index2word_label[i]] # for gensim word2vec models
                vector_j=model.wv[vocabulary_index2word_label[j]] # for gensim word2vec models
                #unit normalise
                vector_i = vector_i/np.linalg.norm(vector_i)
                vector_j = vector_j/np.linalg.norm(vector_j)
                #result[i][j] = np.dot(vector_i,vector_j.T) # can be negative here, result in [-1,1]
                result[i][j] = (1+np.dot(vector_i,vector_j.T))/2 # result in [0,1]
                if result[i][j] < threshold:
                    print('less than 0:',vocabulary_index2word_label[i],vocabulary_index2word_label[j],result[i][j])
                    count_less_th = count_less_th + 1
                    result[i][j] = 0
        print("result",result)
        print("result",result.shape)
        print("retained similarities percentage:", str(1-count_less_th/float(m)/float(m)))
        #save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f:
                pickle.dump(result, data_f)
    return result

def create_vocabulary(word2vec_model_path,name_scope=''):
    cache_path ='../cache_vocabulary_label_pik/'+ name_scope + "_word_vocabulary.pik"
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index, vocabulary_index2word=pickle.load(data_f)
            return vocabulary_word2index, vocabulary_index2word
    else:
        vocabulary_word2index={}
        vocabulary_index2word={}
        print("create vocabulary. word2vec_model_path:",word2vec_model_path)
        #model=word2vec.load(word2vec_model_path,kind='bin') # for danielfrg's word2vec models
        model = Word2Vec.load(word2vec_model_path) # for gensim word2vec models
        vocabulary_word2index['PAD_ID']=0
        vocabulary_index2word[0]='PAD_ID'
        special_index=0
        if 'biLstmTextRelation' in name_scope:
            vocabulary_word2index['EOS']=1 # a special token for biLstTextRelation model. which is used between two sentences.
            vocabulary_index2word[1]='EOS'
            special_index=1
        for i,vocab in enumerate(model.wv.vocab.keys()):
            #if vocab == '':
            #    print(i,vocab)
            vocabulary_word2index[vocab]=i+1+special_index
            vocabulary_index2word[i+1+special_index]=vocab

        #save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path):
            with open(cache_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index,vocabulary_index2word), data_f)
    return vocabulary_word2index,vocabulary_index2word

# create vocabulary of labels. label is sorted. 1 is high frequency, 2 is low frequency.
def create_vocabulary_label_for_predict(name_scope=''):
    cache_path ='../cache_vocabulary_label_pik/'+ name_scope + "_label_vocabulary.pik"
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index_label, vocabulary_index2word_label=pickle.load(data_f)
            return vocabulary_word2index_label, vocabulary_index2word_label
    else:
        return None,None
        
# create vocabulary of labels. label is sorted. 1 is high frequency, 2 is low frequency.
def create_vocabulary_label_pre_split(training_data_path,validation_data_path,testing_data_path,name_scope='',use_seq2seq=False,label_freq_th=0):
    '''
    create vocabulary from data split files - validation data path can be None or empty string if not exists.
    '''
    cache_path ='../cache_vocabulary_label_pik/'+ name_scope + "_label_vocabulary.pik"
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index_label, vocabulary_index2word_label=pickle.load(data_f)
            return vocabulary_word2index_label, vocabulary_index2word_label
    else:
        count=0
        vocabulary_word2index_label={}
        vocabulary_index2word_label={}
        vocabulary_label_count_dict={} #{label:count}
        
        # the list of data split files: not including validation data if it is set as None
        list_data_split_path = [training_data_path,validation_data_path,testing_data_path] if validation_data_path != None and validation_data_path != '' else [training_data_path,testing_data_path] 
        for data_path in list_data_split_path:
            print("create_vocabulary_label_sorted.started.data_path:",data_path)
            #zhihu_f_train = codecs.open(data_path, 'r', 'utf8')
            zhihu_f_train = codecs.open(data_path, 'r', 'latin-1')
            lines=zhihu_f_train.readlines()
            for i,line in enumerate(lines):
                if '__label__' in line:  #'__label__-2051131023989903826
                    label=line[line.index('__label__')+len('__label__'):].strip().replace("\n","")
                    # add multi-label processing
                    #print(label)
                    labels=label.split(" ")
                    for label in labels:
                        if label == '':
                            print('found empty label!') 
                            continue # this is a quick fix of the empty label problem, simply not recording it.
                        if vocabulary_label_count_dict.get(label,None) is not None:
                            vocabulary_label_count_dict[label]=vocabulary_label_count_dict[label]+1
                        else:
                            vocabulary_label_count_dict[label]=1
        list_label=sort_by_value(vocabulary_label_count_dict) # sort the labels by their frequency in the training dataset.

        print("length of list_label:",len(list_label));#print(";list_label:",list_label)
        countt=0

        ##########################################################################################
        if use_seq2seq:#if used for seq2seq model,insert two special label(token):_GO AND _END
            i_list=[0,1,2];label_special_list=[_GO,_END,_PAD]
            for i,label in zip(i_list,label_special_list):
                vocabulary_word2index_label[label] = i
                vocabulary_index2word_label[i] = label
        #########################################################################################
        for i,label in enumerate(list_label):
            if i<10:
                count_value=vocabulary_label_count_dict[label]
                print("label:",label,"count_value:",count_value)
                countt=countt+count_value
            if vocabulary_label_count_dict[label]>=label_freq_th:
                indexx = i + 3 if use_seq2seq else i
                vocabulary_word2index_label[label]=indexx
                vocabulary_index2word_label[indexx]=label
        print("count top10:",countt)

        #save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path): #如果不存在写到缓存文件中
            with open(cache_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index_label,vocabulary_index2word_label), data_f)
    print("create_vocabulary_label_sorted.ended.len of vocabulary_label:",len(vocabulary_index2word_label))
    return vocabulary_word2index_label,vocabulary_index2word_label
    
# create vocabulary of lables. label is sorted. 1 is high frequency, 2 is low frequency.
def create_vocabulary_label(vocabulary_label,name_scope='',use_seq2seq=False,label_freq_th=0):
    print("create_vocabulary_label_sorted.started.full_data_path:",vocabulary_label)
    cache_path ='../cache_vocabulary_label_pik/'+ name_scope + "_label_vocabulary.pik"
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index_label, vocabulary_index2word_label=pickle.load(data_f)
            return vocabulary_word2index_label, vocabulary_index2word_label
    else:
        zhihu_f_train = codecs.open(vocabulary_label, 'r', 'utf8')
        lines=zhihu_f_train.readlines()
        count=0
        vocabulary_word2index_label={}
        vocabulary_index2word_label={}
        vocabulary_label_count_dict={} #{label:count}
        for i,line in enumerate(lines):
            if '__label__' in line:  #'__label__-2051131023989903826
                label=line[line.index('__label__')+len('__label__'):].strip().replace("\n","")
                # add multi-label processing
                #print(label)
                labels=label.split(" ")
                for label in labels:
                    if label == '':
                        print('found empty label!') 
                        continue # this is a quick fix of the empty label problem, simply not recording it.
                    if vocabulary_label_count_dict.get(label,None) is not None:
                        vocabulary_label_count_dict[label]=vocabulary_label_count_dict[label]+1
                    else:
                        vocabulary_label_count_dict[label]=1
        list_label=sort_by_value(vocabulary_label_count_dict) # sort the labels by their frequency in the training dataset.

        print("length of list_label:",len(list_label));#print(";list_label:",list_label)
        countt=0

        ##########################################################################################
        if use_seq2seq:#if used for seq2seq model,insert two special label(token):_GO AND _END
            i_list=[0,1,2];label_special_list=[_GO,_END,_PAD]
            for i,label in zip(i_list,label_special_list):
                vocabulary_word2index_label[label] = i
                vocabulary_index2word_label[i] = label
        #########################################################################################
        for i,label in enumerate(list_label):
            if i<10:
                count_value=vocabulary_label_count_dict[label]
                print("label:",label,"count_value:",count_value)
                countt=countt+count_value
            if vocabulary_label_count_dict[label]>=label_freq_th:
                indexx = i + 3 if use_seq2seq else i
                vocabulary_word2index_label[label]=indexx
                vocabulary_index2word_label[indexx]=label
        print("count top10:",countt)

        #save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path): #如果不存在写到缓存文件中
            with open(cache_path, 'ab') as data_f:
                pickle.dump((vocabulary_word2index_label,vocabulary_index2word_label), data_f)
    print("create_vocabulary_label_sorted.ended.len of vocabulary_label:",len(vocabulary_index2word_label))
    return vocabulary_word2index_label,vocabulary_index2word_label

def sort_by_value(d):
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort(reverse=True)
    return [ backitems[i][1] for i in range(0,len(backitems))]

#loading data with pre-defined split
def load_data_multilabel_pre_split(vocabulary_word2index,vocabulary_word2index_label,keep_label_percent=1,data_path='',multi_label_flag=True,use_seq2seq=False,seq2seq_label_length=6,label_missing=False):  # n_words=100000,
    """
    input: a file path
    :return: data. where data=(X, Y). where
                X: is a list of list.each list representation a sentence.Y: is a list of label. each label is a number
    """
    # 1.load a zhihu data from file
    # example:"w305 w6651 w3974 w1005 w54 w109 w110 w3974 w29 w25 w1513 w3645 w6 w111 __label__-400525901828896492"
    print("load_data.started...")
    print("load_data_multilabel_new.data_path:",data_path)
    #zhihu_f = codecs.open(data_path, 'r', 'utf8') #-zhihu4-only-title.txt
    zhihu_f = codecs.open(data_path, 'r', 'latin-1') #-zhihu4-only-title.txt
    lines = zhihu_f.readlines()
    # 2.transform X as indices
    # 3.transform y as scalar
    X = []
    Y = [] # full labels
    Y_missing = [] # missing labels
    Y_decoder_input=[] #ADD 2017-06-15
    for i, line in enumerate(lines):
        x, y = line.split('__label__') #x='w17314 w5521 w7729 w767 w10147 w111'
        y=y.strip().replace('\n','')
        x = x.strip()
        if i<1:
            print(i,"x0:",x) #get raw x
        #x_=process_one_sentence_to_get_ui_bi_tri_gram(x)
        #1) Get indexes of word sequences
        x=x.split(" ")
        x = [vocabulary_word2index.get(e,0) for e in x] #if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        # see https://stackoverflow.com/questions/11041405/why-dict-getkey-instead-of-dictkey
        if i<2:
            print(i,"x1:",x) #word to index
            
        #2) Get multihot representation of label sets
        if use_seq2seq:        # 1)prepare label for seq2seq format(ADD _GO,_END,_PAD for seq2seq)
            ys = y.replace('\n', '').split(" ")  # ys is a list
            _PAD_INDEX=vocabulary_word2index_label[_PAD]
            ys_mulithot_list=[_PAD_INDEX]*seq2seq_label_length #[3,2,11,14,1]
            ys_decoder_input=[_PAD_INDEX]*seq2seq_label_length
            # below is label.
            for j,y in enumerate(ys):
                if j<seq2seq_label_length-1:
                    ys_mulithot_list[j]=vocabulary_word2index_label[y]
            if len(ys)>seq2seq_label_length-1:
                ys_mulithot_list[seq2seq_label_length-1]=vocabulary_word2index_label[_END]#ADD END TOKEN
            else:
                ys_mulithot_list[len(ys)] = vocabulary_word2index_label[_END]

            # below is input for decoder.
            ys_decoder_input[0]=vocabulary_word2index_label[_GO]
            for j,y in enumerate(ys):
                if j < seq2seq_label_length - 1:
                    ys_decoder_input[j+1]=vocabulary_word2index_label[y]
            if i<10:
                print(i,"ys:==========>0", ys)
                print(i,"ys_mulithot_list:==============>1", ys_mulithot_list)
                print(i,"ys_decoder_input:==============>2", ys_decoder_input)
        else:
            if multi_label_flag: # 2)prepare multi-label format for classification
                ys = y.replace('\n', '').split(" ")  # ys is a list
                ys_index=[]
                for y in ys:
                    #y_index = vocabulary_word2index_label[y]
                    #ys_index.append(y_index)
                    if vocabulary_word2index_label.get(y,None) is not None:
                        y_index = vocabulary_word2index_label[y]
                        ys_index.append(y_index)
                if ys_index == []: # if it is empty, due to no labels have their frequency above the threshold label_freq_th.
                    continue
                # truncating the label by keep_label_percent    
                ys_index_missing = ys_index
                if i<2:
                    print('original ys_index:',ys_index_missing)    
                random.seed(1234)
                random.shuffle(ys_index_missing)
                ys_index_missing = ys_index_missing[:round(len(ys_index_missing)*keep_label_percent)]
                if i<2:
                    print('truncated ys_index by',str(keep_label_percent*100),'percent:',ys_index_missing)     
                    
                ys_mulithot_list=transform_multilabel_as_multihot(ys_index,len(vocabulary_word2index_label))
                ys_mulithot_list_missing=transform_multilabel_as_multihot(ys_index_missing,len(vocabulary_word2index_label))
            else:                #3)prepare single label format for classification
                ys_mulithot_list=vocabulary_word2index_label[y]
        if i<=3:
            print("ys_index:")
            #print(ys_index)
            print(i,"y:",ys," ;ys_mulithot_list:",ys_mulithot_list) #," ;ys_decoder_input:",ys_decoder_input)
        
        #3) store word indexes and multihot label sets into X and Y (with simulation of missing label cases as an option)        
        X.append(x)
        Y.append(ys_mulithot_list) #every element in Y is a multihot list, a 5770 dimensional vector.
        Y_missing.append(ys_mulithot_list_missing)
        if use_seq2seq:
            Y_decoder_input.append(ys_decoder_input) #decoder input
        #if i>50000:
        #    break
    if label_missing:
        data = (X,Y_missing)
    else:
        data = (X,Y)
    print("load_data.ended...")
    return data

#loading data with pre-defined split for prediction (i.e. no labels)
def load_data_multilabel_pre_split_for_pred(vocabulary_word2index,vocabulary_word2index_label,data_path='',verbose=True):
    """
    input: a file path
    :return: data. where data=(X, Y). where
                X: is a list of list.each list representation a sentence.
                Y: is a list of *fake (empty)* label if no labels assigned; or a list of ground truth labels. each label is a binary number in a multihot representation.
    """
    # 1.load a zhihu data from file
    # example:"w305 w6651 w3974 w1005 w54 w109 w110 w3974 w29 w25 w1513 w3645 w6 w111 __label__-400525901828896492"
    if verbose:
        print("\nload_data.started...")
        print("load_data_multilabel_new.data_path:",data_path,'\n')
    #zhihu_f = codecs.open(data_path, 'r', 'utf8') #-zhihu4-only-title.txt
    zhihu_f = codecs.open(data_path, 'r', 'latin-1') #-zhihu4-only-title.txt
    lines = zhihu_f.readlines()
    # 2.transform X as indices
    # 3.transform y as scalar
    X = []
    Y = []
    for i, line in enumerate(lines):
        # if no labels assigned, treat as empty labels (this won't affect the prediction)
        # otherwise, use the ground truth label set
        parts = line.split('__label__') 
        if len(parts)==1:
            x = parts[0]
            y = ""            
        elif len(parts)==2:
            x,y = parts
            y=y.strip().replace('\n','')
            #print(y)
        else:
            print('data format wrong: num_parts as %s' % len(parts))
        
        #1) Get indexes of word sequences        
        if i<1 and verbose:
            print("---------- After Preprocessing (sentence parsing, word tokenisation, padding) ----------")
            print(x) #get raw x
        #x_=process_one_sentence_to_get_ui_bi_tri_gram(x)
        #1) Get indexes of word sequences
        x=x.split(" ")
        x = [vocabulary_word2index.get(e,0) for e in x] #if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        # see https://stackoverflow.com/questions/11041405/why-dict-getkey-instead-of-dictkey
        if i<1 and verbose:
            print("---------- After changing to vocabulary indexes ----------")
            print(x) #word to index
        
        #2) Get multihot representation of label sets
        ys_index=[]
        ys = y.replace('\n', '').split(" ")
        for y in ys:
            #y_index = vocabulary_word2index_label[y]
            #ys_index.append(y_index)
            if vocabulary_word2index_label.get(y,None) is not None:
                y_index = vocabulary_word2index_label[y]
                ys_index.append(y_index)
        ys_mulithot_list=transform_multilabel_as_multihot(ys_index,len(vocabulary_word2index_label)) 
        #print(ys_mulithot_list)
        #3) store word indexes and multihot label sets into X and Y (with simulation of missing label cases as an option) 
        X.append(x)
        Y.append(ys_mulithot_list)
    data = (X,Y)
    if verbose:
        print("load_data.ended...")
    return data
    
#loading data with held-out evaluation    
def load_data_multilabel_new(vocabulary_word2index,vocabulary_word2index_label,keep_label_percent=1,valid_portion=0.111,test_portion=0.1,max_training_data=1000000,training_data_path='',multi_label_flag=True,use_seq2seq=False,seq2seq_label_length=6):  # n_words=100000,
    """
    input: a file path
    :return: train, valid, test. where train=(trainX, trainY). where
                trainX: is a list of list.each list representation a sentence.trainY: is a list of label. each label is a number
    """
    # 1.load a zhihu data from file
    # example:"w305 w6651 w3974 w1005 w54 w109 w110 w3974 w29 w25 w1513 w3645 w6 w111 __label__-400525901828896492"
    print("load_data.started...")
    print("load_data_multilabel_new.training_data_path:",training_data_path)
    zhihu_f = codecs.open(training_data_path, 'r', 'utf8') #-zhihu4-only-title.txt
    lines = zhihu_f.readlines()
    # 2.transform X as indices
    # 3.transform y as scalar
    X = []
    Y = [] # full labels
    Y_missing = [] # missing labels
    Y_decoder_input=[] #ADD 2017-06-15
    for i, line in enumerate(lines):
        x, y = line.split('__label__') #x='w17314 w5521 w7729 w767 w10147 w111'
        y=y.strip().replace('\n','')
        x = x.strip()
        if i<1:
            print(i,"x0:",x) #get raw x
        #x_=process_one_sentence_to_get_ui_bi_tri_gram(x)
        x=x.split(" ")
        x = [vocabulary_word2index.get(e,0) for e in x] #if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        # see https://stackoverflow.com/questions/11041405/why-dict-getkey-instead-of-dictkey
        if i<2:
            print(i,"x1:",x) #word to index
        if use_seq2seq:        # 1)prepare label for seq2seq format(ADD _GO,_END,_PAD for seq2seq)
            ys = y.replace('\n', '').split(" ")  # ys is a list
            _PAD_INDEX=vocabulary_word2index_label[_PAD]
            ys_mulithot_list=[_PAD_INDEX]*seq2seq_label_length #[3,2,11,14,1]
            ys_decoder_input=[_PAD_INDEX]*seq2seq_label_length
            # below is label.
            for j,y in enumerate(ys):
                if j<seq2seq_label_length-1:
                    ys_mulithot_list[j]=vocabulary_word2index_label[y]
            if len(ys)>seq2seq_label_length-1:
                ys_mulithot_list[seq2seq_label_length-1]=vocabulary_word2index_label[_END]#ADD END TOKEN
            else:
                ys_mulithot_list[len(ys)] = vocabulary_word2index_label[_END]

            # below is input for decoder.
            ys_decoder_input[0]=vocabulary_word2index_label[_GO]
            for j,y in enumerate(ys):
                if j < seq2seq_label_length - 1:
                    ys_decoder_input[j+1]=vocabulary_word2index_label[y]
            if i<10:
                print(i,"ys:==========>0", ys)
                print(i,"ys_mulithot_list:==============>1", ys_mulithot_list)
                print(i,"ys_decoder_input:==============>2", ys_decoder_input)
        else:
            if multi_label_flag: # 2)prepare multi-label format for classification
                ys = y.replace('\n', '').split(" ")  # ys is a list
                ys_index=[]
                for y in ys:
                    #y_index = vocabulary_word2index_label[y]
                    #ys_index.append(y_index)
                    if vocabulary_word2index_label.get(y,None) is not None:
                        y_index = vocabulary_word2index_label[y]
                        ys_index.append(y_index)
                if ys_index == []: # if it is empty, due to no labels have their frequency above the threshold label_freq_th.
                    continue
                # truncating the label by keep_label_percent    
                ys_index_missing = ys_index
                if i<2:
                    print('original ys_index:',ys_index_missing)    
                random.seed(1234)
                random.shuffle(ys_index_missing)
                ys_index_missing = ys_index_missing[:round(len(ys_index_missing)*keep_label_percent)]
                if i<2:
                    print('truncated ys_index by',str(keep_label_percent*100),'percent:',ys_index_missing)     
                    
                ys_mulithot_list=transform_multilabel_as_multihot(ys_index,len(vocabulary_word2index_label))
                ys_mulithot_list_missing=transform_multilabel_as_multihot(ys_index_missing,len(vocabulary_word2index_label))
            else:                #3)prepare single label format for classification
                ys_mulithot_list=vocabulary_word2index_label[y]
        if i<=3:
            print("ys_index:")
            #print(ys_index)
            print(i,"y:",ys," ;ys_mulithot_list:",ys_mulithot_list) #," ;ys_decoder_input:",ys_decoder_input)
        X.append(x)
        Y.append(ys_mulithot_list) #every element in Y is a multihot list, a 5770 dimensional vector.
        Y_missing.append(ys_mulithot_list_missing)
        if use_seq2seq:
            Y_decoder_input.append(ys_decoder_input) #decoder input
        #if i>50000:
        #    break
        
    # 4.split to train,test and valid data, here using the first (1 - valid_portion) percentage as training, and valid_portion percentage as testing.
    number_examples = len(X)
    print("number_examples:",number_examples) #
    test = (X[int((1 - test_portion) * number_examples):], Y[int((1 - test_portion) * number_examples):]) # get test first
    
    X_train_valid = X[:int((1 - test_portion) * number_examples)] # get the train+valid
    Y_train_valid = Y[:int((1 - test_portion) * number_examples)]
    Y_missing_train_valid = Y_missing[:int((1 - test_portion) * number_examples)]
    
    # get train and valid from train+valid
    number_examples_tv = len(X_train_valid) # number of examples for training and validation.
    print("number_examples_tv:",number_examples_tv) #
    train = (X_train_valid[0:int((1 - valid_portion) * number_examples_tv)], Y_missing_train_valid[0:int((1 - valid_portion) * number_examples_tv)]) # tuple of lists # using Y_missing in training data
    valid = (X_train_valid[int((1 - valid_portion) * number_examples_tv) + 1:], Y_train_valid[int((1 - valid_portion) * number_examples_tv) + 1:]) # using Y in test data
    # no validation here. test only. 0.05%.
    if use_seq2seq:
        train=train+(Y_decoder_input[0:int((1 - valid_portion) * number_examples)],)
        test=test+(Y_decoder_input[int((1 - valid_portion) * number_examples) + 1:],)
    # 5.return
    print("load_data.ended...")
    return train, valid, test

# #loading data with spliting training, validation, and testing dataset to k-fold sets
def load_data_multilabel_new_k_fold(vocabulary_word2index,vocabulary_word2index_label,keep_label_percent=1,kfold=10,test_portion=0.1,max_training_data=1000000,training_data_path='',multi_label_flag=True,use_seq2seq=False,seq2seq_label_length=6,shuffle=False):
    """
    input: a file path
    :return: train, valid, test. where train is a list of kfold tuples(datasets), separated based on k-fold cross-validation, for example, train[0] is a tuple (trainX, trainY).
                                       valid is a list of kfold tuples, valid[0] is a tuple (validX, validY)
                                       test is a list of 1 tuple, test[0] is a tuple (testX, testY)
    """
    # 0.initialise k-fold training and testing data as lists
    train, valid, test=list(), list(), list()
    # 1.load a zhihu data from file
    # example:"w305 w6651 w3974 w1005 w54 w109 w110 w3974 w29 w25 w1513 w3645 w6 w111 __label__-400525901828896492"
    print("load_data.started...")
    print("load_data_multilabel_new.training_data_path:",training_data_path)
    zhihu_f = codecs.open(training_data_path, 'r', 'utf8') #-zhihu4-only-title.txt
    lines = zhihu_f.readlines()
    if shuffle:
        random.seed(1234)
        random.shuffle(lines)        
    # 2.transform X as indices
    # 3.transform  y as scalar
    X = []
    Y = []
    Y_missing = [] # missing labels
    Y_decoder_input=[] #ADD 2017-06-15
    for i, line in enumerate(lines):
        x, y = line.split('__label__') #x='w17314 w5521 w7729 w767 w10147 w111'
        y=y.strip().replace('\n','')
        x = x.strip()
        if i<1:
            print(i,"x0:",x) #get raw x
        #x_=process_one_sentence_to_get_ui_bi_tri_gram(x)
        x=x.split(" ")
        x = [vocabulary_word2index.get(e,0) for e in x] #if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        # see https://stackoverflow.com/questions/11041405/why-dict-getkey-instead-of-dictkey
        if i<2:
            print(i,"x1:",x) #word to index
        if use_seq2seq:        # 1)prepare label for seq2seq format(ADD _GO,_END,_PAD for seq2seq)
            ys = y.replace('\n', '').split(" ")  # ys is a list
            _PAD_INDEX=vocabulary_word2index_label[_PAD]
            ys_mulithot_list=[_PAD_INDEX]*seq2seq_label_length #[3,2,11,14,1]
            ys_decoder_input=[_PAD_INDEX]*seq2seq_label_length
            # below is label.
            for j,y in enumerate(ys):
                if j<seq2seq_label_length-1:
                    ys_mulithot_list[j]=vocabulary_word2index_label[y]
            if len(ys)>seq2seq_label_length-1:
                ys_mulithot_list[seq2seq_label_length-1]=vocabulary_word2index_label[_END]#ADD END TOKEN
            else:
                ys_mulithot_list[len(ys)] = vocabulary_word2index_label[_END]

            # below is input for decoder.
            ys_decoder_input[0]=vocabulary_word2index_label[_GO]
            for j,y in enumerate(ys):
                if j < seq2seq_label_length - 1:
                    ys_decoder_input[j+1]=vocabulary_word2index_label[y]
            if i<10:
                print(i,"ys:==========>0", ys)
                print(i,"ys_mulithot_list:==============>1", ys_mulithot_list)
                print(i,"ys_decoder_input:==============>2", ys_decoder_input)
        else:
            if multi_label_flag: # 2)prepare multi-label format for classification
                ys = y.replace('\n', '').split(" ")  # ys is a list
                ys_index=[]
                for y in ys:
                    #y_index = vocabulary_word2index_label[y]
                    #ys_index.append(y_index)
                    if vocabulary_word2index_label.get(y,None) is not None:
                        y_index = vocabulary_word2index_label[y]
                        ys_index.append(y_index)
                if ys_index == []: # if it is empty, due to no labels have their frequency above the threshold label_freq_th.
                    continue
                # truncating the label by keep_label_percent    
                ys_index_missing = ys_index
                if i<2:
                    print('original ys_index:',ys_index_missing)    
                random.seed(1234)
                random.shuffle(ys_index_missing)
                ys_index_missing = ys_index_missing[:round(len(ys_index_missing)*keep_label_percent)]
                if i<2:
                    print('truncated ys_index by',str(keep_label_percent*100),'percent:',ys_index_missing)     
                    
                ys_mulithot_list=transform_multilabel_as_multihot(ys_index,len(vocabulary_word2index_label))
                ys_mulithot_list_missing=transform_multilabel_as_multihot(ys_index_missing,len(vocabulary_word2index_label))
            else:                #3)prepare single label format for classification
                ys_mulithot_list=vocabulary_word2index_label[y]
        if i<=3:
            print("ys_index:")
            #print(ys_index)
            print(i,"y:",ys," ;ys_mulithot_list:",ys_mulithot_list) #," ;ys_decoder_input:",ys_decoder_input)
        X.append(x)
        Y.append(ys_mulithot_list) #every element in Y is a multihot list, a 5770 dimensional vector.
        Y_missing.append(ys_mulithot_list_missing)
        if use_seq2seq:
            Y_decoder_input.append(ys_decoder_input) #decoder input
        #if i>50000:
        #    break
    # 4.split to train,test and valid data
    number_examples = len(X)
    print("number_examples_whole:",number_examples) #
    test_ = (X[int((1 - test_portion) * number_examples):], Y[int((1 - test_portion) * number_examples):]) # get test first
    test.append(test_)
    
    X_train_valid = X[:int((1 - test_portion) * number_examples)] # get the train+valid
    Y_train_valid = Y[:int((1 - test_portion) * number_examples)]
    Y_missing_train_valid = Y_missing[:int((1 - test_portion) * number_examples)]
    
    fold_percent=1./float(kfold)
    number_examples = len(X_train_valid) # update the number of examples by removing those in the testing set
    for k in range(kfold):
        valid_k = (X_train_valid[int(k * fold_percent * number_examples):int((k+1) * fold_percent * number_examples)], Y_train_valid[int(k * fold_percent * number_examples):int((k+1) * fold_percent * number_examples)]) # using Y in testing data
        valid.append(valid_k)
        train_k = (X_train_valid[:int(k * fold_percent * number_examples)]+X_train_valid[int((k+1) * fold_percent * number_examples):],Y_missing_train_valid[:int(k * fold_percent * number_examples)]+Y_missing_train_valid[int((k+1) * fold_percent * number_examples):]) # using Y_missing in training data
        train.append(train_k)
        #print("fold",str(k+1),"prepared")
        #print()
    #train = (X[0:int((1 - valid_portion) * number_examples)], Y[0:int((1 - valid_portion) * number_examples)]) # tuple of lists
    #test = (X[int((1 - valid_portion) * number_examples) + 1:], Y[int((1 - valid_portion) * number_examples) + 1:])
    # no validation here. test only. 0.05%.
    if use_seq2seq:
        train=train+(Y_decoder_input[0:int((1 - valid_portion) * number_examples)],)
        test=test+(Y_decoder_input[int((1 - valid_portion) * number_examples) + 1:],)
    # 5.return
    print("load_data.ended...")
    return train, valid, test

# need to change: okay with repeated ones.
def transform_multilabel_as_multihot(label_list,label_size=5196): #1999label_list=[0,1,4,9,5]
    """
    :param label_list: e.g.[0,1,4]
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result=np.zeros(label_size)
    #set those location as 1, all else place as 0.
    result[label_list] = 1
    return result