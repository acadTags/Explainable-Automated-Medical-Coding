#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
import sys

from HAN_model_dynamic import HAN

from data_util_gensim import load_data_multilabel_pre_split_for_pred,create_vocabulary,create_vocabulary_label_for_predict,get_label_sim_matrix,get_label_sub_matrix
from tflearn.data_utils import to_categorical, pad_sequences
from gensim.models import Word2Vec
import pickle
import random as rn
import statistics
from sklearn import metrics
from tqdm import tqdm

#for preprocessing of raw clinical notes (e.g. discharge summaries)
from nltk.tokenize import RegexpTokenizer
import spacy
from spacy.lang.en import English
import re 

# for visulisation of the final projection layer and labelwise attention layer with or without label embedding intialisation; and for attention visualisation
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pass

# final formatting function for demo
def formatting(list_of_doc_label_pair_desc,list_of_att_viz_htmls,mark='top50'):
    return '<p><b>%s</b></p>' % mark + '\n'.join(['\n' + '<p><b>%s</b></p>' % doc_label_pair_info + '\n' + att_viz_html for doc_label_pair_info, att_viz_html in zip(list_of_doc_label_pair_desc,list_of_att_viz_htmls)])
    #for doc_label_pair_info, att_viz_html in zip(list_of_doc_label_pair_desc,list_of_att_viz_htmls):
        # if att_viz_html != '':        
            # print(doc_label_pair_info, att_viz_html[:15] + att_viz_html[-15:]) 
            # # here just displayed the first and last 15 chars in html
        # else:
            # print('*it is time to summarise*:')
            # print(doc_label_pair_info)
        
# final processing function for demo 
'''input: doc_string, a raw document as string
          prediction mode, full code prediction as 'full' or top-50 code prediction as 'top50'
          verbose, whether to verbose the prediction steps and details
          display_viz, whether to display the attention visualisations here (in the jupyter notebook)
   output: a tuple of two lists
        (i)  list_of_doc_label_pair_descriptions (one for each attention visualisation)
        (ii) list_of_attention_viz_htmls (if empty string then *it is time to summarise*)
'''
def code_and_explain(doc_str,mode='top50',verbose=False,display_viz=True):
    #remove head and trailing spaces and newlines
    doc_str = doc_str.strip()
    #key model settings
    to_input=False
    if mode == 'top50':
        if '\n' in doc_str:
            ckpt_dir="../checkpoints/checkpoint_HAN_50_per_label_bs32_sent_split_LE/";dataset = "mimic3-ds-50";batch_size = 32;per_label_attention=True;per_label_sent_only=False;sent_split=True #HLAN trained on MIMIC-III-50 (sent split)
        else:
            ckpt_dir="../checkpoints/checkpoint_HAN_50_per_label_bs32_LE/";dataset = "mimic3-ds-50";batch_size = 32;per_label_attention=True;per_label_sent_only=False;sent_split=False #HLAN trained on MIMIC-III-50
    elif mode == 'full':
        if '\n' in doc_str:
            ckpt_dir="../checkpoints/checkpoint_HAN_sent_split_LE/";dataset = "mimic3-ds";batch_size = 128;per_label_attention=False;per_label_sent_only=False;sent_split=False #HAN trained on MIMIC-III (sent split)
        else:
            ckpt_dir="../checkpoints/checkpoint_HAN_LE/";dataset = "mimic3-ds";batch_size = 128;per_label_attention=False;per_label_sent_only=False;sent_split=True #HAN trained on MIMIC-III
    else:
        print('prediction mode unrecognisable: using top50')
        ckpt_dir="../checkpoints/checkpoint_HAN_50_per_label_bs32_sent_split_LE/";dataset = "mimic3-ds-50";batch_size = 32;per_label_attention=True;per_label_sent_only=False;sent_split=True #HLAN trained on MIMIC-III-50
        
    verbose=False # for a cleaner demo
    
    #other settings and hyper-parameters
    word2vec_model_path = "../embeddings/processed_full.w2v"
    emb_model_path = "../embeddings/word-mimic3-ds-label.model" #using the one learned from the full label sets of mimic-iii discharge summaries
    label_embedding_model_path = "../embeddings/code-emb-mimic3-tr-400.model" # for label embedding initialisation (W_projection)
    label_embedding_model_path_per_label = "../embeddings/code-emb-mimic3-tr-200.model" # for label embedding initialisation (per_label context_vectors)
    kb_icd9 = "../knowledge_bases/kb-icd-sub.csv"

    gpu=True
    learning_rate = 0.01
    decay_steps = 6000
    decay_rate = 1.0
    sequence_length = 2500
    num_sentences = 100
    embed_size=100
    hidden_size=100
    is_training=False
    lambda_sim=0.0
    lambda_sub=0.0
    dynamic_sem=True
    dynamic_sem_l2=False
    multi_label_flag=True
    pred_threshold=0.5
    use_random_sampling=False
    miu_factor=5
    
    #using gpu or not
    if not gpu: 
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
        
    #load the label list
    vocabulary_word2index_label,vocabulary_index2word_label = create_vocabulary_label_for_predict(name_scope=dataset + "-HAN") # keep a distinct name scope for each model and each dataset.
    if vocabulary_word2index_label == None:
        print('_label_vocabulary.pik file unavailable')
        sys.exit()

    #get the number of labels
    num_classes=len(vocabulary_word2index_label)

    #building the vocabulary list from the pre-trained word embeddings
    vocabulary_word2index, vocabulary_index2word = create_vocabulary(word2vec_model_path,name_scope=dataset + "-HAN")
    vocab_size = len(vocabulary_word2index)

    #preprocessing
    #transform the string of the document into a file and preprocess it.
    preprocess = True
    filename_to_predict = 'raw dis sum input.txt'
    output_to_file("../datasets/%s" % filename_to_predict,doc_str)
        
    testing_data_path = "../datasets/%s" % filename_to_predict
    
    if preprocess:
        #preprocess data (sentence parsing and tokenisation)
        #this is only used for a *raw* discharge summary (one each time) and when to_input is set as true.
        clinical_note_preprocessed_str = preprocessing(raw_clinical_note_file=testing_data_path,sent_parsing=sent_split,num_of_sen=100,num_of_sen_len=25) # tokenisation, padding, lower casing, sentence splitting
        output_to_file('clinical_note_temp.txt',clinical_note_preprocessed_str) #load the preprocessed data
        testX, testY = load_data_multilabel_pre_split_for_pred(vocabulary_word2index,vocabulary_word2index_label,data_path='clinical_note_temp.txt',verbose=verbose)
    else:
        #this allows processing many preprocessed documents together, each in a row of the file in the testing_data_path
        testX, testY = load_data_multilabel_pre_split_for_pred(vocabulary_word2index,vocabulary_word2index_label,data_path=testing_data_path,verbose=verbose)
    
    #padding to the maximum sequence length
    testX = pad_sequences(testX, maxlen=sequence_length, value=0.)  # padding to max length
    #print(len(testX),len(testY))
    
    #clean previous graph
    tf.reset_default_graph()
    
    #create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=False
    with tf.Session(config=config) as sess:
        #Instantiate Model
        model=HAN(num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,num_sentences,vocab_size,embed_size,hidden_size,is_training,lambda_sim,lambda_sub,dynamic_sem,dynamic_sem_l2,per_label_attention,per_label_sent_only,multi_label_flag=multi_label_flag, verbose=verbose)
        saver=tf.train.Saver(max_to_keep = 1) # only keep the latest model, here is the best model
        if os.path.exists(ckpt_dir+"checkpoint"):
            #print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop")
            sys.exit()

        #get prediction results and attention scores
        if per_label_attention: # to do for per_label_sent_only
            prediction_str = display_for_qualitative_evaluation_per_label(sess,model,testX,testY,batch_size,vocabulary_index2word,vocabulary_index2word_label,sequence_length,per_label_sent_only,num_sentences=num_sentences,threshold=pred_threshold,use_random_sampling=use_random_sampling,miu_factor=miu_factor) 
        else:
            prediction_str = display_for_qualitative_evaluation(sess,model,testX,testY,batch_size,vocabulary_index2word,vocabulary_index2word_label,sequence_length=sequence_length,num_sentences=num_sentences,threshold=pred_threshold,use_random_sampling=use_random_sampling,miu_factor=miu_factor)
    
    print('prediction_str:',prediction_str)
    #visualisation
    list_doc_label_marks,list_doc_att_viz,dict_doc_pred = viz_attention_scores_new(prediction_str)
    
    list_doc_label_pairs = []
    list_doc_att_viz_html = []
    if len(list_doc_att_viz) == 0: # if no ICD code assigned for the document.
        print('No ICD code predicted for this document.')    
    else:    
        for ind, (doc_label_mark, doc_att_viz) in enumerate(zip(list_doc_label_marks,list_doc_att_viz)):
            # retrieve and display ICD-9 codes and descriptions 
            doc_label_mark_ele_list = doc_label_mark.split('-')
            if len(doc_label_mark_ele_list)==2: # HAN model, same visualisation for all codes
                doc_label_mark_without_code = '-'.join(doc_label_mark_ele_list[:3])
                verbose_doc_code_info = doc_label_mark_without_code
                #print(doc_label_mark_without_code)
                #filename = 'att-%s.xlsx' % (doc_label_mark[:len(doc_label_mark)-1])     
                predictions = dict_doc_pred[doc_label_mark_without_code]
                predictions = predictions.split('labels:')[0]
                ICD_9_codes = predictions.split(' ')[1:]
                verbose_doc_code_info = verbose_doc_code_info + '\n' + 'Predicted code list:'
                #print('Predicted code list:')
                for ICD_9_code in ICD_9_codes:
                    # retrieve the short title and the long title of this ICD-9 code
                    _, long_tit,code_type = retrieve_icd_descs(ICD_9_code)
                    verbose_doc_code_info = verbose_doc_code_info + '\n' + code_type + ' code: ' + ICD_9_code + ' ( ' + long_tit + ' )'
                    #print(code_type,'code:',ICD_9_code,'(',long_tit,')')
                list_doc_label_pairs.append(verbose_doc_code_info)
            else: # HLAN or HA-GRU, a different visualisation for each label
                ICD_9_code = doc_label_mark_ele_list[3] # retrieve the predicted ICD-9 code
                ICD_9_code = ICD_9_code[:len(ICD_9_code)-1] # drop the trailing colon
                short_tit, long_tit,code_type = retrieve_icd_descs(ICD_9_code) # retrieve the short title and the long title of this ICD-9 code
                doc_label_mark_without_code = '-'.join(doc_label_mark_ele_list[:3])
                verbose_doc_code_info = doc_label_mark_without_code + ' to predict %s code ' % code_type + ICD_9_code + ' (%s)' % (long_tit)
                #print(doc_label_mark_without_code,'to predict %s code' % code_type,ICD_9_code,'(%s)' % (long_tit))
                list_doc_label_pairs.append(verbose_doc_code_info)
                
                #filename = 'att-%s(%s).xlsx' % (doc_label_mark[:len(doc_label_mark)-1],short_tit) #do not include the colon in the last char
                #filename = filename.replace('/','').replace('<','').replace('>','') # avoid slash / or <, > signs in the filename
            
            # display the visualisation
            #doc_att_viz.set_properties(**{'font-size': '9pt'})
            #list_doc_att_viz_html.append(doc_att_viz.render())
            list_doc_att_viz_html.append(doc_att_viz)
            #print(type(doc_att_viz.render()))
            if display_viz:
                print(verbose_doc_code_info)
                display(doc_att_viz)
            
            # export the visualisation to an Excel sheet
            # filename = '..\explanations\\' + filename # put the files under the ..\explanations\ folder.
            # # reset the font for the Excel sheet
            # doc_att_viz.set_properties(**{'font-size': '9pt'})\
                       # .to_excel(filename, engine='openpyxl')
            # print('Visualisation above saved to %s.' % filename) 
            
            # display the prediction when the label-wise visualisations for the document end
            if ind!=len(list_doc_label_marks)-1:
                #this is not the last doc label mark
                if list_doc_label_marks[ind+1][:len(doc_label_mark_without_code)] != doc_label_mark_without_code:                
                    #the next doc label mark is not the current one
                    verbose_doc_code_summary = dict_doc_pred[doc_label_mark_without_code]
                    #append the predition summary to the end
                    #list_doc_label_pairs.append(verbose_doc_code_summary)
                    #list_doc_att_viz_html.append('')
                    #insert the predition summary at the beginning
                    list_doc_label_pairs.insert(0,verbose_doc_code_summary)
                    list_doc_att_viz_html.insert(0,'')
                    if display_viz:
                        print(verbose_doc_code_summary)
                    #print('Visualisation for %s ended.\n' % doc_label_mark_without_code)
            else:
                #this is the last doc label mark
                verbose_doc_code_summary = dict_doc_pred[doc_label_mark_without_code]
                #append the predition summary to the end
                #list_doc_label_pairs.append(verbose_doc_code_summary)
                #list_doc_att_viz_html.append('')                
                #insert the predition summary at the beginning
                list_doc_label_pairs.insert(0,verbose_doc_code_summary)
                list_doc_att_viz_html.insert(0,'')
                if display_viz:
                    print(verbose_doc_code_summary)
                #print('Visualisation for %s ended.\n' % doc_label_mark_without_code)
    return list_doc_label_pairs, list_doc_att_viz_html

# for attention visualisation with matplotlib and html: code adapted from https://gist.github.com/ihsgnef/f13c35cd46624c8f458a4d23589ac768
def colorize(words, color_array):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    #print(words)
    #print(color_array)
    
    # colormaps: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html#lightness-of-matplotlib-colormaps
    cmap_sents = matplotlib.cm.get_cmap('Purples')
    cmap_words = matplotlib.cm.get_cmap('Reds')
    template = '<span class="barcode"; style="color: {}; background-color: {}">{}</span>'
    colored_string = ''
    for ind, (word, color) in enumerate(zip(words, color_array)):
        font_color = 'white' if color > 0.7 else 'black' #font color as white if too dark background
        if (color > 0.1 and ind == 0) or (color > 0.1 and ind > 0):
            # only to highlight those with sent att > 0.1 or word att > 0.1
            color = matplotlib.colors.rgb2hex(cmap_words(color)[:3]) if ind>0 else matplotlib.colors.rgb2hex(cmap_sents(color)[:3]) # sentence attention weights using cmap_sents, word att weights using cmap_words
        else:    
            color = 'white'
        colored_string += template.format(font_color, color, '&nbsp' + word + '&nbsp')
    return colored_string
    
# for attention visualisation with pandas dataframe and styler: code adapted from https://stackoverflow.com/a/53883859/5319143
class CharVal(object):
    def __init__(self, char, val):
        self.char = char
        self.val = val

    def __str__(self):
        return self.char
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
def color_charvals_with_sent_score(s):
    r = 255-int(s.val*255)        
    if s.char == '':
        #print('sent catched')
        color = rgb_to_hex((255, r, r))
    else:
        #print('word catched:',s.char)
        #color = rgb_to_hex((255, 255, r))
        color = rgb_to_hex((r,r,255))
    #print(s,color)
    return 'background-color: %s' % color
def deep_blue_background_word_whiten(s):
    """
    if a word or token having a too high word-level attention score in the attention heatmap, 
    turn the word into white color so that it can be easily seen in a deep blue background.
    """
    color = 'white' if s.val > 0.5 else 'black'
    return 'color: %s' % color    
    
def limit_width(s):
    if s.char == '':
        width = '3'
    else:
        width = '5'
    return 'width: %s' % width
def magnify():
    return [dict(selector="th",
                 props=[("font-size", "4pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]
    
#input the generated doc with attention scores (i.e. the output of display_for_qualitative_evaluation() below)
#output the visualisation of the doc with respect to each label
#the new visualisation is based on the function colorize(words, color_array)
def viz_attention_scores_new(prediction_str):
    #print('start viz_attention_scores')
    prediction_str = prediction_str.strip('\n')
    docs_att = prediction_str.split('\n\n')
    list_doc_label_marks=[]
    list_doc_att_viz=[]
    prediction=''
    dict_doc_pred = {} # a dictionary of document_label_mark without code i.e. doc-0-0 matching to values as a prediction string for each document 
    for doc_att in tqdm(docs_att):
        lines_att = doc_att.split('\n')
        if lines_att[0][:4]=='doc-':
            # a document, then we start the hierarchical attention visualisation
            #char_df = pd.DataFrame()
            doc_att_viz_html_str = ''
            for ind_line, line_att in enumerate(lines_att): # for every sentence
                line_att = line_att.strip()
                words_with_score = line_att.split(' ')
                if ind_line == 0:
                    #print(words_with_score[0]) # print the document-label prediction mark, e.g. doc-0-0-33.24
                    list_doc_label_marks.append(words_with_score[0]) # store the document-label prediction mark to the list
                    words_with_score = words_with_score[1:] # remove the document-label prediction mark 
                if len(words_with_score) > 1: # if the sentence is not empty, have at least a word besides the sentence mark.
                    #char_vals = []
                    list_words = []
                    list_att_score_float = []
                    for i, word_with_score in enumerate(words_with_score): #for every word
                        ind_left_p = word_with_score.find('(')
                        ind_right_p = word_with_score.find(')')
                        if ind_left_p>0 and ind_right_p>0:
                            if i != len(words_with_score) - 1: # if not the last word， which is a sentence mark
                                #print(word_with_score)
                                word = word_with_score[:ind_left_p]
                                word_att_score = float(word_with_score[ind_left_p+1:ind_right_p])
                                # create an CharVal object storing the word-attention score pair and append it to the list of CharVal objects
                                #char_vals = char_vals + [CharVal(word, word_att_score)]
                                list_words.append(word)
                                list_att_score_float.append(word_att_score)
                            else: # a sentence mark, e.g. '/s1(ori-0.0)/'
                                #the end of this sentence is reached
                                sent_att_score = float(word_with_score[ind_left_p+1+4:ind_right_p])
                                # append the sentence mark and sentence attention score pair to the list of CharVal objects
                                #char_vals = [CharVal('', sent_att_score)] + char_vals
                                # create a dataframe of CharVals for this whole sentence and append it to the dataframe of all previous sentences.
                                #char_df = char_df.append(pd.DataFrame(char_vals).transpose(),ignore_index=True).fillna(CharVal('', 0.0))
                                # about .fillna to address appending df with different columns: https://stackoverflow.com/a/43578742/5319143        
                                list_words = ['sent%s' % str(ind_line+1)] + list_words
                                list_att_score_float = [sent_att_score] + list_att_score_float
                        #else:
                        #    print(word_with_score, 'at', str(i), 'no parenthesis')
                    sent_html_str = colorize(list_words,np.array(list_att_score_float))
                    #print(list_words,list_att_score_float)
                    doc_att_viz_html_str = doc_att_viz_html_str + '\n' + sent_html_str                
            # apply coloring values
            #char_df = char_df.style.applymap(color_charvals_with_sent_score)\
            #                       .applymap(deep_blue_background_word_whiten)\
            #                       .set_properties(**{'max-width': '80px', 'font-size': '4pt'})\
            #                       .set_caption("Hover to magnify")\
            #                       .set_table_styles(magnify())
            #char_df = char_df.applymap(limit_width)
            #char_df = char_df.set_properties(**{'width': '30px'}) #limit the column width
            #list_doc_att_viz.append(char_df) # store the pandas styler doc att visulisation to the list
            #char_df#display(char_df)
            list_doc_att_viz.append(doc_att_viz_html_str)
        elif lines_att[0][:10]=='prediction':
            # the prediction
            prediction = doc_att
            #print(doc_att)
            if len(list_doc_label_marks) != 0:
                doc_label_mark_current = list_doc_label_marks[-1]
                doc_label_mark_without_code = '-'.join(doc_label_mark_current.split('-')[:3])
                #if the document's prediction was not stored in the dictionary: store it.
                if dict_doc_pred.get(doc_label_mark_without_code,None) == None:
                    dict_doc_pred[doc_label_mark_without_code] = doc_att
                #print(dict_doc_pred)
                
    return list_doc_label_marks,list_doc_att_viz,dict_doc_pred
    
#input the generated doc with attention scores (i.e. the output of display_for_qualitative_evaluation() below)
#output the visualisation of the doc with respect to each label
def viz_attention_scores(prediction_str):
    #print('start viz_attention_scores')
    prediction_str = prediction_str.strip('\n')
    docs_att = prediction_str.split('\n\n')
    list_doc_label_marks=[]
    list_doc_att_viz=[]
    prediction=''
    dict_doc_pred = {} # a dictionary of document_label_mark without code i.e. doc-0-0 matching to values as a prediction string for each document 
    for doc_att in tqdm(docs_att):
        lines_att = doc_att.split('\n')
        if lines_att[0][:4]=='doc-':
            # a document, then we start the hierarchical attention visualisation
            char_df = pd.DataFrame()
            for ind_line, line_att in enumerate(lines_att): # for every sentence
                line_att = line_att.strip()
                words_with_score = line_att.split(' ')
                if ind_line == 0:
                    #print(words_with_score[0]) # print the document-label prediction mark, e.g. doc-0-0-33.24
                    list_doc_label_marks.append(words_with_score[0]) # store the document-label prediction mark to the list
                    words_with_score = words_with_score[1:] # remove the document-label prediction mark 
                if len(words_with_score) > 1: # if the sentence is not empty, have at least a word besides the sentence mark.
                    char_vals = []
                    for i, word_with_score in enumerate(words_with_score): #for every word
                        ind_left_p = word_with_score.find('(')
                        ind_right_p = word_with_score.find(')')
                        if ind_left_p>0 and ind_right_p>0:
                            if i != len(words_with_score) - 1: # if not the last word， which is a sentence mark
                                #print(word_with_score)
                                word = word_with_score[:ind_left_p]
                                word_att_score = float(word_with_score[ind_left_p+1:ind_right_p])
                                # create an CharVal object storing the word-attention score pair and append it to the list of CharVal objects
                                char_vals = char_vals + [CharVal(word, word_att_score)]
                            else: # a sentence mark, e.g. '/s1(ori-0.0)/'
                                #the end of this sentence is reached
                                sent_att_score = float(word_with_score[ind_left_p+1+4:ind_right_p])
                                # append the sentence mark and sentence attention score pair to the list of CharVal objects
                                char_vals = [CharVal('', sent_att_score)] + char_vals
                                # create a dataframe of CharVals for this whole sentence and append it to the dataframe of all previous sentences.
                                char_df = char_df.append(pd.DataFrame(char_vals).transpose(),ignore_index=True).fillna(CharVal('', 0.0))
                                # about .fillna to address appending df with different columns: https://stackoverflow.com/a/43578742/5319143
                        #else:
                        #    print(word_with_score, 'at', str(i), 'no parenthesis')
            # apply coloring values
            char_df = char_df.style.applymap(color_charvals_with_sent_score)\
                                   .applymap(deep_blue_background_word_whiten)\
                                   .set_properties(**{'max-width': '80px', 'font-size': '4pt'})\
                                   .set_caption("Hover to magnify")\
                                   .set_table_styles(magnify())
            #char_df = char_df.applymap(limit_width)
            #char_df = char_df.set_properties(**{'width': '30px'}) #limit the column width
            list_doc_att_viz.append(char_df) # store the pandas styler doc att visulisation to the list
            #char_df#display(char_df)
        elif lines_att[0][:10]=='prediction':
            # the prediction
            prediction = doc_att
            #print(doc_att)
            if len(list_doc_label_marks) != 0:
                doc_label_mark_current = list_doc_label_marks[-1]
                doc_label_mark_without_code = '-'.join(doc_label_mark_current.split('-')[:3])
                #if the document's prediction was not stored in the dictionary: store it.
                if dict_doc_pred.get(doc_label_mark_without_code,None) == None:
                    dict_doc_pred[doc_label_mark_without_code] = doc_att
                #print(dict_doc_pred)
                
    return list_doc_label_marks,list_doc_att_viz,dict_doc_pred

# sentence split, tokenisation, padding (with 100*25)
# input: 
    #a raw clinical note (with labels specified at the end after __label__), 
    #whether to parse sentence, 
    #number of sentences to pad, 
    #number of tokens in each sentence
# output: a preprossed clinical note with its labelset if exists in the input
def preprocessing(raw_clinical_note_file,sent_parsing=True,num_of_sen=100,num_of_sen_len=25):

    with open(raw_clinical_note_file, 'r') as file:
        raw_clinical_note = file.read()
    
    #separate the label part if any
    raw_clinical_note_ele = raw_clinical_note.split('__label__')
    assert len(raw_clinical_note_ele) <= 2 # this value should be either 1 or 2 according to the input format
    if len(raw_clinical_note_ele) == 2: 
        raw_clinical_note_labelset = raw_clinical_note_ele[1]
    else:
        raw_clinical_note_labelset = ''
    raw_clinical_note_text = raw_clinical_note_ele[0]
    
    #set the tokenizer: retain only alphanumeric
    tokenizer = RegexpTokenizer(r'\w+') # original
        
    if sent_parsing:
        ##First: sentence tokenisation
        nlp = English()  # just the language with no model
        sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe(sentencizer) #rule-based sentencizer: .?!
        nlp.add_pipe(set_custom_boundaries) #add custom rules: \n\n
        #see https://spacy.io/usage/linguistic-features#sbd
        
        doc = nlp(raw_clinical_note_text)
        tokens = []
        for i,sent_tokens in enumerate(doc.sents):
            ##Second: tokenisation same as in the original CAML-MIMIC step for tokens in each sentence
            list_token_str = [t.lower() for t in tokenizer.tokenize(sent_tokens.text) if not t.isnumeric()]
            
            ##Third: add all the tokens in all sentences together with sentence sign as dot 
            if len(list_token_str) != 0:
                tokens = tokens + list_token_str + ['.'] # add tokens of sentences all together with sentence split sign as dot.
        clinical_note_tokenised = ' '.join(tokens)
        
        ##Forth: comine short sentences (length below 5)
        clinical_note_tokenised_combined = short_sentence_combined_with_previous_one(clinical_note_tokenised, length_threshold=10)
        
        ##Fifth: padding to 100 sentences and 25 tokens per sentence
        sentences = clinical_note_tokenised_combined.split(".")
        sen_n=len(sentences)
        padded_clinical_note = ""
        for i in range(num_of_sen):
            if i+1<=sen_n: # i starts from 0
                padded_clinical_note=padded_clinical_note.strip() + " " + pad(sentences[i],num_of_sen_len)
            else:
                padded_clinical_note=padded_clinical_note.strip() + " " + pad("",num_of_sen_len)
        #add the labelset if it exists in the input
        padded_clinical_note = padded_clinical_note + '__label__' + raw_clinical_note_labelset if raw_clinical_note_labelset != '' else padded_clinical_note
        return padded_clinical_note
    else:
        #directly tokenise each word in the document
        #tokenize, lowercase and remove numerics
        tokens = [t.lower() for t in tokenizer.tokenize(raw_clinical_note_text) if not t.isnumeric()]
        preprocessed_clinical_note = '"' + ' '.join(tokens) + '"'
        #add the labelset if it exists in the input
        preprocessed_clinical_note = preprocessed_clinical_note + '__label__' + raw_clinical_note_labelset if raw_clinical_note_labelset != '' else preprocessed_clinical_note
        return preprocessed_clinical_note
        
def set_custom_boundaries(doc):
    for token in doc[:-1]:
    	#print(repr(token.text))
    	if '\n\n' in token.text: # adding a custom rule here, if there is a consequtive 2 newline signs in the token, set it as a sentence boundary.
            doc[token.i+1].is_sent_start = True
    return doc

def short_sentence_combined_with_previous_one(text, length_threshold = 5):
    text_sent_combined = ''
    sentences = text.split('.')
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) >= length_threshold: 
            # if the current 'sentence' has token length above or equal to length_threshold, put an end to the previous 'sentence'
            if text_sent_combined != '':
                text_sent_combined = text_sent_combined + '. ' + sentence + ' '
            else:
                text_sent_combined = sentence + ' '
        else:
            # otherwise, combine the current 'sentence' with the previous 'sentence'
            text_sent_combined = text_sent_combined + sentence + ' '
    return text_sent_combined[:len(text_sent_combined)-1]
    
def pad(text,num_of_sen_len=25):
    padded=""
    words = text.strip().split(" ")
    for i in range(num_of_sen_len):
        #print(i)
        if words[0] != "" and i+1<=len(words):
            padded=padded + " " + words[i]
        else:
            padded=padded + " _PAD"
    return padded.strip()    

#retrieve descriptions of ICD-9 codes
#input: an ICD code, path to the diagnosis ICD code, path to the procedure ICD code
#output: short title, long title, code type
#if input is an empty string, then return empty strings for the short title, long title, and code type.
def retrieve_icd_descs(str_icd_code,path_icd_diag='../knowledge_bases/D_ICD_DIAGNOSES.csv',path_icd_proc='../knowledge_bases/D_ICD_PROCEDURES.csv'):
    if str_icd_code == '': 
        return '','',''
    code_type = check_code_type(str_icd_code)
    if code_type == 'diag':
        #print('diagnosis code:',str_icd_code)
        #remove dot in ICD-9 code
        str_icd_code = str_icd_code.replace('.','')
        #match to .csv file to query the short title and the long title
        map_icd_diag = pd.read_csv(path_icd_diag)
        short_title_tmp = map_icd_diag[map_icd_diag['ICD9_CODE'].astype(str)==str_icd_code]['SHORT_TITLE'].to_string(index=False)
        long_title_tmp = map_icd_diag[map_icd_diag['ICD9_CODE'].astype(str)==str_icd_code]['LONG_TITLE'].to_string(index=False)        
    elif code_type == 'proc':
        #print('procedure code:',str_icd_code)
        #remove dot in ICD-9 code
        str_icd_code = str_icd_code.replace('.','')
        #print(str_icd_code)
        #match to .csv file to query the short title and the long title
        map_icd_proc = pd.read_csv(path_icd_proc)
        #print(map_icd_proc)
        short_title_tmp = map_icd_proc[map_icd_proc['ICD9_CODE'].astype(str)==str_icd_code]['SHORT_TITLE'].to_string(index=False)
        long_title_tmp = map_icd_proc[map_icd_proc['ICD9_CODE'].astype(str)==str_icd_code]['LONG_TITLE'].to_string(index=False)
    else:
        return
    return short_title_tmp.strip(),long_title_tmp.strip(),code_type #remove the whitespace before each of the titles
        
# to check the type of code: diagnostic or procedural
def check_code_type(ICD9_code):
    try:
        pos = ICD9_code.index('.')
        if pos == 3 or (ICD9_code[0] == 'E' and pos == 4): # this is diagnostic code
            return 'diag'
        elif pos == 2: # this is procedural code
            return 'proc'
    except: # to catch the ValueError: substring not found from code.index('.')
        if len(ICD9_code) == 3 or (ICD9_code[0] == 'E' and len(ICD9_code) == 4): # still diagnostic code
            return 'diag'
            
#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    with open(file_name, 'w', encoding="utf-8-sig") as f_output:
        f_output.write(str)

def display_for_qualitative_evaluation(sess,modelToEval,evalX,evalY,batch_size,vocabulary_index2word,vocabulary_index2word_label,sequence_length,num_sentences,threshold=0.5,use_random_sampling=False, miu_factor=5):
#miu_factor: the factor to control the magnitude of sentence-weighted word-level attention weights
    prediction_str=""
    n_doc=0
    number_examples=len(evalX)
    
    #random sampling to get the displayed documents
    rn_dict={}
    rn.seed(1) # set the seed to produce same documents for prediction
    for i in range(0,500):
        batch_chosen=rn.randint(0,number_examples//batch_size)
        x_chosen=rn.randint(0,batch_size)
        rn_dict[(batch_chosen*batch_size,x_chosen)]=1
    
    #for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)): # a few samples in the evaluation set can be lost, due to the training and testing with the batch size (this may not be exactly divided).
    for start,end in zip(tqdm(list(range(0,number_examples,batch_size))),list(range(batch_size,number_examples,batch_size))+[number_examples]):
        feed_dict = {modelToEval.input_x: evalX[start:end], modelToEval.dropout_keep_prob: 1}
        #if (start==0):
        #    print(evalX[start:end])
        feed_dict[modelToEval.input_y_multilabel] = evalY[start:end]
        
        word_att,sent_att,logits= sess.run([modelToEval.p_attention,modelToEval.p_attention_sent,modelToEval.logits],feed_dict)
        word_att = np.reshape(word_att, (end-start,sequence_length))
        
        for x in range(0,len(logits)):
            label_list_th = get_label_using_logits_threshold(logits[x],threshold)
            #label_list_topk = get_label_using_logits(logits[x], vocabulary_index2word_label,top_number=11)
            # display a particular prediction result
            #if x==x_chosen and start==batch_chosen*batch_size:
            if rn_dict.get((start,x)) == 1 or (not use_random_sampling):
                # print('doc:',*display_results(evalX[start+x],vocabulary_index2word))
                # print('prediction-0.5:',*display_results(label_list_th,vocabulary_index2word_label))
                # #print('prediction-topk:',*display_results(label_list_topk,vocabulary_index2word_label))
                # get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
                # print('labels:',*display_results(get_indexes(1,evalY[start+x]),vocabulary_index2word_label))
                #doc = 'doc: ' + ' '.join(display_results(evalX[start+x],vocabulary_index2word))
                doc = 'doc-' + str(n_doc) + ': ' + ' '.join(display_results_with_word_att_sent_att(evalX[start+x],vocabulary_index2word,word_att[x],sent_att[x],'ori',sequence_length,num_sentences,miu_factor=miu_factor))
                pred = 'prediction-0.5: ' + ' '.join(display_results(label_list_th,vocabulary_index2word_label))
                #print('prediction-topk:',*display_results(label_list_topk,vocabulary_index2word_label))
                get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
                label = 'labels: ' + ' '.join(display_results(get_indexes(1,evalY[start+x]),vocabulary_index2word_label))
                prediction_str = prediction_str + '\n' + doc + '\n' + pred + '\n' + label + '\n'
                #print(prediction_str)
                n_doc=n_doc+1
    return prediction_str

def display_for_qualitative_evaluation_per_label(sess,modelToEval,evalX,evalY,batch_size,vocabulary_index2word,vocabulary_index2word_label,sequence_length,per_label_sent_only,num_sentences,threshold=0.5,use_random_sampling=False, miu_factor=5):
#miu_factor: the factor to control the magnitude of sentence-weighted word-level attention weights
    prediction_str=""
    n_doc=0
    number_examples=len(evalX)
    
    #random sampling to get the displayed documents
    rn_dict={}
    rn.seed(1) # set the seed to produce same documents for prediction
    for i in range(0,500):
        batch_chosen=rn.randint(0,number_examples//batch_size)
        x_chosen=rn.randint(0,batch_size)
        rn_dict[(batch_chosen*batch_size,x_chosen)]=1
            
    #for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)): # a few samples in the evaluation set can be lost, due to the training and testing with the batch size (this may not be exactly divided).
    for start,end in zip(tqdm(list(range(0,number_examples,batch_size))),list(range(batch_size,number_examples,batch_size))+[number_examples]):
        feed_dict = {modelToEval.input_x: evalX[start:end], modelToEval.dropout_keep_prob: 1}
        #if (start==0):
        #    print(evalX[start:end])
        feed_dict[modelToEval.input_y_multilabel] = evalY[start:end]
        
        word_att_per_label,sent_att_per_label,logits= sess.run([modelToEval.p_attention,modelToEval.p_attention_sent,modelToEval.logits],feed_dict)
        #print('word_att_per_label:',word_att_per_label.shape)
        num_classes = len(vocabulary_index2word_label)
        if not per_label_sent_only: # if also includes per-label word-level attention weights, there is a *dinstinct* word-level attention weight for each different label.
            #word_att_per_label: shape:[num_classes,batch_size*num_sentences,sequence_length_per_sentence]
            list_word_att_per_label = np.split(word_att_per_label,num_classes,axis=0) #print('list_word_att_per_label:',len(list_word_att_per_label),list_word_att_per_label[0].shape)
        else:
            #there is a *shared* word-level attention weight for any of the labels.
            #word_att_per_label: shape:[batch_size*num_sentences,sequence_length_per_sentence]
            word_att = np.reshape(word_att_per_label, (end-start,sequence_length))
        
        #sent_att_per_label: shape:# shape:[num_classes,batch_size,num_sentences]
        list_sent_att_per_label = np.split(sent_att_per_label,num_classes,axis=0)
        #print('list_sent_att_per_label:',len(list_sent_att_per_label),list_sent_att_per_label[0].shape)
        #for word_att in word_att_per_label:
        #    word_att = np.reshape(word_att, (batch_size,FLAGS.sequence_length))
        
        for x in range(0,len(logits)):
            label_list_th = get_label_using_logits_threshold(logits[x],threshold)
            #label_list_topk = get_label_using_logits(logits[x], vocabulary_index2word_label,top_number=11)
            # display a particular prediction result
            #if x==x_chosen and start==batch_chosen*batch_size:
            if rn_dict.get((start,x)) == 1 or (not use_random_sampling):
                # print('doc:',*display_results(evalX[start+x],vocabulary_index2word))
                # print('prediction-0.5:',*display_results(label_list_th,vocabulary_index2word_label))
                # #print('prediction-topk:',*display_results(label_list_topk,vocabulary_index2word_label))
                # get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
                # print('labels:',*display_results(get_indexes(1,evalY[start+x]),vocabulary_index2word_label))
                #doc = 'doc: ' + ' '.join(display_results(evalX[start+x],vocabulary_index2word))
                docs = ''
                for pred_label_index in label_list_th:
                    if not per_label_sent_only: # if also includes per-label word-level attention weights
                        word_att = list_word_att_per_label[pred_label_index]
                        word_att = np.reshape(word_att, (end-start,sequence_length))
                    sent_att = list_sent_att_per_label[pred_label_index]
                    sent_att = np.reshape(sent_att, (end-start,num_sentences))
                    #print('word_att:',word_att)
                    #print('sent_att:',sent_att)
                    label_to_explain = vocabulary_index2word_label[pred_label_index]
                    docs = docs + 'doc-' + str(n_doc) + '-' + str(start+x) + '-' + label_to_explain + ': ' + ' '.join(display_results_with_word_att_sent_att(evalX[start+x],vocabulary_index2word,word_att[x],sent_att[x],'ori',sequence_length,num_sentences,miu_factor=miu_factor)) + '\n'
                    #formatting: "doc - the nth doc to be presented - the kth doc in the who testing set - the label to explain"
                    
                    #todo
                    #top-3 sentences: score
                    #top-3 tokens (weighted by sentences): score
                    
                pred = 'prediction-0.5: ' + ' '.join(display_results(label_list_th,vocabulary_index2word_label))
                #print('prediction-topk:',*display_results(label_list_topk,vocabulary_index2word_label))
                get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
                label = 'labels: ' + ' '.join(display_results(get_indexes(1,evalY[start+x]),vocabulary_index2word_label))
                #print(label)
                prediction_str = prediction_str + '\n' + docs + pred + '\n' + label + '\n'
                n_doc = n_doc + 1
                #print(prediction_str)
    return prediction_str
    
# display results with word-level attention weights and sentence-level attention weights
# this can be used for both display a sequence of words (with vocabulary_index2word) or a sequence of labels (with vocabulary_index2word_label, as below).
def display_results_with_word_att_sent_att(index_list,vocabulary_index2word_label,word_att,sent_att,att_note,sequence_length,num_sentences,miu_factor=5):
    label_list=[]
    count = 1                   
    for index in index_list:
        if index!=0: # this ensures that the padded values not being displayed.
            label=vocabulary_index2word_label[index]
            #label_list.append(label)
            #if word_att != '': #FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
            if word_att.size != 0:    
                if miu_factor != -1:
                    #update word att scores to a sentence-weighted version
                    sent_index = int((count-1) / (sequence_length/num_sentences))
                    word_att_sent_weighted = miu_factor*word_att[count-1]*sent_att[sent_index]
                    word_att_sent_weighted = min(word_att_sent_weighted,1.0) # cap to 1 if above 1.
                    label_list.append(label + '(' + str(round(word_att_sent_weighted,3)) + ')')
                else: # do not weight word att scores by sent att scores
                    label_list.append(label + '(' + str(round(word_att[count-1],3)) + ')')                
            else:
                print('word_att as empty:',word_att)
                label_list.append(label)
        if count % (sequence_length/num_sentences) == 0: # when it arrives to an end of a sentence
            sent_index = int(count / (sequence_length/num_sentences))
            #if sent_att != '':
            if sent_att.size != 0:
                label_list.append('/s' + str(int(sent_index)) + '(' + att_note + '-' + str(round(sent_att[sent_index-1],2)) + ')/' + '\n')
            else:
                label_list.append('/s' + str(int(sent_index)) + '\n')
        count = count + 1
    
    return label_list
                    
#input: vector, embedding vectors; list_label_str; dicts; k
#output a list of list of topk labels (for each query label in the list).
def get_k_similar_label(vectors,list_label_str,vocabulary_index2word_label,vocabulary_word2index_label,k):
    list_query_ind = [vocabulary_word2index_label[label_str] for label_str in list_label_str]
    #sklearn.metrics.pairwise.cosine_similarity(X, Y=None, dense_output=True)
    vec_sim_square_mat = pairwise.cosine_similarity(vectors)
    list_of_list_topk_labels = []
    for query_ind in list_query_ind:
        topk_label_inds = get_k_similar_vec(vec_sim_square_mat,query_ind,k)
        list_topk_labels = []
        for label_ind in topk_label_inds:
            list_topk_labels.append(vocabulary_index2word_label[label_ind])
        list_of_list_topk_labels.append(list_topk_labels)    
    return list_of_list_topk_labels
    
def get_k_similar_vec(vec_sim_square_mat,query_ind,k):
    #vec_sim_square_mat = pairwise.cosine_similarity(vectors)
    vec_to_query = vec_sim_square_mat[query_ind]
    return vec_to_query.argsort()[-k:][::-1]     
    
def show_per_label_results(vocabulary_index2word_label,prec_per_label,rec_per_label,f1_per_label):
    per_label_results = ''
    num_classes = len(vocabulary_index2word_label)
    for i in range(num_classes):
        if i==0:
            per_label_results = vocabulary_index2word_label[i] + ' prec: %.3f rec: %.3f f1: %.3f' % (prec_per_label[i],rec_per_label[i],f1_per_label[i])
        else:   
            per_label_results = per_label_results + '\n' + vocabulary_index2word_label[i] + ' prec: %.3f rec: %.3f f1: %.3f' % (prec_per_label[i],rec_per_label[i],f1_per_label[i])
    return per_label_results
    
#从logits中取出前五 get label using logits
def get_label_using_logits(logits,vocabulary_index2word_label,top_number=1):
    #print("get_label_using_logits.logits:",logits) #1-d array: array([-5.69036102, -8.54903221, -5.63954401, ..., -5.83969498,-5.84496021, -6.13911009], dtype=float32))
    index_list=np.argsort(logits)[-top_number:]
    index_list=index_list[::-1]
    #label_list=[]
    #for index in index_list:
    #    label=vocabulary_index2word_label[index]
    #    label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
    return index_list

#从sigmoid(logits)中取出大于0.5的
def get_label_using_logits_threshold(logits,threshold=0.5):
    sig = sigmoid_array(logits)
    index_list = np.where(sig > threshold)[0]
    return index_list

#turning the logit matrix into topk binary predictions
def get_topk_binary_using_logits_matrix(logits_matrix,top_number=1):
    logits_matrix_ori = np.copy(logits_matrix) #copy the logit matrix
    logits_matrix.sort(axis=1) #sort the logit matrix
    klargest = np.expand_dims(logits_matrix[:,-top_number],axis=1) #get the kth largest element per row in the logit matrix and format it to the row dimension same as the logits_matrix and column dimension as 1.
    logits_binary_topk = (logits_matrix_ori >= klargest).astype(float) # broadcasting comparison by row to get the top-k binary predictions
    return logits_binary_topk
    
def display_results(index_list,vocabulary_index2word_label, for_label=True):
    label_list=[]
    for index in index_list:
        if for_label or index!=0: # if to display words, then index as 0 ('pad' sign) is omitted; if to display label, then keep index as 0.
            label=vocabulary_index2word_label[index]
            label_list.append(label)
    return label_list

def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))    

#print(retrieve_icd_descs('401.9'))    
#print(retrieve_icd_descs('96.6'))
#print(retrieve_icd_descs(''))

