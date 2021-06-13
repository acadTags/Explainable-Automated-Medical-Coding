# -*- coding: utf-8 -*-
#code to train, validate, and test the model, with all the evaluation functions applied.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
import sys
from HAN_model_dynamic import HAN

#from data_util import load_data_multilabel_new,load_data_multilabel_new_k_fold,create_vocabulary,create_vocabulary_label,get_label_sim_matrix,get_label_sub_matrix
from data_util_gensim import load_data_multilabel_pre_split, load_data_multilabel_new,load_data_multilabel_new_k_fold,create_vocabulary,create_vocabulary_label,create_vocabulary_label_pre_split,get_label_sim_matrix,get_label_sub_matrix
from tflearn.data_utils import to_categorical, pad_sequences
#import word2vec
from gensim.models import Word2Vec
import pickle
import random as rn
import statistics
from sklearn import metrics
from tqdm import tqdm

# for visulisation of the final projection layer and labelwise attention layer with or without label embedding intialisation
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt

#tf.reset_default_graph()

# Setting the seed for numpy-generated random numbers
#np.random.seed(1)

# Setting the seed for python random numbers
#rn.seed(1)

# Setting the graph-level random seed.
#tf.set_random_seed(1)

#start time
start_time = time.time()
#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("dataset","mimic3-ds-50","dataset to chose")

tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_string("ckpt_dir","checkpoint_HAN/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",300,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",100,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",100,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_integer("validate_step", 1000, "how many step to validate.") #this validation is also for decaying the learning rate based on the evaluation loss
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")

#for semantic-based loss regularisers (not used)
tf.app.flags.DEFINE_float("label_sim_threshold",0,"similarity value below this threshold will be set as 0.")
tf.app.flags.DEFINE_float("lambda_sim",0,"the lambda for sim-loss.")
tf.app.flags.DEFINE_float("lambda_sub",0,"the lambda for sub-loss.")
#tf.app.flags.DEFINE_string("cache_path","text_cnn_checkpoint/data_cache.pik","checkpoint location for the model")
#train-zhihu4-only-title-all.txt
tf.app.flags.DEFINE_boolean("dynamic_sem",True,"whether to finetune the sim and sub matrices during training, default as True.")
tf.app.flags.DEFINE_boolean("dynamic_sem_l2",False,"whether to L2 regularise while finetuning the sim and sub matrices during training.")

#label embedding initialisation 
tf.app.flags.DEFINE_boolean("use_label_embedding",False,"whether to initialise label embedding.")
tf.app.flags.DEFINE_boolean("visualise_labelwise_layers",False,"whether to visualise *labelwise* layers, and show differences between label embedding (le) initialisation and not using le, including the final projection layer and the labelwise attention context matrices.")

#prediction threshold
tf.app.flags.DEFINE_float("pred_threshold",0.5,"prediction threshold to turn logit into binary.")

#per-label-attention
tf.app.flags.DEFINE_boolean("per_label_attention",False,"whether to use per-label attention mechanisms.")
tf.app.flags.DEFINE_boolean("per_label_sent_only",False,"whether to only use the per-label sentence-level attention as in HA-GRU, if per_label_attention is used.") #setting per_label_sent_only as True and per_label_attention as True is generally equivalant to HA-GRU, see T. Baumel, J. Nassour-Kassis, R. Cohen, M. Elhadad, and N. Elhadad, ‘Multi-label classification of patient notes: case study on ICD code assignment’, 2018.

#for simulating missing labels
tf.app.flags.DEFINE_float("keep_label_percent",1,"the percentage of labels in each instance of the training data to be randomly reserved, the rest labels are dropped to simulate the missing label scenario.")

#validaition and testing with holdout and k-fold cross-validation
tf.app.flags.DEFINE_float("valid_portion",0.1,"dev set or test set portion") # this is only valid when kfold is -1, which means we hold out a fixed set for validation. If we set this as 0.1, then there will be 0.81 0.09 0.1 for train-valid-test split (same as the split of 10-fold cross-validation); if we set this as 0.111, then there will be 0.8 0.1 0.1 for train-valid-test split.
tf.app.flags.DEFINE_float("test_portion",0.1,"held-out evaluation: test set portion")
tf.app.flags.DEFINE_integer("kfold",10,"k-fold cross-validation") # if k is -1, then not using kfold cross-validation; if k is 0, then using the pre-defined data split (for mimic3, split must be provided, see below).
tf.app.flags.DEFINE_integer("running_times",1,"running the model for a number of times to get averaged results. This is only applied if using pre-defined data split (kfold=0)")
tf.app.flags.DEFINE_boolean("remove_ckpts_before_train",False,"whether to remove checkpoints before training each fold or each running time.") #default False, but need to manually set it as true for training of several folds (kfold > 0) or runs (running_times > 0).

#training, validation and testing with pre-split datasets
tf.app.flags.DEFINE_boolean("use_sent_split_padded_version",False,"whether to use the sentenece splitted and padded version [MIMIC-III-related datasets only].") #whether to use the sentenece splitted and padded version. If set as true, HAN will deal with *real* sentences instead of fixed-length text chunks. This is used for MIMIC-III-related datasets. The sentence splitting was using a rule-based algorithm in the Spacy package with adding a double newline rule '\n\n' as another sentence boundary. The number of tokens in each sentence was padded to 25, and the number of sentences was padded to 100.

tf.app.flags.DEFINE_string("training_data_path_mimic3_ds","../datasets/mimiciii_train_th0.txt","path of training data.") # for mimic3-ds dataset
tf.app.flags.DEFINE_string("validation_data_path_mimic3_ds","../datasets/mimiciii_dev_th0.txt","path of validation/dev data.") # for mimic3-ds dataset
tf.app.flags.DEFINE_string("testing_data_path_mimic3_ds","../datasets/mimiciii_test_th0.txt","path of testing data.") # for mimic3-ds dataset

tf.app.flags.DEFINE_string("training_data_path_mimic3_ds_50","../datasets/mimiciii_train_50_th0.txt","path of training data.") # for mimic3-ds-50 dataset
tf.app.flags.DEFINE_string("validation_data_path_mimic3_ds_50","../datasets/mimiciii_dev_50_th0.txt","path of validation/dev data.") # for mimic3-ds-50 dataset
tf.app.flags.DEFINE_string("testing_data_path_mimic3_ds_50","../datasets/mimiciii_test_50_th0.txt","path of testing data.") # for mimic3-ds-50 dataset

#freq th as 50 (20 labels)
tf.app.flags.DEFINE_string("training_data_path_mimic3_ds_shielding_th50","../datasets/mimiciii_train_full_th_50_covid_shielding.txt","path of training data.") # for mimic3-ds-shielding-50 dataset
tf.app.flags.DEFINE_string("validation_data_path_mimic3_ds_shielding_th50","../datasets/mimiciii_dev_full_th_50_covid_shielding.txt","path of validation/dev data.") # for mimic3-ds-shielding-50 dataset
tf.app.flags.DEFINE_string("testing_data_path_mimic3_ds_shielding_th50","../datasets/mimiciii_test_full_th_50_covid_shielding.txt","path of testing data.") # for mimic3-ds-shielding-50 dataset

tf.app.flags.DEFINE_string("marking_id","","a marking_id (or group_id) for better marking: will show in the output filenames")

tf.app.flags.DEFINE_string("word2vec_model_path_mimic3_ds","../embeddings/processed_full.w2v","gensim word2vec's vocabulary and vectors") #for both mimic-iii and mimic-iii-50
tf.app.flags.DEFINE_string("word2vec_model_path_mimic3_ds_50","../embeddings/processed_full.w2v","gensim word2vec's vocabulary and vectors") #for mimic-iii-50

tf.app.flags.DEFINE_string("emb_model_path_mimic3_ds","../embeddings/word-mimic3-ds-label.model","pre-trained model from mimic3-ds labels")
# label emebedding for initialisation, also see def instantiate_weights(self) in HAN_model_dynamic.py
tf.app.flags.DEFINE_string("emb_model_path_mimic3_ds_init","../embeddings/code-emb-mimic3-tr-400.model","pre-trained model from mimic3-ds labels for label embedding initialisation: final projection matrix, self.W_projection.")
tf.app.flags.DEFINE_string("emb_model_path_mimic3_ds_init_per_label","../embeddings/code-emb-mimic3-tr-200.model","pre-trained model from mimic3-ds labels for label embedding initialisation: per label context matrices, self.context_vector_word_per_label and self.context_vector_sentence_per_label") # per_label means the per-label Context_vectors.

tf.app.flags.DEFINE_string("kb_icd9","../knowledge_bases/kb-icd-sub.csv","label relations from icd9, for mimic3") # for zhihu dataset

tf.app.flags.DEFINE_boolean("multi_label_flag",True,"use multi label or single label.")
tf.app.flags.DEFINE_integer("num_sentences", 10, "number of sentences in the document")
tf.app.flags.DEFINE_integer("hidden_size",100,"hidden size") # same as embedding size
tf.app.flags.DEFINE_boolean("weight_decay_testing",True,"weight decay based on validation data.") # decay the weight by half if validation loss increases.
tf.app.flags.DEFINE_boolean("report_rand_pred",True,"report prediction for qualitative analysis")
tf.app.flags.DEFINE_boolean("use_random_sampling",False,"whether to use a random sampling to show results.") # default false, if true then will show the sampled 500 examples for display
tf.app.flags.DEFINE_float("early_stop_lr",0.00002,"early stop point when learning rate is belwo is threshold") #0.00002
tf.app.flags.DEFINE_float("ave_labels_per_doc",11.59,"average labels per document for bibsonomy dataset")
tf.app.flags.DEFINE_integer("topk",5,"using top-k predicted labels for evaluation")

#output logits or not
tf.app.flags.DEFINE_boolean("output_logits",True,"output testing logit files for each run")

tf.app.flags.DEFINE_boolean("gpu",True,"use gpu")

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    #1.load data(X:list of lint,y:int).    
    if not FLAGS.gpu:
        #not using gpu
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
        
    # assign data specific variables - this part needs to be customised when you apply your own data, simply create a new if/elif section and assign data path and values for all the variables:
    #add your customised dataset
    '''
    if FLAGS.dataset == "YOUR_DATASET_NAME":
        #set values for 
        word2vec_model_path (Gensim embedding path)
        training_data_path
        validation_data_path (can be an empty path)
        testing_data_path (can be an empty path)
       
        #the embeddings below need to be pretrained (with Gensim) and paths to be added here
        label_embedding_model_path # for label embedding initialisation (W_projection)
        label_embedding_model_path_per_label  # for label embedding initialisation (per_label context_vectors)
        
        #if having pre-split set for evaluation
        vocabulary_word2index_label,vocabulary_index2word_label = create_vocabulary_label_pre_split(training_data_path=training_data_path, validation_data_path=validation_data_path, testing_data_path=testing_data_path, name_scope=FLAGS.dataset + "-HAN") # keep a distinct name scope for each model and each dataset.
        #OR, if using k-fold and/or held-out evaluation
        vocabulary_word2index_label,vocabulary_index2word_label = create_vocabulary_label(training_data_path=training_data_path, name_scope=FLAGS.dataset + "-HAN") # keep a distinct name scope for each model and each dataset.
        
        #configurations:
        #FLAGS.batch_size = 128
        FLAGS.sequence_length
        FLAGS.num_sentences
        FLAGS.ave_labels_per_doc #please pre-calculate this: this only affects the hamming_loss metric
        FLAGS.topk # for precision@k, recall@k, f1@k metrics
        FLAGS.kfold #fold for cross-validation, if 0 then using pre-defined data split, if -1 then using held-out validation
        
        #for semantic-based loss regularisers - in paper Dong et al, 2020, https://core.ac.uk/reader/327124320
        #FLAGS.lambda_sim = 0 # lambda1 - default as 0, i.e. not using the L_sim regulariser
        #FLAGS.lambda_sub = 0 # lambda2 - default as 0, i.e. not using the L_sub regulariser
        #keep the lines below unchanged
        #similarity relations: using self-trained label embedding - here Gensim pretrained label embedding path need to be added to the second argument (doesn't matter if FLAGS.lambda_sim is 0)               
        label_sim_mat = get_label_sim_matrix(vocabulary_index2word_label,FLAGS.emb_model_path_mimic3_ds,name_scope=FLAGS.dataset,random_init=FLAGS.lambda_sim==0)
        #subsumption relations: using external knowledge bases - here needs the subsumption relation of labels in a .csv file and its path to be added to the kb_path argument (doesn't matter if FLAGS.lambda_sub is 0)               
        label_sub_mat = get_label_sub_matrix(vocabulary_word2index_label,kb_path=FLAGS.kb_icd9,name_scope='icd9',zero_init=FLAGS.lambda_sim==0);print('using icd9 relations')
 
    #then change the if below to elif.
    '''
    if FLAGS.dataset == "mimic3-ds": # MIMIC-III full codes - # change to elif if you add another dataset option above
        word2vec_model_path = FLAGS.word2vec_model_path_mimic3_ds
        #choose the one based on the FLAGS.use_sent_split_padded_version option
        training_data_path = FLAGS.training_data_path_mimic3_ds.replace('_th0','_full_sent_split_th0_for_HAN') if FLAGS.use_sent_split_padded_version else FLAGS.training_data_path_mimic3_ds
        validation_data_path = FLAGS.validation_data_path_mimic3_ds.replace('_th0','_full_sent_split_th0_for_HAN') if FLAGS.use_sent_split_padded_version else FLAGS.validation_data_path_mimic3_ds
        testing_data_path = FLAGS.testing_data_path_mimic3_ds.replace('_th0','_full_sent_split_th0_for_HAN') if FLAGS.use_sent_split_padded_version else FLAGS.testing_data_path_mimic3_ds
        emb_model_path = FLAGS.emb_model_path_mimic3_ds
        label_embedding_model_path = FLAGS.emb_model_path_mimic3_ds_init # for label embedding initialisation (W_projection)
        label_embedding_model_path_per_label = FLAGS.emb_model_path_mimic3_ds_init_per_label # for label embedding initialisation (per_label context_vectors)
        
        #using all the train, validation, and test data to build the label list
        vocabulary_word2index_label,vocabulary_index2word_label = create_vocabulary_label_pre_split(training_data_path=training_data_path, validation_data_path=validation_data_path, testing_data_path=testing_data_path, name_scope=FLAGS.dataset + "-HAN") # keep a distinct name scope for each model and each dataset.
        
        #similarity relations: using self-trained label embedding
        label_sim_mat = get_label_sim_matrix(vocabulary_index2word_label,emb_model_path,name_scope=FLAGS.dataset,random_init=FLAGS.lambda_sim==0)
        
        #subsumption relations: using external knowledge bases
        label_sub_mat = get_label_sub_matrix(vocabulary_word2index_label,kb_path=FLAGS.kb_icd9,name_scope='icd9',zero_init=FLAGS.lambda_sub==0);print('using icd9 relations')
        
        #configurations:
        #FLAGS.batch_size = 128
        FLAGS.sequence_length = 2500 #2500 as in Mullenbach et al., 2018, but can be more if the memory allows
        FLAGS.num_sentences = 100 #length of sentence 25
        FLAGS.ave_labels_per_doc = 15.88 #actually 15.88
        #FLAGS.lambda_sim = 0 # lambda1
        #FLAGS.lambda_sub = 0 # lambda2
        FLAGS.topk = 8 # consistent to Mullenbach et al., 2018
        FLAGS.kfold = 0 #using pre-defined data split
    
    elif FLAGS.dataset == "mimic3-ds-50": # MIMIC-III top 50 codes
        #word2vec_model_path = FLAGS.word2vec_model_path_mimic3_ds #using the one learned from the full mimic-iii discharge summaries
        word2vec_model_path = FLAGS.word2vec_model_path_mimic3_ds_50
        #choose the one based on the FLAGS.use_sent_split_padded_version option
        training_data_path = FLAGS.training_data_path_mimic3_ds_50.replace('_th0','_sent_split_th0_for_HAN') if FLAGS.use_sent_split_padded_version else FLAGS.training_data_path_mimic3_ds_50
        print('path selected:',training_data_path, FLAGS.use_sent_split_padded_version)
        validation_data_path = FLAGS.validation_data_path_mimic3_ds_50.replace('_th0','_sent_split_th0_for_HAN') if FLAGS.use_sent_split_padded_version else FLAGS.validation_data_path_mimic3_ds_50
        testing_data_path = FLAGS.testing_data_path_mimic3_ds_50.replace('_th0','_sent_split_th0_for_HAN') if FLAGS.use_sent_split_padded_version else FLAGS.testing_data_path_mimic3_ds_50
        emb_model_path = FLAGS.emb_model_path_mimic3_ds #using the one learned from the full label sets of mimic-iii discharge summaries
        label_embedding_model_path = FLAGS.emb_model_path_mimic3_ds_init # for label embedding initialisation (W_projection)
        label_embedding_model_path_per_label = FLAGS.emb_model_path_mimic3_ds_init_per_label # for label embedding initialisation (per_label context_vectors)
        
        #using all the train, validation, and test data to build the label list
        vocabulary_word2index_label,vocabulary_index2word_label = create_vocabulary_label_pre_split(training_data_path=training_data_path, validation_data_path=validation_data_path, testing_data_path=testing_data_path, name_scope=FLAGS.dataset + "-HAN") # keep a distinct name scope for each model and each dataset.
        
        #similarity relations: using self-trained label embedding
        label_sim_mat = get_label_sim_matrix(vocabulary_index2word_label,emb_model_path,name_scope=FLAGS.dataset,random_init=FLAGS.lambda_sim==0)
        
        #subsumption relations: using external knowledge bases
        label_sub_mat = get_label_sub_matrix(vocabulary_word2index_label,kb_path=FLAGS.kb_icd9,name_scope='icd9-50',zero_init=FLAGS.lambda_sub==0);print('using icd9 relations')
        
        #configurations:
        #FLAGS.batch_size = 128
        FLAGS.sequence_length = 2500
        FLAGS.num_sentences = 100 #length of sentence 25 or 30
        FLAGS.ave_labels_per_doc = 5.69
        #FLAGS.lambda_sim = 0 # lambda1
        #FLAGS.lambda_sub = 0 # lambda2
        FLAGS.topk = 5 # consistent to Mullenbach et al., 2018
        FLAGS.kfold = 0 #using pre-defined data split
    
    elif FLAGS.dataset == "mimic3-ds-shielding-th50": # MIMIC-III shielding code, 20 codes (freq above 50)
        word2vec_model_path = FLAGS.word2vec_model_path_mimic3_ds #using the one learned from the full mimic-iii discharge summaries
        #choose the one based on the FLAGS.use_sent_split_padded_version option
        training_data_path = FLAGS.training_data_path_mimic3_ds_shielding_th50.replace('_th_50_covid_shielding','_sent_split_th_50_covid_shielding_for_HAN') if FLAGS.use_sent_split_padded_version else FLAGS.training_data_path_mimic3_ds_shielding_th50
        validation_data_path = FLAGS.validation_data_path_mimic3_ds_shielding_th50.replace('_th_50_covid_shielding','_sent_split_th_50_covid_shielding_for_HAN') if FLAGS.use_sent_split_padded_version else FLAGS.validation_data_path_mimic3_ds_shielding_th50
        testing_data_path = FLAGS.testing_data_path_mimic3_ds_shielding_th50.replace('_th_50_covid_shielding','_sent_split_th_50_covid_shielding_for_HAN') if FLAGS.use_sent_split_padded_version else FLAGS.testing_data_path_mimic3_ds_shielding_th50
        emb_model_path = FLAGS.emb_model_path_mimic3_ds #using the one learned from the full label sets of mimic-iii discharge summaries
        label_embedding_model_path = FLAGS.emb_model_path_mimic3_ds_init # for label embedding initialisation (W_projection)
        label_embedding_model_path_per_label = FLAGS.emb_model_path_mimic3_ds_init_per_label # for label embedding initialisation (per_label context_vectors)
        
        #using all the train, validation, and test data to build the label list
        vocabulary_word2index_label,vocabulary_index2word_label = create_vocabulary_label_pre_split(training_data_path=training_data_path, validation_data_path=validation_data_path, testing_data_path=testing_data_path, name_scope=FLAGS.dataset + "-HAN") # keep a distinct name scope for each model and each dataset.
        
        #similarity relations: using self-trained label embedding
        label_sim_mat = get_label_sim_matrix(vocabulary_index2word_label,emb_model_path,name_scope=FLAGS.dataset,random_init=FLAGS.lambda_sim==0)
        
        #subsumption relations: using external knowledge bases
        label_sub_mat = get_label_sub_matrix(vocabulary_word2index_label,kb_path=FLAGS.kb_icd9,name_scope='icd9-shielding-th50',zero_init=FLAGS.lambda_sub==0);print('using icd9 relations')
        
        #configurations:
        #FLAGS.batch_size = 128
        FLAGS.sequence_length = 2500
        FLAGS.num_sentences = 100 #length of sentence 25
        FLAGS.ave_labels_per_doc = 1.08
        #FLAGS.lambda_sim = 0 # lambda1
        #FLAGS.lambda_sub = 0 # lambda2
        FLAGS.topk = 1
        FLAGS.kfold = 0 #using pre-defined data split
        
    else:
        print("dataset unrecognisable")
        sys.exit()
    
    # create common filename prefix for the outputs
    #filename_common_prefix = 'l2 ' + str(FLAGS.lambda_sim) + " l3 " + str(FLAGS.lambda_sub) + ' th' + str(FLAGS.label_sim_threshold) + ' keep_label_percent' + str(FLAGS.keep_label_percent) + ' kfold' + str(FLAGS.kfold) + ' b_s' + str(FLAGS.batch_size) + ' gp_id' + str(FLAGS.marking_id)
    filename_common_prefix = 'l2 ' + str(FLAGS.lambda_sim) + " l3 " + str(FLAGS.lambda_sub) + ' b_s' + str(FLAGS.batch_size) + ' pred_th' + str(FLAGS.pred_threshold) + ' gp_id' + str(FLAGS.marking_id)
    
    num_classes=len(vocabulary_word2index_label)
    print(vocabulary_index2word_label[0],vocabulary_index2word_label[1])
    trainX, trainY, testX, testY = None, None, None, None
    #building the vocabulary list from the pre-trained word embeddings
    vocabulary_word2index, vocabulary_index2word = create_vocabulary(word2vec_model_path,name_scope=FLAGS.dataset + "-HAN")
    
    # check sim and sub relations
    print("label_sim_mat:",label_sim_mat.shape)
    print("label_sim_mat[0]:",label_sim_mat[0])
    print("label_sub_mat:",label_sub_mat.shape)
    print("label_sub_mat[0]:",label_sub_mat[0])
    print("label_sub_mat_sum:",np.sum(label_sub_mat))
    
    vocab_size = len(vocabulary_word2index)
    print("vocab_size:",vocab_size)
    
    # choosing whether to use k-fold cross-validation, pre-defined data split (much be provided), or hold-out validation
    if FLAGS.kfold == -1: # hold-out
        train, valid, test = load_data_multilabel_new(vocabulary_word2index, vocabulary_word2index_label,keep_label_percent=FLAGS.keep_label_percent,valid_portion=FLAGS.valid_portion,test_portion=FLAGS.test_portion,multi_label_flag=FLAGS.multi_label_flag,training_data_path=training_data_path) 
        # here train, test are tuples; turn train into trainlist.
        trainlist, validlist, testlist = list(), list(), list()
        trainlist.append(train)
        validlist.append(valid)
        testlist.append(test)
    elif FLAGS.kfold == 0: # pre-defined data split
        train = load_data_multilabel_pre_split(vocabulary_word2index, vocabulary_word2index_label,keep_label_percent=FLAGS.keep_label_percent,multi_label_flag=FLAGS.multi_label_flag,data_path=training_data_path)
        valid = load_data_multilabel_pre_split(vocabulary_word2index, vocabulary_word2index_label,keep_label_percent=FLAGS.keep_label_percent,multi_label_flag=FLAGS.multi_label_flag,data_path=validation_data_path)
        test = load_data_multilabel_pre_split(vocabulary_word2index, vocabulary_word2index_label,keep_label_percent=FLAGS.keep_label_percent,multi_label_flag=FLAGS.multi_label_flag,data_path=testing_data_path)
        # here train, test are tuples; turn train into trainlist.
        trainlist, validlist, testlist = list(), list(), list()
        for i in range(FLAGS.running_times):
            trainlist.append(train)
            validlist.append(valid)
            if i==0:
                testlist.append(test)
    else: # k-fold
        trainlist, validlist, testlist = load_data_multilabel_new_k_fold(vocabulary_word2index, vocabulary_word2index_label,keep_label_percent=FLAGS.keep_label_percent,kfold=FLAGS.kfold,test_portion=FLAGS.test_portion,multi_label_flag=FLAGS.multi_label_flag,training_data_path=training_data_path)
        # here trainlist, testlist are list of tuples.
    # get and pad testing data: there is only one testing data, but kfold training and validation data
    assert len(testlist) == 1
    testX, testY = testlist[0]
    testX = pad_sequences(testX, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length

    #2.create session.
    config=tf.ConfigProto()
    #config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    config.gpu_options.allow_growth=False
    with tf.Session(config=config) as sess:
        #Instantiate Model
        #num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,num_sentences,vocab_size,embed_size,
        #hidden_size,is_training
        model=HAN(num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate,FLAGS.sequence_length,FLAGS.num_sentences,vocab_size,FLAGS.embed_size,FLAGS.hidden_size,FLAGS.is_training,FLAGS.lambda_sim,FLAGS.lambda_sub,FLAGS.dynamic_sem,FLAGS.dynamic_sem_l2,FLAGS.per_label_attention,FLAGS.per_label_sent_only,multi_label_flag=FLAGS.multi_label_flag)
        
        num_runs = len(trainlist)
        #validation results variables
        valid_loss, valid_acc_th,valid_prec_th,valid_rec_th,valid_fmeasure_th,valid_hamming_loss_th,valid_prec_per_label_th,valid_rec_per_label_th,valid_f1_per_label_th,valid_acc_topk,valid_prec_topk,valid_rec_topk,valid_fmeasure_topk,valid_hamming_loss_topk,valid_prec_per_label_topk,valid_rec_per_label_topk,valid_f1_per_label_topk,valid_macro_accuracy, valid_macro_precision, valid_macro_recall, valid_macro_f1, valid_macro_auc, valid_micro_accuracy, valid_micro_precision, valid_micro_recall, valid_micro_f1, valid_micro_auc, valid_micro_precision_diag, valid_micro_recall_diag, valid_micro_f1_diag, valid_micro_precision_proc, valid_micro_recall_proc, valid_micro_f1_proc = [0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs # initialise the testing result lists
        
        test_loss, test_acc_th,test_prec_th,test_rec_th,test_fmeasure_th,test_hamming_loss_th,test_prec_per_label_th,test_rec_per_label_th,test_f1_per_label_th,test_acc_topk,test_prec_topk,test_rec_topk,test_fmeasure_topk,test_hamming_loss_topk,test_prec_per_label_topk,test_rec_per_label_topk,test_f1_per_label_topk,test_macro_accuracy, test_macro_precision, test_macro_recall, test_macro_f1, test_macro_auc, test_micro_accuracy, test_micro_precision, test_micro_recall, test_micro_f1, test_micro_auc, test_micro_precision_diag, test_micro_recall_diag, test_micro_f1_diag, test_micro_precision_proc, test_micro_recall_proc, test_micro_f1_proc = [0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs,[0]*num_runs # initialise the testing result lists
        #outputs
        output_valid, output_test = "", ""
        
        # start iterating over k-folds for training and testing  
        num_run = 0
        time_train = [0]*num_runs # get time spent in training        
        for train, valid in zip(trainlist, validlist):
            # remove older checkpoints
            if FLAGS.remove_ckpts_before_train and os.path.exists(FLAGS.ckpt_dir):
                filelist = [f for f in os.listdir(FLAGS.ckpt_dir)]
                for f in filelist:
                    os.remove(os.path.join(FLAGS.ckpt_dir, f))
                print('Checkpoints from the previous fold or run removed.')
            
            print('\n--RUN',num_run,'START--\n')
            start_time_train = time.time() # staring time in training
            # k-fold dataset creation
            trainX, trainY = train
            validX, validY = valid
            # Data preprocessing.Sequence padding
            print("start padding & transform to one hot...")
            trainX = pad_sequences(trainX, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length
            validX = pad_sequences(validX, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length
            #with open(FLAGS.cache_path, 'w') as data_f: #save data to cache file, so we can use it next time quickly.
            #    pickle.dump((trainX,trainY,testX,testY,vocabulary_index2word),data_f)
            print("trainX[0]:", trainX[0]) ;#print("trainY[0]:", trainY[0])
            #print("validX[0]:", validX[0])
            # Converting labels to binary vectors
            print("end padding & transform to one hot...")
            
            saver=tf.train.Saver(max_to_keep = 1) # only keep the latest model, here is the best model
            if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
                print("Restoring Variables from Checkpoint")
                saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            else:
                print('Initializing Variables')
                sess.run(tf.global_variables_initializer()) # which initialise parameters
                if FLAGS.use_embedding: #load pre-trained word embedding
                    assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, model,num_run,word2vec_model_path=word2vec_model_path)
                if FLAGS.dynamic_sem:
                    assign_sim_sub_matrices(sess,FLAGS.lambda_sim,FLAGS.lambda_sub,label_sim_mat,label_sub_mat,model)
                if FLAGS.use_label_embedding: #initialise label embedding
                    #initialise the final projection matrix
                    assign_pretrained_label_embedding(sess,vocabulary_index2word_label,model,num_run,label_embedding_model_path=label_embedding_model_path)
                    #initialise the per-label context vectors
                    if FLAGS.per_label_attention:
                        assign_pretrained_label_embedding_per_label(sess,vocabulary_index2word_label,model,num_run,label_embedding_model_path=label_embedding_model_path_per_label)
                    
            #print('loaded Uw', sess.run(model.context_vector_word))
            curr_epoch=sess.run(model.epoch_step) # after restoring, the parameters are initialised.
            print('curr_epoch:',curr_epoch)
            #3.feed data & training
            number_of_training_data=len(trainX)
            print("number_of_training_data:",number_of_training_data)
            #previous_eval_loss=10000
            #previous_eval_fmeasure=0
            previous_micro_f1=0 # we optimise micro-f1 with validation set during training
            #best_eval_loss=10000
            best_micro_f1=0
            batch_size=FLAGS.batch_size
            curr_step = curr_epoch*batch_size
            # iterating over epoches
            for epoch in range(curr_epoch,FLAGS.num_epochs):
                print('start next epoch:',epoch)
                # if epoch%10==0:
                    # display_results_bool=True
                # else:
                    # display_results_bool=False            
                loss, acc, counter = 0.0, 0.0, 0
                #for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)): # might have lost a very little part of data (105 out of 15849) here which is the mod after dividing the batch_size
                for start, end in zip(list(range(0, number_of_training_data, batch_size)),list(range(batch_size, number_of_training_data, batch_size))+[number_of_training_data]):
                    #print('in training:',start,end)
                    if num_run==0 and epoch==0 and counter==0: #num_run for folds, epoch for iterations, counter for batches
                        print("trainX[start:end]:",trainX[start:end]);#print("trainY[start:end]:",trainY[start:end])
                    feed_dict = {model.input_x: trainX[start:end],model.dropout_keep_prob: 0.5}
                    if not FLAGS.multi_label_flag:
                        feed_dict[model.input_y] = trainY[start:end]
                    else:
                        feed_dict[model.input_y_multilabel]=trainY[start:end]
                    feed_dict[model.label_sim_matrix_static]=label_sim_mat
                    feed_dict[model.label_sub_matrix_static]=label_sub_mat
                    # now we start training
                    curr_summary_str,curr_summary_l_epoch,curr_loss,curr_acc,label_sim_mat_updated,label_sub_mat_updated,_=sess.run([model.training_loss,model.training_loss_per_epoch,model.loss_val,model.accuracy,model.label_sim_matrix,model.label_sub_matrix,model.train_op],feed_dict)#curr_acc--->modelToEval.accuracy
                    
                    if FLAGS.dynamic_sem == True:
                        # # check the amount of changes
                        #print('sim_absolute_update_sum:',np.sum(np.absolute(label_sim_mat - label_sim_mat_updated)))
                        #print('sub_absolute_update_sum:',np.sum(np.absolute(label_sub_mat - label_sub_mat_updated)))
                        label_sim_mat = label_sim_mat_updated
                        label_sub_mat = label_sub_mat_updated
                        # print("label_sim_mat[0]-updated:",label_sim_mat[0])
                        # print("label_sub_mat[0]-updated:",label_sub_mat[0])
                        
                    curr_step=curr_step+1
                    model.writer.add_summary(curr_summary_str,curr_step)
                    if counter==0:
                        model.writer.add_summary(curr_summary_l_epoch,epoch) # this is the training loss per epoch
                    loss,counter,acc=loss+curr_loss,counter+1,acc+curr_acc
                    # output every 50 batches
                    if counter %50==0:
                        print("HAN==>Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" %(epoch,counter,loss/float(counter),acc/float(counter))) #tTrain Accuracy:%.3f---》acc/float(counter)
                    
                    # using validation set to calculate validation loss, then to see whether we need to decay the learning rate.
                    # and the decay of learning rate is used for early stopping.
                    if FLAGS.weight_decay_testing:                    
                        ##VALIDATION VALIDATION VALIDATION PART######################################################################################################
                        # to check whether the evaluation loss on testing data is decreasing, if not then half the learning rate: so the update of weights gets halved.
                        if FLAGS.batch_size!=0 and (start%(FLAGS.validate_step*FLAGS.batch_size)==0):
                            print(epoch, FLAGS.validate_step, FLAGS.batch_size) # here shows only when start being 0, the program goes under this condition. This is okay as our dataset is not too large.
                            #eval_loss, eval_acc = do_eval(sess, model, testX, testY, batch_size,vocabulary_index2word_label)
                            #eval_loss, eval_acc,eval_prec,eval_rec,eval_fmeasure = do_eval_multilabel(sess, model, tag_pair_matrix, label_sim_matrix, testX, testY, batch_size,vocabulary_index2word_label,epoch,number_labels_to_predict=11)
                            eval_loss,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,micro_f1,_,_,_,_,_,_,_ = do_eval_multilabel_threshold(sess,model,label_sim_mat,label_sub_mat,validX,validY,batch_size,vocabulary_index2word,vocabulary_index2word_label,epoch,threshold=FLAGS.pred_threshold,display_results_bool=False,hamming_q=FLAGS.ave_labels_per_doc,top_number=FLAGS.topk,record_to_tensorboard=False)
                            #print("validation.part. previous_eval_loss:", previous_eval_loss,";current_eval_loss:", eval_loss)
                            print("validation.part. previous_micro_f1:", previous_micro_f1,";current_micro_f1:", micro_f1)
                            #print("validation.part. previous_eval_fmeasure:", previous_eval_fmeasure,";current_eval_fmeasure:", eval_fmeasure)
                            #if eval_loss > previous_eval_loss: #if loss is not decreasing compared to the previous evaluation step (here is an epoch)
                            if micro_f1 < previous_micro_f1:
                            #if eval_fmeasure < previous_eval_fmeasure: # if f-measure is not increasing
                            # reduce the learning rate by a factor of 0.5
                                print("HAN==>validation.part.going to reduce the learning rate.")
                                learning_rate1 = sess.run(model.learning_rate)
                                lrr=sess.run([model.learning_rate_decay_half_op])
                                learning_rate2 = sess.run(model.learning_rate) # the new learning rate
                                print("HAN==>validation.part.learning_rate_original:", learning_rate1, " ;learning_rate_new:",learning_rate2)
                            else:
                                #if eval_loss<best_eval_loss:
                                if micro_f1 > best_micro_f1:
                                    #print("validation step: HAN==>going to save the model.eval_loss:",eval_loss,";best_eval_loss:",best_eval_loss)
                                    print("validation step: HAN==>going to save the model.micro_f1:",micro_f1,";best_micro_f1:",best_micro_f1)
                                    # save model to checkpoint
                                    save_path = FLAGS.ckpt_dir + "model.ckpt"
                                    saver.save(sess, save_path, global_step=epoch)
                                    #best_eval_loss=eval_loss
                                    best_micro_f1 = micro_f1
                            
                            #previous_eval_loss = eval_loss
                            previous_micro_f1 = micro_f1
                            #previous_eval_fmeasure = eval_fmeasure
                        ##VALIDATION VALIDATION VALIDATION PART######################################################################################################
                
                #epoch increment
                print("going to increment epoch counter....")
                sess.run(model.epoch_increment)

                # 4.show validation results during training [not testing results!]
                if epoch % FLAGS.validate_every==0: # for every epoch, evaluate with the validation set and save model to checkpoint
                    if epoch%50 == 0 and epoch != 0:
                        display_results_bool=True
                    else:
                        display_results_bool=False
                    eval_loss,eval_acc_th,eval_prec_th,eval_rec_th,eval_fmeasure_th,eval_hamming_loss_th,eval_prec_per_label_th,eval_rec_per_label_th,eval_f1_per_label_th,eval_acc_topk,eval_prec_topk,eval_rec_topk,eval_fmeasure_topk,eval_hamming_loss_topk,eval_prec_per_label_topk,eval_rec_per_label_topk,eval_f1_per_label_topk,macro_accuracy, macro_precision, macro_recall, macro_f1, macro_auc, micro_accuracy, micro_precision, micro_recall, micro_f1, micro_auc,_,_,_,_,_,_ = do_eval_multilabel_threshold(sess,model,label_sim_mat,label_sub_mat,validX,validY,batch_size,vocabulary_index2word,vocabulary_index2word_label,epoch,threshold=FLAGS.pred_threshold,display_results_bool=display_results_bool,hamming_q=FLAGS.ave_labels_per_doc,top_number=FLAGS.topk,record_to_tensorboard=True)
                    print('lambda_sim', FLAGS.lambda_sim, 'lambda_sub', FLAGS.lambda_sub)
                    print("HAN==>Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f\tValidation Hamming Loss: %.3f\tValidation Precision: %.3f\tValidation Recall: %.3f\tValidation F-measure: %.3f\tValidation Accuracy@k: %.3f\tValidation Hamming Loss@k: %.3f\tValidation Precision@k: %.3f\tValidation Recall@k: %.3f\tValidation F-measure@k: %.3f\tValidation macro-Accuracy: %.3f\tValidation macro-Precision: %.3f\tValidation macro-Recall: %.3f\tValidation macro-F-measure: %.3f\tValidation macro-AUC: %.3f\tValidation micro-Accuracy: %.3f\tValidation micro-Precision: %.3f\tValidation micro-Recall: %.3f\tValidation micro-F-measure: %.3f\tValidation micro-AUC: %.3f" % (epoch,eval_loss,eval_acc_th,eval_hamming_loss_th,eval_prec_th,eval_rec_th,eval_fmeasure_th,eval_acc_topk,eval_hamming_loss_topk,eval_prec_topk,eval_rec_topk,eval_fmeasure_topk,macro_accuracy, macro_precision, macro_recall, macro_f1, macro_auc, micro_accuracy, micro_precision, micro_recall, micro_f1, micro_auc))
                    # #output per-label metrics [not showing them here, otherwise too many texts]
                    # print('Validation results, per label, threshold:\n' + show_per_label_results(vocabulary_index2word_label,eval_prec_per_label_th,eval_rec_per_label_th,eval_f1_per_label_th))
                    # print('Validation results, per label, top %d:\n' % FLAGS.topk + show_per_label_results(vocabulary_index2word_label,eval_prec_per_label_topk,eval_rec_per_label_topk,eval_f1_per_label_topk))
                    
                    # if FLAGS.weight_decay_testing:        
                        # print("validation.part. previous_eval_loss:", previous_eval_loss,";current_eval_loss:", eval_loss)
                        # #print("validation.part. previous_eval_fmeasure:", previous_eval_fmeasure,";current_eval_fmeasure:", eval_fmeasure)
                        # if eval_loss > previous_eval_loss: #if loss is not decreasing
                        # #if eval_fmeasure < previous_eval_fmeasure: # if f-measure is not increasing
                        # # reduce the learning rate by a factor of 0.5
                            # print("HAN==>validation.part.going to reduce the learning rate.")
                            # learning_rate1 = sess.run(model.learning_rate)
                            # lrr=sess.run([model.learning_rate_decay_half_op])
                            # learning_rate2 = sess.run(model.learning_rate) # the new learning rate
                            # print("HAN==>validation.part.learning_rate_original:", learning_rate1, " ;learning_rate_new:",learning_rate2)
                        # else:
                            # if eval_loss<best_eval_loss:
                                # print("HAN==>going to save the model.eval_loss:",eval_loss,";best_eval_loss:",best_eval_loss)
                                # # save model to checkpoint
                                # save_path = FLAGS.ckpt_dir + "model.ckpt"
                                # saver.save(sess, save_path, global_step=epoch)
                                # best_eval_loss=eval_loss
                        
                        # previous_eval_loss = eval_loss
                        # #previous_eval_fmeasure = eval_fmeasure
                    #if eval_loss<best_eval_loss:
                    if micro_f1>best_micro_f1:
                        #print("after epoch: HAN==>going to save the model.eval_loss:",eval_loss,";best_eval_loss:",best_eval_loss)
                        print("after the epoch: HAN==>going to save the model.micro_f1:",micro_f1,";best_micro_f1:",best_micro_f1)
                        # save model to checkpoint
                        save_path = FLAGS.ckpt_dir + "model.ckpt"
                        saver.save(sess, save_path, global_step=epoch)
                        #best_eval_loss=eval_loss
                        best_micro_f1=micro_f1
                current_learning_rate = sess.run(model.learning_rate)    
                if current_learning_rate<FLAGS.early_stop_lr:
                    break
            time_train[num_run] = time.time() - start_time_train # store the training time for this fold to the list time_train().
            
            # reload the best model to get the final validation results and testing 
            if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
                print("Restoring Variables from Checkpoint of the Best Validation Model")
                saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            
            # visualise the model weights (so far, for HLAN only)
            if FLAGS.visualise_labelwise_layers: # includes the final projection layer and the labelwise attention context matrices
                list_query_labels = vocabulary_word2index_label.keys() # to query the topk similar labels for all labels
                viz_le(sess,model,vocabulary_index2word_label,vocabulary_word2index_label,list_query_labels)
        
            # 5.report final validation results
            if curr_epoch >= FLAGS.num_epochs:
                # to initialise epoch in case that curr_epoch >= FLAGS.num_epochs (otherwise the variable "epoch" is already initialised in the for loop, over epochs)
                epoch = FLAGS.num_epochs - 1
                
            valid_loss[num_run], valid_acc_th[num_run],valid_prec_th[num_run],valid_rec_th[num_run],valid_fmeasure_th[num_run],valid_hamming_loss_th[num_run],valid_prec_per_label_th[num_run],valid_rec_per_label_th[num_run],valid_f1_per_label_th[num_run],valid_acc_topk[num_run],valid_prec_topk[num_run],valid_rec_topk[num_run],valid_fmeasure_topk[num_run],valid_hamming_loss_topk[num_run],valid_prec_per_label_topk[num_run],valid_rec_per_label_topk[num_run],valid_f1_per_label_topk[num_run],valid_macro_accuracy[num_run], valid_macro_precision[num_run], valid_macro_recall[num_run], valid_macro_f1[num_run], valid_macro_auc[num_run], valid_micro_accuracy[num_run], valid_micro_precision[num_run], valid_micro_recall[num_run], valid_micro_f1[num_run], valid_micro_auc[num_run],valid_micro_precision_diag[num_run], valid_micro_recall_diag[num_run], valid_micro_f1_diag[num_run], valid_micro_precision_proc[num_run], valid_micro_recall_proc[num_run], valid_micro_f1_proc[num_run] = do_eval_multilabel_threshold(sess,model,label_sim_mat,label_sub_mat,validX,validY,batch_size,vocabulary_index2word,vocabulary_index2word_label,epoch,threshold=FLAGS.pred_threshold,display_results_bool=True,hamming_q=FLAGS.ave_labels_per_doc,top_number=FLAGS.topk,record_to_tensorboard=False)
            
            print("HAN==>Run %d Validation results:%.3f\tValidation Accuracy: %.3f\tValidation Hamming Loss: %.3f\tValidation Precision: %.3f\tValidation Recall: %.3f\tValidation F-measure: %.3f\tValidation Accuracy@k: %.3f\tValidation Hamming Loss@k: %.3f\tValidation Precision@k: %.3f\tValidation Recall@k: %.3f\tValidation F-measure@k: %.3f\tValidation macro-Accuracy: %.3f\tValidation macro-Precision: %.3f\tValidation macro-Recall: %.3f\tValidation macro-F-measure: %.3f\tValidation macro-AUC: %.3f\tValidation micro-Accuracy: %.3f\tValidation micro-Precision: %.3f\tValidation micro-Recall: %.3f\tValidation micro-F-measure: %.3f\tValidation micro-AUC: %.3f" % (num_run,valid_loss[num_run],valid_acc_th[num_run],valid_hamming_loss_th[num_run],valid_prec_th[num_run],valid_rec_th[num_run],valid_fmeasure_th[num_run],valid_acc_topk[num_run],valid_hamming_loss_topk[num_run],valid_prec_topk[num_run],valid_rec_topk[num_run],valid_fmeasure_topk[num_run],valid_macro_accuracy[num_run], valid_macro_precision[num_run], valid_macro_recall[num_run], valid_macro_f1[num_run], valid_macro_auc[num_run], valid_micro_accuracy[num_run], valid_micro_precision[num_run], valid_micro_recall[num_run], valid_micro_f1[num_run], valid_micro_auc[num_run]))
            
            #output code type metrics (if mimic)
            if 'mimic3-ds' in FLAGS.dataset:
                print('Validation diagnosis results, prec, rec, F1:', valid_micro_precision_diag[num_run], valid_micro_recall_diag[num_run], valid_micro_f1_diag[num_run])
                print('Validation procedure results, prec, rec, F1:', valid_micro_precision_proc[num_run], valid_micro_recall_proc[num_run], valid_micro_f1_proc[num_run])
                
            #output per-label metrics
            print('Validation results, per label, threshold:\n' + show_per_label_results(vocabulary_index2word_label,valid_prec_per_label_th[num_run],valid_rec_per_label_th[num_run],valid_f1_per_label_th[num_run]))
            print('Validation results, per label, top %d:\n' % FLAGS.topk + show_per_label_results(vocabulary_index2word_label,valid_prec_per_label_topk[num_run],valid_rec_per_label_topk[num_run],valid_f1_per_label_topk[num_run]))
            
            output_valid = output_valid + "\n" + "HAN=>Run %d Validation results Validation Loss:%.3f\tValidation Accuracy: %.3f\tValidation Hamming Loss: %.3f\tValidation Precision: %.3f\tValidation Recall: %.3f\tValidation F-measure: %.3f\tValidation Accuracy@k: %.3f\tValidation Hamming Loss@k: %.3f\tValidation Precision@k: %.3f\tValidation Recall@k: %.3f\tValidation F-measure@k: %.3f\tValidation macro-Accuracy: %.3f\tValidation macro-Precision: %.3f\tValidation macro-Recall: %.3f\tValidation macro-F-measure: %.3f\tValidation macro-AUC: %.3f\tValidation micro-Accuracy: %.3f\tValidation micro-Precision: %.3f\tValidation micro-Recall: %.3f\tValidation micro-F-measure: %.3f\tValidation micro-AUC: %.3f" % (num_run,valid_loss[num_run],valid_acc_th[num_run],valid_hamming_loss_th[num_run],valid_prec_th[num_run],valid_rec_th[num_run],valid_fmeasure_th[num_run],valid_acc_topk[num_run],valid_hamming_loss_topk[num_run],valid_prec_topk[num_run],valid_rec_topk[num_run],valid_fmeasure_topk[num_run],valid_macro_accuracy[num_run], valid_macro_precision[num_run], valid_macro_recall[num_run], valid_macro_f1[num_run], valid_macro_auc[num_run], valid_micro_accuracy[num_run], valid_micro_precision[num_run], valid_micro_recall[num_run], valid_micro_f1[num_run], valid_micro_auc[num_run]) + "\n" # store validation results of each run to the output_valid string.
            
            # 6.here we use the testing data, to report testing results
            test_loss[num_run], test_acc_th[num_run],test_prec_th[num_run],test_rec_th[num_run],test_fmeasure_th[num_run],test_hamming_loss_th[num_run],test_prec_per_label_th[num_run],test_rec_per_label_th[num_run],test_f1_per_label_th[num_run],test_acc_topk[num_run],test_prec_topk[num_run],test_rec_topk[num_run],test_fmeasure_topk[num_run],test_hamming_loss_topk[num_run],test_prec_per_label_topk[num_run],test_rec_per_label_topk[num_run],test_f1_per_label_topk[num_run],test_macro_accuracy[num_run], test_macro_precision[num_run], test_macro_recall[num_run], test_macro_f1[num_run], test_macro_auc[num_run], test_micro_accuracy[num_run], test_micro_precision[num_run], test_micro_recall[num_run], test_micro_f1[num_run], test_micro_auc[num_run],test_micro_precision_diag[num_run], test_micro_recall_diag[num_run], test_micro_f1_diag[num_run], test_micro_precision_proc[num_run], test_micro_recall_proc[num_run], test_micro_f1_proc[num_run] = do_eval_multilabel_threshold(sess,model,label_sim_mat,label_sub_mat,testX,testY,batch_size,vocabulary_index2word,vocabulary_index2word_label,epoch,threshold=FLAGS.pred_threshold,display_results_bool=True,hamming_q=FLAGS.ave_labels_per_doc,top_number=FLAGS.topk,record_to_tensorboard=False,output_logits=FLAGS.output_logits,output_logits_filename_prefix=filename_common_prefix,num_run=num_run) # output logit set as true for testing.
            
            print("HAN==>Run %d Test results Test Loss:%.3f\tValidation Accuracy: %.3f\tValidation Hamming Loss: %.3f\tValidation Precision: %.3f\tValidation Recall: %.3f\tValidation F-measure: %.3f\tValidation Accuracy@k: %.3f\tValidation Hamming Loss@k: %.3f\tValidation Precision@k: %.3f\tValidation Recall@k: %.3f\tValidation F-measure@k: %.3f\tValidation macro-Accuracy: %.3f\tValidation macro-Precision: %.3f\tValidation macro-Recall: %.3f\tValidation macro-F-measure: %.3f\tValidation macro-AUC: %.3f\tValidation micro-Accuracy: %.3f\tValidation micro-Precision: %.3f\tValidation micro-Recall: %.3f\tValidation micro-F-measure: %.3f\tValidation micro-AUC: %.3f" % (num_run,test_loss[num_run],test_acc_th[num_run],test_hamming_loss_th[num_run],test_prec_th[num_run],test_rec_th[num_run],test_fmeasure_th[num_run],test_acc_topk[num_run],test_hamming_loss_topk[num_run],test_prec_topk[num_run],test_rec_topk[num_run],test_fmeasure_topk[num_run],test_macro_accuracy[num_run], test_macro_precision[num_run], test_macro_recall[num_run], test_macro_f1[num_run], test_macro_auc[num_run], test_micro_accuracy[num_run], test_micro_precision[num_run], test_micro_recall[num_run], test_micro_f1[num_run], test_micro_auc[num_run]))
            
            #output code type metrics (if mimic)
            if 'mimic3-ds' in FLAGS.dataset:
                print('Test diagnosis code results, prec, rec, F1:', test_micro_precision_diag[num_run], test_micro_recall_diag[num_run], test_micro_f1_diag[num_run])
                print('Test procedure code results, prec, rec, F1:', test_micro_precision_proc[num_run], test_micro_recall_proc[num_run], test_micro_f1_proc[num_run])
                
            #output per-label metrics
            print('Test results, per label, threshold:\n' + show_per_label_results(vocabulary_index2word_label,test_prec_per_label_th[num_run],test_rec_per_label_th[num_run],test_f1_per_label_th[num_run]))
            print('Test results, per label, top %d:\n' % FLAGS.topk + show_per_label_results(vocabulary_index2word_label,test_prec_per_label_topk[num_run],test_rec_per_label_topk[num_run],test_f1_per_label_topk[num_run]))
            
            output_test = output_test + "\n" + "HAN==>Run %d Test results Validation Loss:%.3f\tValidation Accuracy: %.3f\tValidation Hamming Loss: %.3f\tValidation Precision: %.3f\tValidation Recall: %.3f\tValidation F-measure: %.3f\tValidation Accuracy@k: %.3f\tValidation Hamming Loss@k: %.3f\tValidation Precision@k: %.3f\tValidation Recall@k: %.3f\tValidation F-measure@k: %.3f\tValidation macro-Accuracy: %.3f\tValidation macro-Precision: %.3f\tValidation macro-Recall: %.3f\tValidation macro-F-measure: %.3f\tValidation macro-AUC: %.3f\tValidation micro-Accuracy: %.3f\tValidation micro-Precision: %.3f\tValidation micro-Recall: %.3f\tValidation micro-F-measure: %.3f\tValidation micro-AUC: %.3f" % (num_run,test_loss[num_run],test_acc_th[num_run],test_hamming_loss_th[num_run],test_prec_th[num_run],test_rec_th[num_run],test_fmeasure_th[num_run],test_acc_topk[num_run],test_hamming_loss_topk[num_run],test_prec_topk[num_run],test_rec_topk[num_run],test_fmeasure_topk[num_run],test_macro_accuracy[num_run], test_macro_precision[num_run], test_macro_recall[num_run], test_macro_f1[num_run], test_macro_auc[num_run], test_micro_accuracy[num_run], test_micro_precision[num_run], test_micro_recall[num_run], test_micro_f1[num_run], test_micro_auc[num_run]) + "\n" # store the testing results of each run to the output_test string.
            
            #output test result immediately
            # get the experimental input settings
            #setting_dict = tf.flags.FLAGS.__flags
            #setting = ""
            #for key, value in setting_dict.items():
            #    setting = setting + key + ": " + str(value) + '\n'
            #setting:batch_size,embed_size,label_sim_threshold,lambda_sim,lambda_l1sig,weight_decay_testing,early_stop_lr,dynamic_sem,dynamic_sem_l2,per_label_attention
            setting = "batch_size: " + str(FLAGS.batch_size) + "\nembed_size: " + str(FLAGS.embed_size) + "\nvalidate_step: " + str(FLAGS.validate_step) + "\nlabel_sim_threshold: " + str(FLAGS.label_sim_threshold) + "\nlambda_sim: " + str(FLAGS.lambda_sim) + "\nlambda_sub: " + str(FLAGS.lambda_sub) + "\nnum_epochs: " + str(FLAGS.num_epochs) + "\nkeep_label_percent: " + str(FLAGS.keep_label_percent) + "\nweight_decay_testing: " + str(FLAGS.weight_decay_testing) + "\nearly_stop_lr: " + str(FLAGS.early_stop_lr) + "\ndynamic_sem: " + str(FLAGS.dynamic_sem) + "\ndynamic_sem_l2: " + str(FLAGS.dynamic_sem_l2) + "\nuse_label_embedding: " + str(FLAGS.use_label_embedding) + "\nper_label_attention: " + str(FLAGS.per_label_attention) + "\nper_label_sent_only: " + str(FLAGS.per_label_sent_only) + "\npred_threshold: " + str(FLAGS.pred_threshold)
            
            output_time_train = "--- This fold, run %s, took %s seconds ---" % (num_run, time_train[num_run])
            print('lambda_sim', FLAGS.lambda_sim, 'lambda_sub', FLAGS.lambda_sub, 'learning_rate', FLAGS.learning_rate)
            print(output_time_train)
            
            prediction_str = ""
            # output final predictions for qualitative analysis (with attention visualisation)
            if FLAGS.report_rand_pred == True:
                if FLAGS.per_label_attention: # to do for per_label_sent_only
                    prediction_str = display_for_qualitative_evaluation_per_label(sess,model,label_sim_mat,label_sub_mat,testX,testY,batch_size,vocabulary_index2word,vocabulary_index2word_label,threshold=FLAGS.pred_threshold,use_random_sampling=FLAGS.use_random_sampling) #default as not using random sampling, that is, to display all results with attention weights (for small test set)
                else:
                    prediction_str = display_for_qualitative_evaluation(sess,model,label_sim_mat,label_sub_mat,testX,testY,batch_size,vocabulary_index2word,vocabulary_index2word_label,threshold=FLAGS.pred_threshold,use_random_sampling=FLAGS.use_random_sampling)
            
            output_to_file(filename_common_prefix + ' results update.txt', setting + '\n' + output_valid + '\n' + output_test + '\n' + prediction_str + '\n' + output_time_train)
            
            # update the num_run
            num_run=num_run+1
                
    print('\n--Final Results--\n')
    print('lambda_sim', FLAGS.lambda_sim, 'lambda_sub', FLAGS.lambda_sub)
    
    # 7. report and output results    
    print("--- The whole program took %s seconds ---" % (time.time() - start_time))
    time_used = "--- The whole program took %s seconds ---" % (time.time() - start_time)
    if FLAGS.kfold != -1 and FLAGS.kfold != 0 or (FLAGS.kfold == 0 and FLAGS.running_times > 1):
        print("--- The average training took %s ± %s seconds ---" % (sum(time_train)/num_runs,statistics.stdev(time_train)))
        average_time_train = "--- The average training took %s ± %s seconds ---" % (sum(time_train)/num_runs,statistics.stdev(time_train))
    else:
        print("--- The average training took %s ± %s seconds ---" % (sum(time_train)/num_runs,0))
        average_time_train = "--- The average training took %s ± %s seconds ---" % (sum(time_train)/num_runs,0)

    # output structured evaluation results to csv files: valid and test
    output_csv_valid = "fold,loss,hamming_loss,acc,prec,rec,f1,hamming_loss@k,acc@k,prec@k,rec@k,f1@k,macro-acc,macro-prec,macro-rec,macro-f1,macro-AUC,micro-acc,micro-prec,micro-rec,micro-f1,micro-AUC" # set header
    output_csv_test = output_csv_valid # set header
    
    for ind, (v_loss,v_ham_loss,v_acc,v_prec,v_rec,v_f1,v_ham_loss_topk,v_acc_topk,v_prec_topk,v_rec_topk,v_f1_topk,v_micro_acc,v_micro_prec,v_micro_rec,v_micro_f1,v_micro_auc,v_macro_acc,v_macro_prec,v_macro_rec,v_macro_f1,v_macro_auc) in enumerate(zip(valid_loss, valid_hamming_loss_th,valid_acc_th,valid_prec_th,valid_rec_th,valid_fmeasure_th,valid_hamming_loss_topk,valid_acc_topk,valid_prec_topk,valid_rec_topk,valid_fmeasure_topk,valid_macro_accuracy, valid_macro_precision, valid_macro_recall, valid_macro_f1, valid_macro_auc, valid_micro_accuracy, valid_micro_precision, valid_micro_recall, valid_micro_f1, valid_micro_auc)):
        output_csv_valid = output_csv_valid + '\n' + ','.join([str(ind), '%.3f' % v_loss,'%.3f' % v_ham_loss,'%.3f' % v_acc,'%.3f' % v_prec,'%.3f' % v_rec,'%.3f' % v_f1,'%.3f' % v_ham_loss_topk,'%.3f' % v_acc_topk,'%.3f' % v_prec_topk,'%.3f' % v_rec_topk,'%.3f' % v_f1_topk,'%.3f' % v_micro_acc,'%.3f' % v_micro_prec,'%.3f' % v_micro_rec,'%.3f' % v_micro_f1,'%.3f' % v_micro_auc,'%.3f' % v_macro_acc,'%.3f' % v_macro_prec,'%.3f' % v_macro_rec,'%.3f' % v_macro_f1,'%.3f' % v_macro_auc]) # filling results per run, with rounding to 3 decimal places
    output_csv_valid = output_csv_valid + '\n' + ','.join(['mean±std']+[cal_ave_std(ele) for ele in [valid_loss, valid_hamming_loss_th,valid_acc_th,valid_prec_th,valid_rec_th,valid_fmeasure_th,valid_hamming_loss_topk,valid_acc_topk,valid_prec_topk,valid_rec_topk,valid_fmeasure_topk,valid_macro_accuracy, valid_macro_precision, valid_macro_recall, valid_macro_f1, valid_macro_auc, valid_micro_accuracy, valid_micro_precision, valid_micro_recall, valid_micro_f1, valid_micro_auc]])
    
    for ind, (t_loss,t_ham_loss,t_acc,t_prec,t_rec,t_f1,t_ham_loss_topk,t_acc_topk,t_prec_topk,t_rec_topk,t_f1_topk,t_micro_acc,t_micro_prec,t_micro_rec,t_micro_f1,t_micro_auc,t_macro_acc,t_macro_prec,t_macro_rec,t_macro_f1,t_macro_auc) in enumerate(zip(test_loss, test_hamming_loss_th,test_acc_th,test_prec_th,test_rec_th,test_fmeasure_th,test_hamming_loss_topk,test_acc_topk,test_prec_topk,test_rec_topk,test_fmeasure_topk,test_macro_accuracy, test_macro_precision, test_macro_recall, test_macro_f1, test_macro_auc, test_micro_accuracy, test_micro_precision, test_micro_recall, test_micro_f1, test_micro_auc)):
        output_csv_test = output_csv_test + '\n' + ','.join([str(ind), '%.3f' % t_loss,'%.3f' % t_ham_loss,'%.3f' % t_acc,'%.3f' % t_prec,'%.3f' % t_rec,'%.3f' % t_f1,'%.3f' % t_ham_loss_topk,'%.3f' % t_acc_topk,'%.3f' % t_prec_topk,'%.3f' % t_rec_topk,'%.3f' % t_f1_topk,'%.3f' % t_micro_acc,'%.3f' % t_micro_prec,'%.3f' % t_micro_rec,'%.3f' % t_micro_f1,'%.3f' % t_micro_auc,'%.3f' % t_macro_acc,'%.3f' % t_macro_prec,'%.3f' % t_macro_rec,'%.3f' % t_macro_f1,'%.3f' % t_macro_auc]) # filling results per run, with rounding to 3 decimal places
    output_csv_test = output_csv_test + '\n' + ','.join(['mean±std']+[cal_ave_std(ele) for ele in [test_loss, test_hamming_loss_th,test_acc_th,test_prec_th,test_rec_th,test_fmeasure_th,test_hamming_loss_topk,test_acc_topk,test_prec_topk,test_rec_topk,test_fmeasure_topk,test_macro_accuracy, test_macro_precision, test_macro_recall, test_macro_f1, test_macro_auc, test_micro_accuracy, test_micro_precision, test_micro_recall, test_micro_f1, test_micro_auc]])
    output_to_file(filename_common_prefix + ' valid.csv',output_csv_valid)
    output_to_file(filename_common_prefix + ' test.csv',output_csv_test)
    
    # output overall information: setting configuration, results, prediction and time used
    #update both output_valid and output_test with structured evaluation results
    structured_results_valid = "HAN==>Final Validation results Validation Loss:%s\tValidation Hamming Loss: %s\tValidation Accuracy: %s\tValidation Precision: %s\tValidation Recall: %s\tValidation F-measure: %s\tValidation Hamming Loss@k: %s\tValidation Accuracy@k: %s\tValidation Precision@k: %s\tValidation Recall@k: %s\tValidation F-measure@k: %s\tValidation macro-Accuracy: %s\tValidation macro-Precision: %s\tValidation macro-Recall: %s\tValidation macro-F-measure: %s\tValidation macro-AUC: %s\tValidation micro-Accuracy: %s\tValidation micro-Precision: %s\tValidation micro-Recall: %s\tValidation micro-F-measure: %s\tValidation micro-AUC: %s" % tuple(cal_ave_std(ele,with_min_max=True) for ele in [valid_loss, valid_hamming_loss_th,valid_acc_th,valid_prec_th,valid_rec_th,valid_fmeasure_th,valid_hamming_loss_topk,valid_acc_topk,valid_prec_topk,valid_rec_topk,valid_fmeasure_topk,valid_macro_accuracy, valid_macro_precision, valid_macro_recall, valid_macro_f1, valid_macro_auc, valid_micro_accuracy, valid_micro_precision, valid_micro_recall, valid_micro_f1, valid_micro_auc])
    print(structured_results_valid) #output to console as well
    output_valid = output_valid + '\n' + structured_results_valid + '\n'
    
    structured_results_test = "HAN==>Final Test results Test Loss:%s\tTest Hamming Loss: %s\tTest Accuracy: %s\tTest Precision: %s\tTest Recall: %s\tTest F-measure: %s\tTest Hamming Loss@k: %s\tTest Accuracy@k: %s\tTest Precision@k: %s\tTest Recall@k: %s\tTest F-measure@k: %s\tTest macro-Accuracy: %s\tTest macro-Precision: %s\tTest macro-Recall: %s\tTest macro-F-measure: %s\tTest macro-AUC: %s\tTest micro-Accuracy: %s\tTest micro-Precision: %s\tTest micro-Recall: %s\tTest micro-F-measure: %s\tTest micro-AUC: %s" % tuple(cal_ave_std(ele,with_min_max=True) for ele in [test_loss, test_hamming_loss_th,test_acc_th,test_prec_th,test_rec_th,test_fmeasure_th,test_hamming_loss_topk,test_acc_topk,test_prec_topk,test_rec_topk,test_fmeasure_topk,test_macro_accuracy, test_macro_precision, test_macro_recall, test_macro_f1, test_macro_auc, test_micro_accuracy, test_micro_precision, test_micro_recall, test_micro_f1, test_micro_auc])
    print(structured_results_test) #output to console as well
    output_test = output_test + '\n' + structured_results_test + '\n'
    output_to_file(filename_common_prefix + '.txt', setting + '\n' + output_valid + '\n' + output_test + '\n' + prediction_str + '\n' + time_used + '\n' + average_time_train)
    
    # output per-label results
    output_per_label_csv = ','.join(['code'] + ['valid_' + ele for ele in ['prec','rec','f1']] + ['test_' + ele for ele in ['prec','rec','f1']]) + '\n' # header
    for ind, (v_prec_f,v_rec_f,v_f1_f,t_prec_f,t_rec_f,t_f1_f) in enumerate(zip(cal_ave_std(valid_prec_per_label_th),cal_ave_std(valid_rec_per_label_th),cal_ave_std(valid_f1_per_label_th),cal_ave_std(test_prec_per_label_th),cal_ave_std(test_rec_per_label_th),cal_ave_std(test_f1_per_label_th))):
        output_per_label_csv = output_per_label_csv + ','.join([vocabulary_index2word_label[ind], v_prec_f,v_rec_f,v_f1_f,t_prec_f,t_rec_f,t_f1_f]) + '\n' # filling every row of mean±std results # _f postfix means formatted results
    output_to_file(filename_common_prefix + ' per label.csv',output_per_label_csv)
    
    # output code type results
    if 'mimic3-ds' in FLAGS.dataset:
        output_code_type_results_csv = ','.join(['num_run'] + ['valid_micro_' + ele + type for type in ['_diag', '_proc'] for ele in ['prec','rec','f1']] + ['test_micro_' + ele + type for type in ['_diag', '_proc'] for ele in ['prec','rec','f1']]) + '\n'
        for ind, (v_prec_diag,v_rec_diag,v_f1_diag,v_prec_proc,v_rec_proc,v_f1_proc,t_prec_diag,t_rec_diag,t_f1_diag,t_prec_proc,t_rec_proc,t_f1_proc) in enumerate(zip(valid_micro_precision_diag, valid_micro_recall_diag, valid_micro_f1_diag, valid_micro_precision_proc, valid_micro_recall_proc, valid_micro_f1_proc,test_micro_precision_diag, test_micro_recall_diag, test_micro_f1_diag, test_micro_precision_proc, test_micro_recall_proc, test_micro_f1_proc)):
            output_code_type_results_csv = output_code_type_results_csv + ','.join([str(ind), '%.3f' % v_prec_diag,'%.3f' % v_rec_diag,'%.3f' % v_f1_diag,'%.3f' % v_prec_proc,'%.3f' % v_rec_proc,'%.3f' % v_f1_proc,'%.3f' % t_prec_diag,'%.3f' % t_rec_diag,'%.3f' % t_f1_diag,'%.3f' % t_prec_proc,'%.3f' % t_rec_proc,'%.3f' % t_f1_proc]) + '\n' # filling results per run
        output_code_type_results_csv = output_code_type_results_csv + ','.join(['mean±std']+[cal_ave_std(ele) for ele in [valid_micro_precision_diag, valid_micro_recall_diag, valid_micro_f1_diag, valid_micro_precision_proc, valid_micro_recall_proc, valid_micro_f1_proc,test_micro_precision_diag, test_micro_recall_diag, test_micro_f1_diag, test_micro_precision_proc, test_micro_recall_proc, test_micro_f1_proc]])
        output_to_file(filename_common_prefix + ' code type results.csv',output_code_type_results_csv)
    pass

#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    with open(file_name, 'w', encoding="utf-8-sig") as f_output:
        f_output.write(str + '\n')

def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,model,num_run,word2vec_model_path=None):
    if num_run==0:
        print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    # transform embedding input into a dictionary
    # word2vecc=word2vec.load('word_embedding.txt') #load vocab-vector fiel.word2vecc['w91874']
   
   #word2vec_model = word2vec.load(word2vec_model_path, kind='bin') # for danielfrg's word2vec models
    word2vec_model = Word2Vec.load(word2vec_model_path) # for gensim word2vec models
    
    word2vec_dict = {}
    #for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors): # for danielfrg's word2vec models
    #    word2vec_dict[word] = vector # for danielfrg's word2vec models
    for _, word in enumerate(word2vec_model.wv.vocab):
        word2vec_dict[word] = word2vec_model.wv[word]
        
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list: which is a list of list, i.e. a list of word, where each word is a list of values as an embedding vector.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    #print(word_embedding_final[0]) # print the original embedding for the first word
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(model.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    if num_run==0:
        print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
        print("using pre-trained word emebedding.ended...")

def assign_sim_sub_matrices(sess,lambda_sim,lambda_sub,label_sim_mat,label_sub_mat,model):
    if lambda_sim != 0:
        label_sim_mat_tf = tf.constant(label_sim_mat, dtype=tf.float32)  # convert to tensor
        t_assign_sim = tf.assign(model.label_sim_matrix,label_sim_mat_tf)  # assign this value to our embedding variables of our model.
        sess.run(t_assign_sim)
    if lambda_sub != 0:
        label_sub_mat_tf = tf.constant(label_sub_mat, dtype=tf.float32)  # convert to tensor
        t_assign_sub = tf.assign(model.label_sub_matrix,label_sub_mat_tf)
        sess.run(t_assign_sub)

def assign_pretrained_label_embedding(sess,vocabulary_index2word_label,model,num_run,label_embedding_model_path=None):
    if num_run==0:
        print("initialsing pre-trained label emebedding:",label_embedding_model_path)
    
    word2vec_model_labels = Word2Vec.load(label_embedding_model_path) # for gensim word2vec models
    
    word2vec_dict_labels = {}
    for _, label in enumerate(word2vec_model_labels.wv.vocab):
        word2vec_dict_labels[label] = word2vec_model_labels.wv[label]
    
    num_classes = len(vocabulary_index2word_label)
    label_embedding_2dlist = [[]] * num_classes  # create an empty word_embedding list: which is a list of list, i.e. a list of word, where each word is a list of values as an embedding vector.
    bound = np.sqrt(6.0) / np.sqrt(num_classes + FLAGS.embed_size * 4)  # bound for random variables for Xavier initialisation.
    count_exist = 0;
    count_not_exist = 0
    for i in range(num_classes):  # loop over each label/class
        label = vocabulary_index2word_label[i]  # get a label
        embedding = None
        try:
            embedding = word2vec_dict_labels[label]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'label' exist a embedding
            label_embedding_2dlist[i] = embedding / float(np.linalg.norm(embedding) + 1e-6) # normalise to unit length
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            print(label, 'embedding inexist')
            label_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size * 4); # dimensionality as the final hidden layer of the model, which is 4 times of the input embedding size.
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    label_embedding_final = np.array(label_embedding_2dlist)  # covert to 2d array.
    label_embedding_final_transposed = label_embedding_final.transpose()
    #print(label_embedding_final.shape, label_embedding_final_transposed.shape,label_embedding_final_transposed[0]) # print the original embedding for the first word
    label_embedding_tp = tf.constant(label_embedding_final_transposed, dtype=tf.float32)  # convert to tensor, tp means transposed
    t_assign_label_embedding = tf.assign(model.W_projection,label_embedding_tp)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_label_embedding);
    if num_run==0:
        print("label. exists embedding:", count_exist, " ;label not exist embedding:", count_not_exist)
        print("using pre-trained label emebedding.ended...")

def assign_pretrained_label_embedding_per_label(sess,vocabulary_index2word_label,model,num_run,label_embedding_model_path=None):
    if num_run==0:
        print("initialsing pre-trained label embedding, per-label:",label_embedding_model_path)
    
    word2vec_model_labels = Word2Vec.load(label_embedding_model_path) # for gensim word2vec models
    
    word2vec_dict_labels = {}
    for _, label in enumerate(word2vec_model_labels.wv.vocab):
        word2vec_dict_labels[label] = word2vec_model_labels.wv[label]
    
    num_classes = len(vocabulary_index2word_label)
    label_embedding_2dlist = [[]] * num_classes  # create an empty word_embedding list: which is a list of list, i.e. a list of word, where each word is a list of values as an embedding vector.
    bound = np.sqrt(6.0) / np.sqrt(num_classes + FLAGS.embed_size * 4)  # bound for random variables for Xavier initialisation.
    count_exist = 0;
    count_not_exist = 0
    for i in range(num_classes):  # loop over each label/class
        label = vocabulary_index2word_label[i]  # get a label
        embedding = None
        try:
            embedding = word2vec_dict_labels[label]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'label' exist a embedding
            label_embedding_2dlist[i] = embedding / float(np.linalg.norm(embedding) + 1e-6) # normalise to unit length
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            print(label, 'embedding inexist')
            label_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size * 4); # dimensionality as the final hidden layer of the model, which is 4 times of the input embedding size.
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    label_embedding_final = np.array(label_embedding_2dlist)  # covert to 2d array.
    #print(label_embedding_final.shape, label_embedding_final_transposed.shape,label_embedding_final_transposed[0]) # print the original embedding for the first word
    label_embedding_tensor = tf.constant(label_embedding_final, dtype=tf.float32)  # convert to tensor
    if not FLAGS.per_label_sent_only:
        t_assign_label_embedding_word_level = tf.assign(model.context_vector_word_per_label,label_embedding_tensor)  # initialise label embedding to word-level per-label context vector
        sess.run(t_assign_label_embedding_word_level)
        print('per-label word-level context vector initialised')
    t_assign_label_embedding_sent_level = tf.assign(model.context_vector_sentence_per_label,label_embedding_tensor)  # initialise label embedding to sent-level per-label context vector
    sess.run(t_assign_label_embedding_sent_level)
    print('per-label sentence-level context vector initialised')
    if num_run==0:
        print("label. exists embedding:", count_exist, " ;label not exist embedding:", count_not_exist)
        print("using pre-trained label emebedding.ended...")
        
# based on a threshold, for multilabel
def do_eval_multilabel_threshold(sess,modelToEval,label_sim_mat,label_sub_mat,evalX,evalY,batch_size,vocabulary_index2word,vocabulary_index2word_label,epoch,threshold=0.5,display_results_bool=True,hamming_q=FLAGS.ave_labels_per_doc,top_number=FLAGS.topk,record_to_tensorboard=True,output_logits=False,output_logits_filename_prefix='',num_run=0):
    #print(display_results_bool)
    number_examples=len(evalX)
    print("number_examples", number_examples)
    #generate random index for batch and document
    #rn.seed(1)
    batch_chosen=rn.randint(0,number_examples//batch_size)
    x_chosen=rn.randint(0,batch_size)
    eval_loss,eval_acc_th,eval_prec_th,eval_rec_th,eval_fmeasure_th,eval_acc_topk,eval_prec_topk,eval_rec_topk,eval_fmeasure_topk,eval_hamming_loss_th,eval_hamming_loss_topk,eval_counter=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0
    eval_step=epoch*(number_examples//batch_size)
    #logits_all = np.array([])
    #for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)): # a few samples in the evaluation set can be lost, due to the training and testing with the batch size (this may not be exactly divided).
    for start,end in zip(list(range(0,number_examples,batch_size)),list(range(batch_size,number_examples,batch_size))+[number_examples]):
        #print('now evaluating:',start,end)
        feed_dict = {modelToEval.input_x: evalX[start:end], modelToEval.dropout_keep_prob: 1, modelToEval.label_sim_matrix:label_sim_mat, modelToEval.label_sub_matrix:label_sub_mat}
        #if (start==0):
        #    print(evalX[start:end])
        if not FLAGS.multi_label_flag:
            feed_dict[modelToEval.input_y] = evalY[start:end]
        else:
            feed_dict[modelToEval.input_y_multilabel] = evalY[start:end]
        #curr_eval_loss, logits,curr_eval_acc= sess.run([modelToEval.loss_val,modelToEval.logits,modelToEval.accuracy],feed_dict)#curr_eval_acc--->modelToEval.accuracy
        curr_summary_str,curr_summary_l_epoch,curr_eval_loss,logits= sess.run([modelToEval.validation_loss,modelToEval.validation_loss_per_epoch,modelToEval.loss_val,modelToEval.logits],feed_dict)#curr_eval_acc--->modelToEval.accuracy
        #storing all logits across epochs: concatenating the logits together by row
        #get the full raw prediction matrix
        if eval_counter==0:
            logits_all=logits
        else:
            logits_all=np.concatenate((logits_all,logits),axis=0)
        logits_all_sigmoid = sigmoid_array(logits_all)
    
        if record_to_tensorboard:
            eval_step = eval_step + 1
            modelToEval.writer.add_summary(curr_summary_str,eval_step)
            if eval_counter==0:
                modelToEval.writer.add_summary(curr_summary_l_epoch,epoch)
        
        eval_counter=eval_counter+1
        #print(type(logits))
        #n=0
        #print(len(logits)) #=batch_size
        curr_eval_acc_th=0.0
        curr_eval_prec_th=0.0
        curr_eval_rec_th=0.0
        curr_hamming_loss_th=0.0
        curr_eval_acc_topk=0.0
        curr_eval_prec_topk=0.0
        curr_eval_rec_topk=0.0
        curr_hamming_loss_topk=0.0
        for x in range(0,len(logits)):
            label_list_th = get_label_using_logits_threshold(logits[x],threshold)
            label_list_topk = get_label_using_logits(logits[x], vocabulary_index2word_label,top_number)
            # display a particular prediction result
            if x==x_chosen and start==batch_chosen*batch_size and display_results_bool==True:
                print('doc:',*display_results(evalX[start+x],vocabulary_index2word,for_label=False))
                print('prediction-0.5:',*display_results(label_list_th,vocabulary_index2word_label))
                print('prediction-topk:',*display_results(label_list_topk,vocabulary_index2word_label))
                get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
                print('labels:',*display_results(get_indexes(1,evalY[start+x]),vocabulary_index2word_label))
            #print(label_list_top5)
            #print(evalY[start:end][x])
            curr_eval_acc_th=curr_eval_acc_th + calculate_accuracy(list(label_list_th), evalY[start:end][x],eval_counter)
            precision, recall = calculate_precision_recall(list(label_list_th), evalY[start:end][x],eval_counter)
            curr_eval_prec_th = curr_eval_prec_th + precision
            curr_eval_rec_th = curr_eval_rec_th + recall
            hamming_loss_th = calculate_hamming_loss(list(label_list_th), evalY[start:end][x])
            curr_hamming_loss_th = curr_hamming_loss_th + hamming_loss_th
            
            curr_eval_acc_topk=curr_eval_acc_topk + calculate_accuracy(list(label_list_topk), evalY[start:end][x],eval_counter)
            precision_topk, recall_topk = calculate_precision_recall(list(label_list_topk), evalY[start:end][x],eval_counter)
            curr_eval_prec_topk = curr_eval_prec_topk + precision_topk
            curr_eval_rec_topk = curr_eval_rec_topk + recall_topk
            hamming_loss_topk = calculate_hamming_loss(list(label_list_topk), evalY[start:end][x])
            curr_hamming_loss_topk = curr_hamming_loss_topk + hamming_loss_topk

            #print(curr_eval_acc)
        eval_acc_th = eval_acc_th + curr_eval_acc_th/float(len(logits))
        eval_prec_th = eval_prec_th + curr_eval_prec_th/float(len(logits))
        eval_rec_th = eval_rec_th + curr_eval_rec_th/float(len(logits))
        eval_hamming_loss_th = eval_hamming_loss_th + curr_hamming_loss_th/float(len(logits))
        
        eval_acc_topk = eval_acc_topk + curr_eval_acc_topk/float(len(logits))
        eval_prec_topk = eval_prec_topk + curr_eval_prec_topk/float(len(logits))
        eval_rec_topk = eval_rec_topk + curr_eval_rec_topk/float(len(logits))
        eval_hamming_loss_topk = eval_hamming_loss_topk + curr_hamming_loss_topk/float(len(logits))
        #print("eval_acc", eval_acc)
        eval_loss=eval_loss+curr_eval_loss
        #eval_counter=eval_counter+1
        #print("eval_counter", eval_counter)
    
    #0. output raw prediction results (logits_all)
    if output_logits:
        #numpy.savetxt("pred_test.csv", logits_all, delimiter=",")
        label_list = [vocabulary_index2word_label[i] for i in range(len(vocabulary_index2word_label))]
        df_logits_all = pd.DataFrame(logits_all_sigmoid,columns=label_list)
        print('Output pred_test_run%s.csv' % str(num_run))
        df_logits_all.to_csv("%s pred_test_run%s.csv" % (output_logits_filename_prefix, str(num_run)))
        
    #1. example-based metrics
    eval_prec_th = eval_prec_th/float(eval_counter)
    eval_rec_th = eval_rec_th/float(eval_counter)
    eval_hamming_loss_th = eval_hamming_loss_th/float(eval_counter)
    if (eval_prec_th+eval_rec_th)>0:
        eval_fmeasure_th = 2*eval_prec_th*eval_rec_th/(eval_prec_th+eval_rec_th)
    
    eval_prec_topk = eval_prec_topk/float(eval_counter)
    eval_rec_topk = eval_rec_topk/float(eval_counter)
    eval_hamming_loss_topk = eval_hamming_loss_topk/float(eval_counter)
    if (eval_prec_topk+eval_rec_topk)>0:
        eval_fmeasure_topk = 2*eval_prec_topk*eval_rec_topk/(eval_prec_topk+eval_rec_topk)    
        
    #2. label-based matrics - micro and macro precision, recall, F-measure, AUC, 
    #'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
    #'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    # definitions above, see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    logits_binary = (logits_all_sigmoid>threshold).astype(float) # binary prediction matrix
    #print('logits_binary:',logits_binary)
    #logits_binary_np_array = np.array(logits_binary)
    #evalY = evalY[:len(logits_binary)] # adjust the number of evalY to those fully divided by the batch_size
    evalY_np_array = np.array(evalY)
    macro_accuracy, macro_precision, macro_recall, macro_f1 = all_macro(logits_binary,evalY_np_array)
    
    evalY_mic = evalY_np_array.ravel()
    logits_binary_mic = logits_binary.ravel()
    micro_accuracy, micro_precision, micro_recall, micro_f1 = all_micro(logits_binary_mic, evalY_mic)
    
    roc_auc = auc_metrics(logits_all_sigmoid, evalY_np_array, evalY_mic)
    macro_auc, micro_auc = roc_auc["auc_macro"], roc_auc["auc_micro"]    
    
    # diagnostic and procedural code results: for MIMIC-III datasets only (--dataset mimic3-ds or mimic3-ds-50)
    if 'mimic3-ds' in FLAGS.dataset:
        list_ind_diag,list_ind_proc = [],[]
        num_classes = len(vocabulary_index2word_label)
        #construct diagnose and procedural index
        for ind in range(num_classes):
            code_to_check = vocabulary_index2word_label[ind]
            code_type = check_code_type(code_to_check)
            if code_type == 'diag':
                list_ind_diag.append(ind)
            elif code_type == 'proc':
                list_ind_proc.append(ind)
            else:
                print('Error: neither diag or proc,', code_to_check)
        #get logits_binary_diag and logits_binary_proc
        logits_binary_diag = logits_binary[:,list_ind_diag]
        logits_binary_proc = logits_binary[:,list_ind_proc]
        #get evalY_np_array_diag and evalY_np_array_proc
        evalY_np_array_diag = evalY_np_array[:,list_ind_diag]
        evalY_np_array_proc = evalY_np_array[:,list_ind_proc]
        #calculate micro-averaging results
        _, micro_precision_diag, micro_recall_diag, micro_f1_diag = all_micro(logits_binary_diag.ravel(), evalY_np_array_diag.ravel())
        _, micro_precision_proc, micro_recall_proc, micro_f1_proc = all_micro(logits_binary_proc.ravel(), evalY_np_array_proc.ravel())
        #print('diag results,prec,rec,f1:',micro_precision_diag, micro_recall_diag, micro_f1_diag)
        #print('proc results,prec,rec,f1:',micro_precision_proc, micro_recall_proc, micro_f1_proc)
    else:
        micro_precision_diag, micro_recall_diag, micro_f1_diag, micro_precision_proc, micro_recall_proc, micro_f1_proc = 0.0,0.0,0.0,0.0,0.0,0.0
        
    #3. per-label metrics - threshold
    #logits_binary = (logits_all>threshold).astype(float)
    prec_per_label_th = metrics.precision_score(evalY,logits_binary,average=None)
    rec_per_label_th = metrics.recall_score(evalY,logits_binary,average=None)
    f1_per_label_th = metrics.f1_score(evalY,logits_binary,average=None)
    #per-label metrics - topk
    logits_binary_topk = get_topk_binary_using_logits_matrix(logits_all_sigmoid,top_number)
    #print('logits_binary_topk:',logits_binary_topk)
    prec_per_label_topk = metrics.precision_score(evalY,logits_binary_topk,average=None)
    rec_per_label_topk = metrics.recall_score(evalY,logits_binary_topk,average=None)
    f1_per_label_topk = metrics.f1_score(evalY,logits_binary_topk,average=None)
    return eval_loss/float(eval_counter),eval_acc_th/float(eval_counter),eval_prec_th,eval_rec_th,eval_fmeasure_th,eval_hamming_loss_th/hamming_q,prec_per_label_th,rec_per_label_th,f1_per_label_th,eval_acc_topk/float(eval_counter),eval_prec_topk,eval_rec_topk,eval_fmeasure_topk,eval_hamming_loss_topk/hamming_q,prec_per_label_topk,rec_per_label_topk,f1_per_label_topk, macro_accuracy, macro_precision, macro_recall, macro_f1, macro_auc, micro_accuracy, micro_precision, micro_recall, micro_f1, micro_auc, micro_precision_diag, micro_recall_diag, micro_f1_diag, micro_precision_proc, micro_recall_proc, micro_f1_proc

#to do: the two functions below could only predict the seeded examples, thus making them to run much faster.
def display_for_qualitative_evaluation(sess,modelToEval,label_sim_mat,label_sub_mat,evalX,evalY,batch_size,vocabulary_index2word,vocabulary_index2word_label,threshold=0.5,use_random_sampling=False):
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
        feed_dict = {modelToEval.input_x: evalX[start:end], modelToEval.dropout_keep_prob: 1, modelToEval.label_sim_matrix:label_sim_mat, modelToEval.label_sub_matrix:label_sub_mat}
        #if (start==0):
        #    print(evalX[start:end])
        if not FLAGS.multi_label_flag:
            feed_dict[modelToEval.input_y] = evalY[start:end]
        else:
            feed_dict[modelToEval.input_y_multilabel] = evalY[start:end]
        #curr_eval_loss, logits,curr_eval_acc= sess.run([modelToEval.loss_val,modelToEval.logits,modelToEval.accuracy],feed_dict)#curr_eval_acc--->modelToEval.accuracy
        #curr_eval_loss,logits= sess.run([modelToEval.loss_val,modelToEval.logits],feed_dict)#curr_eval_acc--->modelToEval.accuracy
        
        word_att,sent_att,curr_eval_loss,logits= sess.run([modelToEval.p_attention,modelToEval.p_attention_sent,modelToEval.loss_val,modelToEval.logits],feed_dict)
        word_att = np.reshape(word_att, (end-start,FLAGS.sequence_length))
        
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
                doc = 'doc-' + str(n_doc) + ': ' + ' '.join(display_results_with_word_att_sent_att(evalX[start+x],vocabulary_index2word,word_att[x],sent_att[x],'ori'))
                pred = 'prediction-0.5: ' + ' '.join(display_results(label_list_th,vocabulary_index2word_label))
                #print('prediction-topk:',*display_results(label_list_topk,vocabulary_index2word_label))
                get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
                label = 'labels: ' + ' '.join(display_results(get_indexes(1,evalY[start+x]),vocabulary_index2word_label))
                prediction_str = prediction_str + '\n' + doc + '\n' + pred + '\n' + label + '\n'
                #print(prediction_str)
                n_doc=n_doc+1
    return prediction_str

def display_for_qualitative_evaluation_per_label(sess,modelToEval,label_sim_mat,label_sub_mat,evalX,evalY,batch_size,vocabulary_index2word,vocabulary_index2word_label,threshold=0.5,use_random_sampling=False):
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
        feed_dict = {modelToEval.input_x: evalX[start:end], modelToEval.dropout_keep_prob: 1, modelToEval.label_sim_matrix:label_sim_mat, modelToEval.label_sub_matrix:label_sub_mat}
        #if (start==0):
        #    print(evalX[start:end])
        if not FLAGS.multi_label_flag:
            feed_dict[modelToEval.input_y] = evalY[start:end]
        else:
            feed_dict[modelToEval.input_y_multilabel] = evalY[start:end]
        #curr_eval_loss, logits,curr_eval_acc= sess.run([modelToEval.loss_val,modelToEval.logits,modelToEval.accuracy],feed_dict)#curr_eval_acc--->modelToEval.accuracy
        #curr_eval_loss,logits= sess.run([modelToEval.loss_val,modelToEval.logits],feed_dict)#curr_eval_acc--->modelToEval.accuracy
        
        word_att_per_label,sent_att_per_label,curr_eval_loss,logits= sess.run([modelToEval.p_attention,modelToEval.p_attention_sent,modelToEval.loss_val,modelToEval.logits],feed_dict)
        #print('word_att_per_label:',word_att_per_label.shape)
        num_classes = len(vocabulary_index2word_label)
        if not FLAGS.per_label_sent_only: # if also includes per-label word-level attention weights, there is a *dinstinct* word-level attention weight for each different label.
            #word_att_per_label: shape:[num_classes,batch_size*num_sentences,sequence_length_per_sentence]
            list_word_att_per_label = np.split(word_att_per_label,num_classes,axis=0) #print('list_word_att_per_label:',len(list_word_att_per_label),list_word_att_per_label[0].shape)
        else:
            #there is a *shared* word-level attention weight for any of the labels.
            #word_att_per_label: shape:[batch_size*num_sentences,sequence_length_per_sentence]
            word_att = np.reshape(word_att_per_label, (end-start,FLAGS.sequence_length))
        
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
                    if not FLAGS.per_label_sent_only: # if also includes per-label word-level attention weights
                        word_att = list_word_att_per_label[pred_label_index]
                        word_att = np.reshape(word_att, (end-start,FLAGS.sequence_length))
                    sent_att = list_sent_att_per_label[pred_label_index]
                    sent_att = np.reshape(sent_att, (end-start,FLAGS.num_sentences))
                    #print('word_att:',word_att)
                    #print('sent_att:',sent_att)
                    label_to_explain = vocabulary_index2word_label[pred_label_index]
                    docs = docs + 'doc-' + str(n_doc) + '-' + str(start+x) + '-' + label_to_explain + ': ' + ' '.join(display_results_with_word_att_sent_att(evalX[start+x],vocabulary_index2word,word_att[x],sent_att[x],'ori')) + '\n'
                    #formatting: "doc - the nth doc to be presented - the kth doc in the who testing set - the label to explain"
                    
                    #todo
                    #top-3 sentences: score
                    
                    #top-3 tokens (weighted by sentences): score
                    
                pred = 'prediction-0.5: ' + ' '.join(display_results(label_list_th,vocabulary_index2word_label))
                #print('prediction-topk:',*display_results(label_list_topk,vocabulary_index2word_label))
                get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
                label = 'labels: ' + ' '.join(display_results(get_indexes(1,evalY[start+x]),vocabulary_index2word_label))
                prediction_str = prediction_str + '\n' + docs + pred + '\n' + label + '\n'
                n_doc = n_doc + 1
                #print(prediction_str)
    return prediction_str
    
# display results with word-level attention weights and sentence-level attention weights
# this can be used for both display a sequence of words (with vocabulary_index2word) or a sequence of labels (with vocabulary_index2word_label, as below).
def display_results_with_word_att_sent_att(index_list,vocabulary_index2word_label,word_att,sent_att,att_note):
    label_list=[]
    count = 1
    #print('word_att is an empty str? ', word_att == '') # for testing
    #print('sent_att is an empty str? ', sent_att == '') # for testing                    
    for index in index_list:
        if index!=0: # this ensures that the padded values not being displayed.
            label=vocabulary_index2word_label[index]
            #label_list.append(label)
            if word_att != '': #FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
                label_list.append(label + '(' + str(round(word_att[count-1],3)) + ')')
            else:
                print('word_att as empty:',word_att)
                label_list.append(label)
        if count % (FLAGS.sequence_length/FLAGS.num_sentences) == 0:
            sent_index = int(count / (FLAGS.sequence_length/FLAGS.num_sentences))
            if sent_att != '':
                label_list.append('/s' + str(int(sent_index)) + '(' + att_note + '-' + str(round(sent_att[sent_index-1],2)) + ')/' + '\n')
            else:
                label_list.append('/s' + str(int(sent_index)) + '\n')
        count = count + 1
    return label_list
    
#visualise the label embedding
def viz_le(sess,model,vocabulary_index2word_label,vocabulary_word2index_label,list_query_labels):
    k=10+1 # top-k similar code to display, +1 as the query label is also there
    num_labels = len(vocabulary_index2word_label)
    sim_list_output = ""
    
    if FLAGS.use_label_embedding:
        sim_list_output = 'label embedding-final projected layer:\n'
        ###part 1: visualise the normalised label embedding for the final projected layer (this has dimension 400)
        #1. get normalised label embedidng
        word2vec_model_labels = Word2Vec.load(FLAGS.emb_model_path_mimic3_ds_init) # for gensim word2vec models
    
        word2vec_dict_labels = {}
        for _, label in enumerate(word2vec_model_labels.wv.vocab):
            word2vec_dict_labels[label] = word2vec_model_labels.wv[label]
        
        label_embedding_2dlist = [[]] * num_labels  # create an empty word_embedding list: which is a list of list, i.e. a list of word, where each word is a list of values as an embedding vector.
        bound = np.sqrt(6.0) / np.sqrt(num_labels + FLAGS.embed_size * 4)  # bound for random variables for Xavier initialisation.
        count_exist = 0;
        count_not_exist = 0
        for i in range(num_labels):  # loop over each label/class
            label = vocabulary_index2word_label[i]  # get a label
            embedding = None
            try:
                embedding = word2vec_dict_labels[label]  # try to get vector:it is an array.
            except Exception:
                embedding = None
            if embedding is not None:  # the 'label' exist a embedding
                label_embedding_2dlist[i] = embedding / float(np.linalg.norm(embedding) + 1e-6) # normalise to unit length
                count_exist = count_exist + 1  # assign array to this word.
            else:  # no embedding for this word
                print(label, 'embedding inexist')
                label_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size * 4); # dimensionality as the final hidden layer of the model, which is 4 times of the input embedding size.
                count_not_exist = count_not_exist + 1  # init a random value for the word.
        label_embedding_final = np.array(label_embedding_2dlist)  # covert to 2d array.
    
        #2. print top-k similar labels to a query label
        list_topk_label_lists = get_k_similar_label(label_embedding_final,list_query_labels,vocabulary_index2word_label,vocabulary_word2index_label,k)
        for query_label, topk_label_list in zip(list_query_labels,list_topk_label_lists):
            sim_list_output = sim_list_output + query_label + ":" + str(topk_label_list) + '\n'
        print(list_topk_label_lists)
        
        #3. plot 2D tsne scatter figure (not for the full label setting)
        if num_labels <= 50:
            code_emb_norm_2D = TSNE(n_components=2, init='pca', random_state=100).fit_transform(label_embedding_final)
            print(code_emb_norm_2D)
            plt.scatter(code_emb_norm_2D[:,0],code_emb_norm_2D[:,1])
            
            for i in range(num_labels):
                plt.annotate(vocabulary_index2word_label[i],(code_emb_norm_2D[:,0][i],code_emb_norm_2D[:,1][i]))
            plt.savefig('code_embedding_ori.png')
            #plt.show()
            plt.clf()
        
        ###part 2: visualise the normalised label embedding for the labelwise (per label) attention layer (this has dimension 200)
        if FLAGS.per_label_attention:
            sim_list_output = sim_list_output + 'label embedding-labelwise attention layer:\n'
            #1. get normalised label embedidng
            word2vec_model_labels = Word2Vec.load(FLAGS.emb_model_path_mimic3_ds_init_per_label) # for gensim word2vec models
        
            word2vec_dict_labels = {}
            for _, label in enumerate(word2vec_model_labels.wv.vocab):
                word2vec_dict_labels[label] = word2vec_model_labels.wv[label]
            
            label_embedding_2dlist = [[]] * num_labels  # create an empty word_embedding list: which is a list of list, i.e. a list of word, where each word is a list of values as an embedding vector.
            bound = np.sqrt(6.0) / np.sqrt(num_labels + FLAGS.embed_size * 4)  # bound for random variables for Xavier initialisation.
            count_exist = 0;
            count_not_exist = 0
            for i in range(num_labels):  # loop over each label/class
                label = vocabulary_index2word_label[i]  # get a label
                embedding = None
                try:
                    embedding = word2vec_dict_labels[label]  # try to get vector:it is an array.
                except Exception:
                    embedding = None
                if embedding is not None:  # the 'label' exist a embedding
                    label_embedding_2dlist[i] = embedding / float(np.linalg.norm(embedding) + 1e-6) # normalise to unit length
                    count_exist = count_exist + 1  # assign array to this word.
                else:  # no embedding for this word
                    print(label, 'embedding inexist')
                    label_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size * 4); # dimensionality as the final hidden layer of the model, which is 4 times of the input embedding size.
                    count_not_exist = count_not_exist + 1  # init a random value for the word.
            label_embedding_final = np.array(label_embedding_2dlist)  # covert to 2d array.
        
            #2. print top-k similar labels to a query label
            list_topk_label_lists = get_k_similar_label(label_embedding_final,list_query_labels,vocabulary_index2word_label,vocabulary_word2index_label,k)
            for query_label, topk_label_list in zip(list_query_labels,list_topk_label_lists):
                sim_list_output = sim_list_output + query_label + ":" + str(topk_label_list) + '\n'
            print(list_topk_label_lists)
            
            #3. plot 2D tsne scatter figure (not for the full label setting)
            if num_labels <= 50:
                code_emb_norm_2D = TSNE(n_components=2, init='pca', random_state=100).fit_transform(label_embedding_final)
                print(code_emb_norm_2D)
                plt.scatter(code_emb_norm_2D[:,0],code_emb_norm_2D[:,1])
                
                for i in range(num_labels):
                    plt.annotate(vocabulary_index2word_label[i],(code_emb_norm_2D[:,0][i],code_emb_norm_2D[:,1][i]))
                plt.savefig('code_embedding_ori_lwatt.png')
                #plt.show()
                plt.clf()
            
    #add header for model projection layer topk sim output:
    if FLAGS.use_label_embedding:
        sim_list_output = sim_list_output + 'model with LE:\n'
        sim_list_filename = 'model topk similar labels + LE.txt'
        fig_output_name = "final_projection_sim+LE.png"
    else:
        sim_list_output = sim_list_output + 'model without LE:\n'
        sim_list_filename = 'model topk similar labels.txt'
        fig_output_name = "final_projection_sim.png"
           
    ###part 3: visualise the final projection layer
    #1. get the layer weights to visualise
    #W_projection, context_vector_word_per_label, context_vector_sentence_per_label = sess.run([model.W_projection,model.context_vector_word_per_label, model.context_vector_sentence_per_label])
    W_projection = sess.run(model.W_projection)
    
    W_projection_tp = np.transpose(W_projection)
    print('W_projection_tp:', W_projection_tp.shape, W_projection_tp[0,0:5])
    
    #2. print top-k similar labels to a query label
    w_project_norm = W_projection_tp/np.linalg.norm(W_projection_tp,axis=1)[:,None]
    #print(get_k_similar_label(LE_learned,list_query_labels,dicts,k))
    list_topk_label_lists = get_k_similar_label(w_project_norm,list_query_labels,vocabulary_index2word_label,vocabulary_word2index_label,k)
    for query_label, topk_label_list in zip(list_query_labels,list_topk_label_lists):
        sim_list_output = sim_list_output + query_label + ":" + str(topk_label_list) + '\n'
    print(list_topk_label_lists)
    
    #2.5 output the topk labels to a file
    output_to_file(sim_list_filename,sim_list_output)
    
    #3. plot 2D tsne scatter figure (not for the full label setting)
    if num_labels <= 50:
        LE_learned_2D = TSNE(n_components=2, init='pca', random_state=100).fit_transform(w_project_norm)
        print(LE_learned_2D)
        plt.scatter(LE_learned_2D[:,0],LE_learned_2D[:,1])
        
        for i in range(num_labels):
            plt.annotate(vocabulary_index2word_label[i],(LE_learned_2D[:,0][i],LE_learned_2D[:,1][i]))
        plt.savefig(fig_output_name)
        #plt.show()
        plt.clf()
    
    ###part 4: visualise the label_wise_att_context_weight: word-level and sent-level
    if FLAGS.per_label_attention:
        if not FLAGS.per_label_sent_only:
            sim_list_output = sim_list_output + 'model label-wise word-level attention context matrix with LE:\n' if FLAGS.use_label_embedding else sim_list_output + 'model label-wise word-level attention context matrix without LE:\n'
            fig_output_name = fig_output_name.replace('final_projection','lw_att_word_level_context')
            
            context_vector_word_per_label = sess.run(model.context_vector_word_per_label)
            #context_vector_sentence_per_label = sess.run(model.context_vector_sentence_per_label)
            lw_att_word_lvl_context_norm = context_vector_word_per_label/np.linalg.norm(context_vector_word_per_label,axis=1)[:,None]
            #lw_att_context_norm = context_vector_sentence_per_label/np.linalg.norm(context_vector_sentence_per_label,axis=1)[:,None]
            list_topk_label_lists = get_k_similar_label(lw_att_word_lvl_context_norm,list_query_labels,vocabulary_index2word_label,vocabulary_word2index_label,k)
            for query_label, topk_label_list in zip(list_query_labels,list_topk_label_lists):
                sim_list_output = sim_list_output + query_label + ":" + str(topk_label_list) + '\n'
            print(list_topk_label_lists)
            
            #2.5 output the topk labels to a file
            output_to_file(sim_list_filename,sim_list_output)
            
            #3. plot 2D tsne scatter figure (not for the full label setting)
            if num_labels <= 50:
                lw_att_context_norm_2D = TSNE(n_components=2, init='pca', random_state=100).fit_transform(lw_att_word_lvl_context_norm)
                print(LE_learned_2D)
                plt.scatter(lw_att_context_norm_2D[:,0],lw_att_context_norm_2D[:,1])
                
                for i in range(num_labels):
                    plt.annotate(vocabulary_index2word_label[i],(lw_att_context_norm_2D[:,0][i],lw_att_context_norm_2D[:,1][i]))
                plt.savefig(fig_output_name)
                plt.clf()
                
        sim_list_output = sim_list_output + 'model label-wise sent-level attention context matrix with LE:\n' if FLAGS.use_label_embedding else sim_list_output + 'model label-wise sent-level attention context matrix without LE:\n'
        fig_output_name = fig_output_name.replace('lw_att_word_level_context','lw_att_sent_level_context')
        
        #context_vector_word_per_label = sess.run(model.context_vector_word_per_label)
        context_vector_sentence_per_label = sess.run(model.context_vector_sentence_per_label)
        #lw_att_context_norm = context_vector_word_per_label/np.linalg.norm(context_vector_word_per_label,axis=1)[:,None]
        lw_att_sent_lvl_context_norm = context_vector_sentence_per_label/np.linalg.norm(context_vector_sentence_per_label,axis=1)[:,None]
        list_topk_label_lists = get_k_similar_label(lw_att_sent_lvl_context_norm,list_query_labels,vocabulary_index2word_label,vocabulary_word2index_label,k)
        for query_label, topk_label_list in zip(list_query_labels,list_topk_label_lists):
            sim_list_output = sim_list_output + query_label + ":" + str(topk_label_list) + '\n'
        print(list_topk_label_lists)
        
        #2.5 output the topk labels to a file
        output_to_file(sim_list_filename,sim_list_output)
        
        #3. plot 2D tsne scatter figure (not for the full label setting)
        if num_labels <= 50:
            lw_att_context_norm_2D = TSNE(n_components=2, init='pca', random_state=100).fit_transform(lw_att_sent_lvl_context_norm)
            print(LE_learned_2D)
            plt.scatter(lw_att_context_norm_2D[:,0],lw_att_context_norm_2D[:,1])
            
            for i in range(num_labels):
                plt.annotate(vocabulary_index2word_label[i],(lw_att_context_norm_2D[:,0][i],lw_att_context_norm_2D[:,1][i]))
            plt.savefig(fig_output_name)
            plt.clf()
                
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
    
#get label using logits
def get_label_using_logits(logits,vocabulary_index2word_label,top_number=1):
    #print("get_label_using_logits.logits:",logits) #1-d array: array([-5.69036102, -8.54903221, -5.63954401, ..., -5.83969498,-5.84496021, -6.13911009], dtype=float32))
    index_list=np.argsort(logits)[-top_number:]
    index_list=index_list[::-1]
    #label_list=[]
    #for index in index_list:
    #    label=vocabulary_index2word_label[index]
    #    label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
    return index_list

# get those above threshold from the sigmoid(logits) values
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

def calculate_accuracy(labels_predicted,labels,eval_counter): # this should be same as the recall value
    # turn the multihot representation to a list of true labels
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    #if eval_counter<2:
        #print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    overlapping = 0
    label_dict = {x: x for x in label_nozero} # create a dictionary of labels for the true labels
    union = len(label_dict)
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            overlapping = overlapping + 1
        else:
            union = union + 1        
    return overlapping / union

def calculate_precision_recall(labels_predicted, labels,eval_counter):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    #if eval_counter<2:
    #    print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            count = count + 1
    if (len(labels_predicted)==0): # if nothing predicted, then set the precision as 0.
        precision=0
    else: 
        precision = count / len(labels_predicted)
    recall = count / len(label_nozero)
    #fmeasure = 2*precision*recall/(precision+recall)
    #print(count, len(label_nozero))
    return precision, recall
   
# calculate the symmetric_difference
def calculate_hamming_loss(labels_predicted, labels):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    count = 0
    label_dict = {x: x for x in label_nozero} # get the true labels

    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
        if flag is not None:
            count = count + 1 # get the number of overlapping labels
    
    return len(label_dict)+len(labels_predicted)-2*count

# calculate and output average ± standard deviation for results among folds or runs
# input: list_results_run can be a 1D list of a scalar metric of k runs or a 1D list of an array of metrics (e.g. per label) of k runs, where k can be kfold or running_time
#        with_min_max (optional), default False, by setting this True, further displays the minimum and maximum of all the folds or runs
# output: a string or a list of strings as mean±std. For 1D list of scalar input, the output is a single string; for a 1D list of one-dimensional np.ndarray input, the output is a list of j elements, where j is the dimension of the second axis (axis=1) of the input.
def cal_ave_std(list_result_runs, with_min_max=False):
    #list_result_runs is a list of 0-dimensional or 1-dimensional np.ndarrays #print(type(list_result_runs), list_result_runs[0],type(list_result_runs[0]))
    if np.ndim(list_result_runs[0]) == 1: # if being at least a list of one-dimensional np.ndarray
        if with_min_max:
            return ['%.3f ± %.3f (%.3f - %.3f)' % (results_mean, results_std, results_min, results_max) for results_mean,results_std,results_min, results_max in zip(np.mean(list_result_runs,axis=0), np.std(list_result_runs,axis=0), np.amin(list_result_runs,axis=0), np.amax(list_result_runs,axis=0))]
        else:
            return ['%.3f ± %.3f' % (results_mean, results_std) for results_mean,results_std in zip(np.mean(list_result_runs,axis=0), np.std(list_result_runs,axis=0))]
    else: # if 1D list # if isinstance(list_result_runs[0], float)
        assert np.ndim(list_result_runs[0]) == 0 # the element in the list is a scalar
        #print(np.mean(list_result_runs,axis=0), type(np.mean(list_result_runs,axis=0)))
        if with_min_max:
            return '%.3f ± %.3f (%.3f - %.3f)' % (np.mean(list_result_runs,axis=0), np.std(list_result_runs,axis=0), np.amin(list_result_runs,axis=0), np.amax(list_result_runs,axis=0))
        else:
            return '%.3f ± %.3f' % (np.mean(list_result_runs,axis=0), np.std(list_result_runs,axis=0))

######################################################################################################################            
# the code below is from https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py under the MIT license
######################################################################################################################            
def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)

def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic, ymic)

#########################################################################
#MACRO METRICS: calculate metric for each label and average across labels
#########################################################################

def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)

def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1
    
##########################################################################
#MICRO METRICS: treat every prediction as an individual binary prediction
##########################################################################

def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)

def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

# calculate AUC metrics
def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    #get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        #only if there are true positives for this label
        if y[:,i].sum() > 0: # if the label has a true instance in the data
            fpr[i], tpr[i], _ = metrics.roc_curve(y[:,i], yhat_raw[:,i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = metrics.auc(fpr[i], tpr[i])
                if not np.isnan(auc_score): 
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    #macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    roc_auc['auc_macro'] = np.mean(aucs)

    #micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(ymic, yhatmic) 
    roc_auc["auc_micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    return roc_auc

def union_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)
    #numpy.logical_and(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'logical_and'>
    #Compute the truth value of x1 AND x2 element-wise.

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
       
if __name__ == "__main__":
    tf.app.run()
