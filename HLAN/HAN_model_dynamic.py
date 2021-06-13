# -*- coding: utf-8 -*-
# HierarchicalAttention: 1.Word Encoder. 2.Word Attention. 3.Sentence Encoder 4.Sentence Attention 5.linear classifier. 2017-06-13
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
import tensorflow.contrib as tf_contrib

class HAN:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length, num_sentences, vocab_size, embed_size, hidden_size, is_training, lambda_sim=0.00001, lambda_sub=0, dynamic_sem=False,dynamic_sem_l2=False,per_label_attention=False,per_label_sent_only=False,need_sentence_level_attention_encoder_flag=True, multi_label_flag=True, initializer= tf.random_normal_initializer(stddev=0.1),clip_gradients=5.0,verbose=True):#0.01 # tf.random_normal_initializer(mean=0.0,stddev=0.1,seed=1)
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_sentences = num_sentences
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5) # using assign to half the learning_rate
        self.initializer = initializer
        self.multi_label_flag = multi_label_flag
        self.hidden_size = hidden_size
        self.need_sentence_level_attention_encoder_flag = need_sentence_level_attention_encoder_flag
        self.clip_gradients=clip_gradients
        self.lambda_sim=lambda_sim
        self.lambda_sub=lambda_sub
        self.dynamic_sem = dynamic_sem
        self.dynamic_sem_l2 = dynamic_sem_l2
        self.per_label_attention = per_label_attention
        self.per_label_sent_only = per_label_sent_only
        
        # add placeholder (X,label)
        #self.input_x = tf.placeholder(tf.int32, [None, self.num_sentences,self.sequence_length], name="input_x")  # X
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        # As can be seen, placeholder with [] shape takes a single scalar value directly. Placeholder with [None] shape takes a 1-dimensional array and placeholder with None shape can take in any value while computation takes place. see https://stackoverflow.com/questions/46940857/what-is-the-difference-between-none-none-and-for-the-shape-of-a-placeh

        self.sequence_length = int(self.sequence_length / self.num_sentences)
        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")  # y:[None,num_classes]
        self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.num_classes],name="input_y_multilabel")  # y:[None,num_classes]. this is for multi-label classification only.
        #self.actual_batch_size = tf.placeholder(tf.int32,name="actual_batch_size") # to avoid the actual, last batch of the dataset which is smaller than the batch size, due to not fully divided.
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        #self.label_sim_matrix = tf.placeholder(tf.float32, [self.num_classes,self.num_classes],name="label_sim_mat")
        #self.label_sub_matrix = tf.placeholder(tf.float32, [self.num_classes,self.num_classes],name="label_sub_mat")
        self.label_sim_matrix_static = tf.placeholder(tf.float32, [self.num_classes,self.num_classes],name="label_sim_mat_const")
        self.label_sub_matrix_static = tf.placeholder(tf.float32, [self.num_classes,self.num_classes],name="label_sub_mat_const")
        if self.dynamic_sem == False:
            self.label_sim_matrix = self.label_sim_matrix_static
            self.label_sub_matrix = self.label_sub_matrix_static
        #print('self.dynamic_sem:',self.dynamic_sem)
        
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        #print('self.label_sim_matrix:',self.label_sim_matrix)
        #print('self.label_sub_matrix:',self.label_sub_matrix)
        
        if verbose:
            print('display trainable variables')
            for v in tf.trainable_variables():
                print(v)

        if self.per_label_attention:
            self.logits = self.inference_per_label() #[None, self.label_size]. main computation graph is here. # this function itself can handle the case of both word and sentence-level per-label attention or sentence-level only per-label attetnion (defined in self.per_label_sent_only).
        else:
            self.logits = self.inference() #[None, self.label_size]. main computation graph is here.
        
        if not is_training:
            return
        if multi_label_flag:
            #print("going to use multi label loss.")
            if self.lambda_sim == 0:
                if self.lambda_sub == 0:
                    # none
                    self.loss_val = self.loss_multilabel() # without any semantic regularisers, no L_sim or L_sub
                else:
                    # using L_sub only
                    #self.loss_val = self.loss_multilabel_onto_new_sub_per_batch(self.label_sub_matrix); # j,k per batch - used in the NAACL paper
                    self.loss_val = self.loss_multilabel_onto_new_sub_per_doc(self.label_sub_matrix,dynamic_sem_l2=self.dynamic_sem_l2); # j,k per document
            else:
                if self.lambda_sub == 0:
                    # using L_sim only
                    #pair_diff_squared on s_d
                    #self.loss_val = self.loss_multilabel_onto_new_sim_per_batch(self.label_sim_matrix) # j,k per batch - used in the NAACL paper
                    #self.loss_val = self.loss_multilabel_onto_new_sim_per_doc_tensor(self.label_sim_matrix) # j,k per document - tensor operations - requiring large GPU memory
                    #self.loss_val = self.loss_multilabel_onto_new_sim_per_doc_not_used(self.label_sim_matrix) # j,k per document - with for loop - requiring large GPU memory 
                    self.loss_val = self.loss_multilabel_onto_new_sim_per_doc(self.label_sim_matrix,dynamic_sem_l2=self.dynamic_sem_l2) # j,k per document - with for loop
                    
                    #pair_diff_abs on rounded s_d
                    #self.loss_val = self.loss_multilabel_onto_new_sim_pair_diff_abs(self.label_sim_matrix) # j,k per document - new sim pair_diff_abs
                else:
                    # sim+sub
                    #self.loss_val = self.loss_multilabel_onto_new_simsub_per_batch(self.label_sim_matrix,self.label_sub_matrix) # j,k per batch - used in the NAACL paper
                    self.loss_val = self.loss_multilabel_onto_new_simsub_per_doc(self.label_sim_matrix,self.label_sub_matrix,dynamic_sem_l2=self.dynamic_sem_l2) # j,k per document
                    #self.loss_val = self.loss_multilabel_onto_new_simsub_pair_diff_abs(self.label_sim_matrix,self.label_sub_matrix) # j,k per document, l_sim pair_diff_abs
        else:
            #print("going to use single label loss.")
            self.loss_val = self.loss()
        self.train_op = self.train()

        # output evaluation results on training data
        sig_output = tf.sigmoid(self.logits)
        if not self.multi_label_flag:
            self.predictions = tf.argmax(sig_output, axis=1, name="predictions")  # shape:[None,]
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32),
                                          self.input_y)  # tf.argmax(self.logits, 1)-->[batch_size]
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
            self.precision = 0
            self.recall = 0
            #self.f_measure = 0
        else:
            self.predictions = tf.round(sig_output) #y = sign(x) = -1 if x < 0; 0 if x == 0 or tf.is_nan(x); 1 if x > 0.
            #self.predictions = tf.cast(tf.greater(self.sig_logits,0.25),tf.float32)
            temp = tf.cast(tf.equal(self.predictions,self.input_y_multilabel), tf.float32)
            tp = tf.reduce_sum(tf.multiply(temp,self.predictions), axis=1)  # [128,1]
            p = tf.reduce_sum(self.predictions, axis=1) + 1e-10 # [128,1]
            t = tf.reduce_sum(self.input_y_multilabel, axis=1) # [128,1]
            union = tf.reduce_sum(tf.cast(tf.greater(self.predictions + self.input_y_multilabel,0),tf.float32), axis=1) # [128,1]
            self.accuracy = tf.reduce_mean(tf.div(tp,union))
            self.precision = tf.reduce_mean(tf.div(tp,p))
            self.recall = tf.reduce_mean(tf.div(tp,t))
        
        self.training_loss = tf.summary.scalar("train_loss_per_batch",self.loss_val)
        self.training_loss_per_epoch = tf.summary.scalar("train_loss_per_epoch",self.loss_val)
        self.validation_loss = tf.summary.scalar("validation_loss_per_batch",self.loss_val)
        self.validation_loss_per_epoch = tf.summary.scalar("validation_loss_per_epoch",self.loss_val)
        self.writer = tf.summary.FileWriter("./logs")
        
        if verbose:
            print('Model initialisation completed.')
    
    # need to check carefully: to avoid non-use weights.
    def instantiate_weights(self): # this is problematic here, the name_scope actually does not affect get_variable.
        """define all weights here"""
        with tf.name_scope("embedding_projection"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 4, self.num_classes],
                                                initializer=self.initializer)  # [embed_size,label_size] # this parameter matrix will be initialised if choosing FLAS.use_label_embedding
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])  #TODO [label_size]
            
            if self.dynamic_sem == True:
                #print('intialise dynamic sem loss weights')
                if self.lambda_sim != 0:
                    self.label_sim_matrix = tf.get_variable("label_sim_mat", shape=[self.num_classes, self.num_classes], initializer=self.initializer)
                    if self.lambda_sub == 0:
                        self.label_sub_matrix = self.label_sub_matrix_static # as static weights
                    else:
                        self.label_sub_matrix = tf.get_variable("label_sub_mat", shape=[self.num_classes, self.num_classes], initializer=self.initializer)
                else:
                    self.label_sim_matrix = self.label_sim_matrix_static # as static weights
                    if self.lambda_sub == 0:
                        self.label_sub_matrix = self.label_sub_matrix_static # as static weights
                    else:
                        self.label_sub_matrix = tf.get_variable("label_sub_mat", shape=[self.num_classes, self.num_classes], initializer=self.initializer)
                        
        # GRU parameters:update gate related
        with tf.name_scope("gru_weights_word_level"):
            self.W_z = tf.get_variable("W_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_z = tf.get_variable("U_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_z = tf.get_variable("b_z", shape=[self.hidden_size])
            # GRU parameters:reset gate related
            self.W_r = tf.get_variable("W_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_r = tf.get_variable("U_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_r = tf.get_variable("b_r", shape=[self.hidden_size])

            self.W_h = tf.get_variable("W_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_h = tf.get_variable("U_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_h = tf.get_variable("b_h", shape=[self.hidden_size])

        with tf.name_scope("gru_weights_sentence_level"):
            self.W_z_sentence = tf.get_variable("W_z_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.U_z_sentence = tf.get_variable("U_z_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.b_z_sentence = tf.get_variable("b_z_sentence", shape=[self.hidden_size * 2])
            # GRU parameters:reset gate related
            self.W_r_sentence = tf.get_variable("W_r_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.U_r_sentence = tf.get_variable("U_r_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.b_r_sentence = tf.get_variable("b_r_sentence", shape=[self.hidden_size * 2])

            self.W_h_sentence = tf.get_variable("W_h_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.U_h_sentence = tf.get_variable("U_h_sentence", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                initializer=self.initializer)
            self.b_h_sentence = tf.get_variable("b_h_sentence", shape=[self.hidden_size * 2])

        with tf.name_scope("attention"):
            self.W_w_attention_word = tf.get_variable("W_w_attention_word",
                                                      shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                      initializer=self.initializer)
            self.W_b_attention_word = tf.get_variable("W_b_attention_word", shape=[self.hidden_size * 2])

            self.W_w_attention_sentence = tf.get_variable("W_w_attention_sentence",
                                                          shape=[self.hidden_size * 4, self.hidden_size * 2],
                                                          initializer=self.initializer)
            self.W_b_attention_sentence = tf.get_variable("W_b_attention_sentence", shape=[self.hidden_size * 2])
            
            if self.per_label_attention:
                if not self.per_label_sent_only: # per_label_attention at both word level and sentence level
                    self.context_vector_word_per_label = tf.get_variable("what_is_the_informative_word_per_label", shape=[self.num_classes,self.hidden_size * 2],initializer=self.initializer) # this parameter matrix will be initialised if choosing FLAS.use_label_embedding
                else: # per_label_attention only at the sentence level
                    self.context_vector_word = tf.get_variable("what_is_the_informative_word", shape=[self.hidden_size * 2],initializer=self.initializer)
                    
                self.context_vector_sentence_per_label = tf.get_variable("what_is_the_informative_sentence_per_label", shape=[self.num_classes,self.hidden_size * 2],initializer=self.initializer) # this parameter matrix will be initialised if choosing FLAS.use_label_embedding
                    
            else:
                self.context_vector_word = tf.get_variable("what_is_the_informative_word", shape=[self.hidden_size * 2],initializer=self.initializer)
                self.context_vector_sentence = tf.get_variable("what_is_the_informative_sentence",shape=[self.hidden_size * 2], initializer=self.initializer)
                
    def attention_word_level(self, hidden_state):
        """
        input1:self.hidden_state: hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        input2:sentence level context vector:[batch_size*num_sentences,hidden_size*2]
        :return:representation.shape:[batch_size*num_sentences,hidden_size*2]
        """
        hidden_state_ = tf.stack(hidden_state, axis=1)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2] #self.hidden_state is a list.
        # using tf.stack to stack a list to a tensor.
        
        # 0) one layer of feed forward network
        hidden_state_2 = tf.reshape(hidden_state_, shape=[-1,
                                                          self.hidden_size * 2])  # shape:[batch_size*num_sentences*sequence_length,hidden_size*2]
        # hidden_state_:[batch_size*num_sentences*sequence_length,hidden_size*2];W_w_attention_sentence:[,hidden_size*2,,hidden_size*2]
        #print('hidden_state_2', hidden_state_2.get_shape()) # hidden_state_2 (32256, 200)
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                     self.W_w_attention_word) + self.W_b_attention_word)  # shape:[batch_size*num_sentences*sequence_length,hidden_size*2]
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.sequence_length,
                                                                         self.hidden_size * 2])  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        #print('hidden_representation', hidden_representation.get_shape()) # hidden_representation (512, 63, 200)
        # equation (5) in the original paper                                                                 
        # attention process:1.get logits for each word in the sentence. 2.get possibility distribution for each word in the sentence. 3.get weighted sum for the sentence as sentence representation.
        # 1) get logits for each word in the sentence.
        hidden_state_context_similiarity = tf.multiply(hidden_representation,
                                                       self.context_vector_word)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2] # element-wise multiplication between a tensor and a matrix (vector)
        #print('self.context_vector_word', self.context_vector_word.get_shape())                                                          
        #print('hidden_state_context_similiarity', hidden_state_context_similiarity.get_shape()) # hidden_state_context_similiarity (512, 63, 200)
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                         axis=2)  # shape:[batch_size*num_sentences,sequence_length]
        # the above calculated the U_it*Uw                                 
        # subtract max for numerical stability (softmax is shift invariant). tf.reduce_max:Computes the maximum of elements across dimensions of a tensor.
        attention_logits_max = tf.reduce_max(attention_logits, axis=1,
                                             keep_dims=True)  # shape:[batch_size*num_sentences,1]
        # 2) get possibility distribution for each word in the sentence.
        self.p_attention = tf.nn.softmax(
            attention_logits - attention_logits_max)  # shape:[batch_size*num_sentences,sequence_length]
        # equation (6)
        # 3) get weighted hidden state by attention vector
        p_attention_expanded = tf.expand_dims(self.p_attention, axis=2)  # shape:[batch_size*num_sentences,sequence_length,1]
        # below sentence_representation'shape:[batch_size*num_sentences,sequence_length,hidden_size*2]<----p_attention_expanded:[batch_size*num_sentences,sequence_length,1];hidden_state_:[batch_size*num_sentences,sequence_length,hidden_size*2]
        sentence_representation = tf.multiply(p_attention_expanded,
                                              hidden_state_)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        sentence_representation = tf.reduce_sum(sentence_representation,
                                                axis=1)  # shape:[batch_size*num_sentences,hidden_size*2]
        # equation (7)                                        
        return sentence_representation  # shape:[batch_size*num_sentences,hidden_size*2]
    
    # a per-label version of the word-level attention weights
    def attention_word_level_per_label(self, hidden_state):
        """
        input1:self.hidden_state: hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        input2:sentence level context vector:[batch_size*num_sentences,hidden_size*2]
        :return:representation.shape:[num_classes,batch_size*num_sentences,hidden_size*2]
        """
        hidden_state_ = tf.stack(hidden_state, axis=1)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2] #self.hidden_state is a list.
        # using tf.stack to stack a list to a tensor.
        
        # 0) one layer of feed forward network
        hidden_state_2 = tf.reshape(hidden_state_, shape=[-1,
                                                          self.hidden_size * 2])  # shape:[batch_size*num_sentences*sequence_length,hidden_size*2]
        # hidden_state_:[batch_size*num_sentences*sequence_length,hidden_size*2];W_w_attention_sentence:[,hidden_size*2,,hidden_size*2]
        #print('hidden_state_2', hidden_state_2.get_shape()) # hidden_state_2 (32256, 200)
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                     self.W_w_attention_word) + self.W_b_attention_word)  # shape:[batch_size*num_sentences*sequence_length,hidden_size*2]
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.sequence_length,
                                                                         self.hidden_size * 2])  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        #print('hidden_representation', hidden_representation.get_shape()) # hidden_representation (512, 63, 200)
        # equation (5) in the original paper                                                                 
        # attention process:1.get logits for each word in the sentence. 2.get possibility distribution for each word in the sentence. 3.get weighted sum for the sentence as sentence representation.
        #context_vector_word_per_label_unstacked = tf.unstack(self.context_vector_word_per_label,axis=1) 
        #sentence_representation_per_label = []
        #self.p_attention = [0]*self.num_classes
        #n=0
        # 1) get logits for each word in the sentence.
        hidden_representation_expanded = tf.expand_dims(hidden_representation,axis=0)
        context_vector_word_per_label_expanded = tf.expand_dims(tf.expand_dims(self.context_vector_word_per_label,axis=1),axis=1)
        hidden_state_context_similiarity = tf.multiply(hidden_representation_expanded,
                                                   context_vector_word_per_label_expanded)  # shape:[num_classes,batch_size*num_sentences,sequence_length,hidden_size*2] # element-wise multiplication between a tensor and a matrix
        #print('self.context_vector_word',self.context_vector_word.get_shape())                       
        #print('hidden_state_context_similiarity', hidden_state_context_similiarity.get_shape()) # hidden_state_context_similiarity (512, 63, 200, 50)
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                     axis=3)  # shape:[num_classes,batch_size*num_sentences,sequence_length]
        # the above calculated the U_it*Uw                                 
        # subtract max for numerical stability (softmax is shift invariant). tf.reduce_max:Computes the maximum of elements across dimensions of a tensor.
        attention_logits_max = tf.reduce_max(attention_logits, axis=2,
                                         keep_dims=True)  # shape:[num_classes,batch_size*num_sentences,1]
        # 2) get possibility distribution for each word in the sentence.
        self.p_attention = tf.nn.softmax(
            attention_logits - attention_logits_max)  # shape:[num_classes,batch_size*num_sentences,sequence_length]
        # equation (6)
        # 3) get weighted hidden state by attention vector
        p_attention_expanded = tf.expand_dims(self.p_attention, axis=3)  # shape:[num_classes,batch_size*num_sentences,sequence_length,1]
        # below sentence_representation'shape:[num_classes,batch_size*num_sentences,sequence_length,hidden_size*2]<----p_attention_expanded:[num_classes,batch_size*num_sentences,sequence_length,1];hidden_state_:[batch_size*num_sentences,sequence_length,hidden_size*2]
        sentence_representation = tf.multiply(p_attention_expanded,
                                          hidden_state_)  # shape:[num_classes,batch_size*num_sentences,sequence_length,hidden_size*2]
        sentence_representation = tf.reduce_sum(sentence_representation,
                                            axis=2)  # shape:[num_classes,batch_size*num_sentences,hidden_size*2]
        # equation (7)  
        #sentence_representation_per_label.append(sentence_representation)
        #n=n+1
        #return sentence_representation_per_label  # shape:[batch_size*num_sentences,hidden_size*2]
        return sentence_representation
        
    def attention_sentence_level(self, hidden_state_sentence):
        """
        input1: hidden_state_sentence: a list,len:num_sentence,element:[None,hidden_size*4]
        input2: sentence level context vector:[self.hidden_size*2]
        :return:representation.shape:[None,hidden_size*4]
        """
        hidden_state_ = tf.stack(hidden_state_sentence, axis=1)  # shape:[None,num_sentence,hidden_size*4]

        # 0) one layer of feed forward
        hidden_state_2 = tf.reshape(hidden_state_,
                                    shape=[-1, self.hidden_size * 4])  # [None*num_sentence,hidden_size*4]
                       # tf.reshape(tensor,shape,name=None)                            
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                     self.W_w_attention_sentence) + self.W_b_attention_sentence)  # shape:[None*num_sentence,hidden_size*2]
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.num_sentences,
                                                                         self.hidden_size * 2])  # [None,num_sentence,hidden_size*2]
        # attention process:1.get logits for each sentence in the doc.2.get possibility distribution for each sentence in the doc.3.get weighted sum for the sentences as doc representation.
        # 1) get logits for each word in the sentence.
        hidden_state_context_similiarity = tf.multiply(hidden_representation,
                                                       self.context_vector_sentence)  # shape:[None,num_sentence,hidden_size*2]
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                         axis=2)  # shape:[None,num_sentence]. that is get logit for each num_sentence.
        # subtract max for numerical stability (softmax is shift invariant). tf.reduce_max:computes the maximum of elements across dimensions of a tensor.
        attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)  # shape:[None,1]
        # 2) get possibility distribution for each word in the sentence.
        self.p_attention_sent = tf.nn.softmax(attention_logits - attention_logits_max)  # shape:[None,num_sentence]
        # 3) get weighted hidden state by attention vector(sentence level)
        p_attention_expanded = tf.expand_dims(self.p_attention_sent, axis=2)  # shape:[None,num_sentence,1]
        document_representation = tf.multiply(p_attention_expanded,
                                              hidden_state_)  # shape:[None,num_sentence,hidden_size*4]<---p_attention_expanded:[None,num_sentence,1];hidden_state_:[None,num_sentence,hidden_size*4]
        document_representation = tf.reduce_sum(document_representation, axis=1)  # shape:[None,hidden_size*4]
        #print('document_representation in attention_sentence_level',document_representation.get_shape()) # document_representation in attention_sentence_level (128, 400)
        return document_representation  # shape:[None,hidden_size*4]
    
    def attention_sentence_level_per_label(self, hidden_state_sentence):
        """
        input1: hidden_state_sentence: a list,len:num_sentence,element:[num_classes,None,hidden_size*4] or [None,hidden_size*4]
        input2 (in self): sentence level context vector:[num_classes,self.hidden_size*2]
        :return:representation.shape:[num_classes,None,hidden_size*4]
        """
        if not self.per_label_sent_only: 
            # the word-level attention weight is also per-label.
            # hidden_state_sentence a list of num_sentence [num_classes,None,hidden_size*4]
            hidden_state_ = tf.stack(hidden_state_sentence, axis=2)  # shape:[num_classes,None,num_sentence,hidden_size*4]
            # 0) one layer of feed forward
            hidden_state_2 = tf.reshape(hidden_state_,
                                        shape=[self.num_classes,-1, self.hidden_size * 4])  # [num_classes,None*num_sentence,hidden_size*4]
                           # tf.reshape(tensor,shape,name=None)                            
            W_w_attention_sentence = tf.tile(tf.expand_dims(self.W_w_attention_sentence, axis=0),[self.num_classes,1,1])
            hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                         W_w_attention_sentence) + self.W_b_attention_sentence)  # shape:[num_classes,None*num_sentence,hidden_size*2], transformed from hidden_size*4
            hidden_representation = tf.reshape(hidden_representation, shape=[self.num_classes,-1, self.num_sentences,self.hidden_size * 2])  # [num_classes,None,num_sentence,hidden_size*2]
        else:
            # the word-level attention weight is shared for all labels.
            # hidden_state_sentence a list of num_sentence [None,hidden_size*4]
            hidden_state_ = tf.stack(hidden_state_sentence, axis=1)  # shape:[num_classes,None,num_sentence,hidden_size*4]
            # 0) one layer of feed forward
            hidden_state_2 = tf.reshape(hidden_state_,
                                        shape=[-1, self.hidden_size * 4])  # [None*num_sentence,hidden_size*4]
                           # tf.reshape(tensor,shape,name=None)                            
            hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                         self.W_w_attention_sentence) + self.W_b_attention_sentence)  # shape:[None*num_sentence,hidden_size*2]
            hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.num_sentences,self.hidden_size * 2])  # [None,num_sentence,hidden_size*2]
            hidden_representation = tf.expand_dims(hidden_representation,axis=0)
        # attention process:1.get logits for each sentence in the doc.2.get possibility distribution for each sentence in the doc.3.get weighted sum for the sentences as doc representation.
        # 1) get logits for each word in the sentence.
        context_vector_sentence_per_label_expanded = tf.expand_dims(tf.expand_dims(self.context_vector_sentence_per_label,axis=1),axis=1)
        #context_vector_sentence_per_label_expanded = tf.expand_dims(tf.expand_dims(self.context_vector_word_per_label,axis=1),axis=1) #sharing the context_vector_word_per_label
        hidden_state_context_similiarity = tf.multiply(hidden_representation,
                                                       context_vector_sentence_per_label_expanded)  # shape:[num_classes,None,num_sentence,hidden_size*2]
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                         axis=3)  # shape:[num_classes,None,num_sentence]. that is get logit for each num_sentence.
        # subtract max for numerical stability (softmax is shift invariant). tf.reduce_max:computes the maximum of elements across dimensions of a tensor.
        attention_logits_max = tf.reduce_max(attention_logits, axis=2, keep_dims=True)  # shape:[num_classes,None,1]
        # 2) get possibility distribution for each word in the sentence.
        self.p_attention_sent = tf.nn.softmax(attention_logits - attention_logits_max)  # shape:[num_classes,None,num_sentence]
        # 3) get weighted hidden state by attention vector(sentence level)
        p_attention_expanded = tf.expand_dims(self.p_attention_sent, axis=3)  # shape:[num_classes,None,num_sentence,1]
        document_representation = tf.multiply(p_attention_expanded,
                                              hidden_state_)  # shape:[num_classes,None,num_sentence,hidden_size*4]<---p_attention_expanded:[num_classes,None,num_sentence,1];hidden_state_:[num_classes,None,num_sentence,hidden_size*4]
        document_representation = tf.reduce_sum(document_representation, axis=2)  # shape:[num_classes,None,hidden_size*4]
        #print('document_representation in attention_sentence_level',document_representation.get_shape()) # document_representation in attention_sentence_level (128, 400)
        return document_representation  # shape:[num_classes,None,hidden_size*4]
    
    #unused, this is considered in the "def attention_sentence_level_per_label(self, hidden_state_sentence)" above.
    def attention_sentence_level_per_label_sent_only(self, hidden_state_sentence):
        """
        input1: hidden_state_sentence: a list,len:num_sentence,element:[None,hidden_size*4]
        input2: sentence level context vector:[self.hidden_size*2]
        :return:representation.shape:[None,hidden_size*4]
        """
        hidden_state_ = tf.stack(hidden_state_sentence, axis=1)  # shape:[None,num_sentence,hidden_size*4]

        # 0) one layer of feed forward
        hidden_state_2 = tf.reshape(hidden_state_,
                                    shape=[-1, self.hidden_size * 4])  # [None*num_sentence,hidden_size*4]
                       # tf.reshape(tensor,shape,name=None)                            
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2,
                                                     self.W_w_attention_sentence) + self.W_b_attention_sentence)  # shape:[None*num_sentence,hidden_size*2]
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.num_sentences,
                                                                         self.hidden_size * 2])  # [None,num_sentence,hidden_size*2]
        # attention process:1.get logits for each sentence in the doc.2.get possibility distribution for each sentence in the doc.3.get weighted sum for the sentences as doc representation.
        # 1) get logits for each word in the sentence.
        hidden_representation_expanded = tf.expand_dims(hidden_representation,axis=0)
        context_vector_sentence_per_label_expanded = tf.expand_dims(tf.expand_dims(self.context_vector_sentence_per_label,axis=1),axis=1)
        hidden_state_context_similiarity = tf.multiply(hidden_representation_expanded,context_vector_sentence_per_label_expanded)  # shape:[num_classes,None,num_sentence,hidden_size*2]
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                         axis=3)  # shape:[num_classes,None,num_sentence]. that is get logit for each num_sentence.
        # subtract max for numerical stability (softmax is shift invariant). tf.reduce_max:computes the maximum of elements across dimensions of a tensor.
        attention_logits_max = tf.reduce_max(attention_logits, axis=2, keep_dims=True)  # shape:[num_classes,None,1]
        # 2) get possibility distribution for each word in the sentence.
        self.p_attention_sent = tf.nn.softmax(attention_logits - attention_logits_max)  # shape:[num_classes,None,num_sentence]
        # 3) get weighted hidden state by attention vector(sentence level)
        p_attention_expanded = tf.expand_dims(self.p_attention_sent, axis=3)  # shape:[num_classes,None,num_sentence,1]
        document_representation = tf.multiply(p_attention_expanded,
                                              hidden_state_)  # shape:[num_classes,None,num_sentence,hidden_size*4]<---p_attention_expanded:[num_classes,None,num_sentence,1];hidden_state_:[num_classes,None,num_sentence,hidden_size*4]
        document_representation = tf.reduce_sum(document_representation, axis=2)  # shape:[num_classes,None,hidden_size*4]
        print('document_representation in attention_sentence_level',document_representation.get_shape()) # document_representation in attention_sentence_level (128, 400)
        return document_representation  # shape:[num_classes,None,hidden_size*4]
        
    def inference(self):
        """main computation graph here: 1.Word Encoder. 2.Word Attention. 3.Sentence Encoder 4.Sentence Attention 5.linear classifier"""
        # 1.Word Encoder
        # 1.1 embedding of words
        #print('before spliting', self.input_x.get_shape()) #shape (?,252)
        input_x = tf.split(self.input_x, self.num_sentences,axis=1)  # a list. length:num_sentences.each element is:[None,self.sequence_length/num_sentences]
        #print('before stacking', input_x.get_shape())
        input_x = tf.stack(input_x, axis=1)  # shape:[None,self.num_sentences,self.sequence_length/num_sentences]
        #print('after stacking', input_x.get_shape()) # shape (?,4,63)
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,input_x)  # [None,num_sentences,sentence_length,embed_size]
        #print('after embedding_lookup', self.embedded_words.get_shape()) # shape (?,4,63)
        embedded_words_reshaped = tf.reshape(self.embedded_words, shape=[-1, self.sequence_length,self.embed_size])  # [batch_size*num_sentences,sentence_length,embed_size]
        #print('after reshaping', embedded_words_reshaped.get_shape()) # shape (?,4,63)
        
        #before spliting (?, 252)
        #after stacking (?, 4, 63)
        #after embedding_lookup (?, 4, 63, 100)
        #after reshaping (?, 63, 100) [batch_size*num_sentences,sentence_length,embed_size]

        # 1.2 forward gru
        hidden_state_forward_list = self.gru_forward_word_level(embedded_words_reshaped)  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        # 1.3 backward gru
        hidden_state_backward_list = self.gru_backward_word_level(embedded_words_reshaped)  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        # 1.4 concat forward hidden state and backward hidden state. hidden_state: a list.len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        self.hidden_state = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in
                             zip(hidden_state_forward_list, hidden_state_backward_list)]  # hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
                             #self.hidden_state is a list.
                             
        # 2.Word Attention
        # for each sentence.
        sentence_representation = self.attention_word_level(self.hidden_state)  # output:[batch_size*num_sentences,hidden_size*2]
        sentence_representation = tf.reshape(sentence_representation, shape=[-1, self.num_sentences, self.hidden_size * 2])  # shape:[batch_size,num_sentences,hidden_size*2]
        #with tf.name_scope("dropout"):#TODO
        #    sentence_representation = tf.nn.dropout(sentence_representation,keep_prob=self.dropout_keep_prob)  # shape:[None,hidden_size*4]

        # 3.Sentence Encoder
        # 3.1) forward gru for sentence
        hidden_state_forward_sentences = self.gru_forward_sentence_level(sentence_representation)  # a list.length is sentence_length, each element is [None,hidden_size*2]
        # 3.2) backward gru for sentence
        hidden_state_backward_sentences = self.gru_backward_sentence_level(sentence_representation)  # a list,length is sentence_length, each element is [None,hidden_size*2]
        # 3.3) concat forward hidden state and backward hidden state
        # below hidden_state_sentence is a list,len:sentence_length,element:[None,hidden_size*4]
        self.hidden_state_sentence = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in zip(hidden_state_forward_sentences, hidden_state_backward_sentences)]
        #print('self.hidden_state_sentence', len(self.hidden_state_sentence), self.hidden_state_sentence[0].get_shape())
        
        # 4.Sentence Attention
        document_representation = self.attention_sentence_level(self.hidden_state_sentence)  # shape:[None,hidden_size*4]
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(document_representation,keep_prob=self.dropout_keep_prob)  # shape:[None,hidden_size*4]
        # dropout some elements in the document_representation.
        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection  # shape:[None,self.num_classes]==tf.matmul([None,hidden_size*2],[hidden_size*2,self.num_classes])
        return logits
    
    def inference_per_label(self):
        """main computation graph here: 1.Word Encoder. 2.Word Attention. 3.Sentence Encoder 4.Sentence Attention 5.linear classifier"""
        # 1.Word Encoder
        # 1.1 embedding of words
        #print('before spliting', self.input_x.get_shape()) #shape (?,252)
        input_x = tf.split(self.input_x, self.num_sentences,axis=1)  # a list. length:num_sentences.each element is:[None,self.sequence_length/num_sentences]
        #print('before stacking', input_x.get_shape())
        input_x = tf.stack(input_x, axis=1)  # shape:[None,self.num_sentences,self.sequence_length/num_sentences]
        #print('after stacking', input_x.get_shape()) # shape (?,4,63)
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,input_x)  # [None,num_sentences,sentence_length,embed_size]
        #print('after embedding_lookup', self.embedded_words.get_shape()) # shape (?,4,63)
        embedded_words_reshaped = tf.reshape(self.embedded_words, shape=[-1, self.sequence_length,self.embed_size])  # [batch_size*num_sentences,sentence_length,embed_size]
        #print('after reshaping', embedded_words_reshaped.get_shape()) # shape (?,4,63)
        
        #before spliting (?, 252)
        #after stacking (?, 4, 63)
        #after embedding_lookup (?, 4, 63, 100)
        #after reshaping (?, 63, 100) [batch_size*num_sentences,sentence_length,embed_size]

        # 1.2 forward gru
        hidden_state_forward_list = self.gru_forward_word_level(embedded_words_reshaped)  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        # 1.3 backward gru
        hidden_state_backward_list = self.gru_backward_word_level(embedded_words_reshaped)  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        # 1.4 concat forward hidden state and backward hidden state. hidden_state: a list.len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        self.hidden_state = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in
                             zip(hidden_state_forward_list, hidden_state_backward_list)]  # hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
                             #self.hidden_state is a list.
                             
        # 2.Word Attention
        # for each sentence.
        if not self.per_label_sent_only: # the word-level and sentence-level attention weights are all per-label.
            sentence_representation = self.attention_word_level_per_label(self.hidden_state)  
            # output: [num_classes,batch_size*num_sentences,hidden_size*2]
            #W_project_unstacked = tf.unstack(self.W_projection,axis=1)
            #b_projection_unstacked = tf.unstack(self.b_projection,axis=0)
            #n=0
            #logits = []
            #for sentence_representation in sentence_representations:
            sentence_representation = tf.reshape(sentence_representation, shape=[self.num_classes,-1, self.num_sentences, self.hidden_size * 2])  # shape:[num_classes,batch_size,num_sentences,hidden_size*2]

            # 3.Sentence Encoder
            # 3.1) forward gru for sentence
            hidden_state_forward_sentences = self.gru_forward_sentence_level_per_label(sentence_representation)  # a list.length is sentence_length, each element is [num_classes,None,hidden_size*2]
            # 3.2) backward gru for sentence
            hidden_state_backward_sentences = self.gru_backward_sentence_level_per_label(sentence_representation)  # a list,length is sentence_length, each element is [num_classes,None,hidden_size*2]
            # 3.3) concat forward hidden state and backward hidden state
            # below hidden_state_sentence is a list,len:sentence_length,element:[num_classes,None,hidden_size*4]
            self.hidden_state_sentence = [tf.concat([h_forward, h_backward], axis=2) for h_forward, h_backward in zip(hidden_state_forward_sentences, hidden_state_backward_sentences)]
            #print('self.hidden_state_sentence', len(self.hidden_state_sentence), self.hidden_state_sentence[0].get_shape())
        else: # only the sentence-level attention weights are per-label.
            sentence_representation = self.attention_word_level(self.hidden_state)  # output:[batch_size*num_sentences,hidden_size*2]
            sentence_representation = tf.reshape(sentence_representation, shape=[-1, self.num_sentences, self.hidden_size * 2])  # shape:[batch_size,num_sentences,hidden_size*2]

            # 3.Sentence Encoder
            # 3.1) forward gru for sentence
            hidden_state_forward_sentences = self.gru_forward_sentence_level(sentence_representation)  # a list.length is sentence_length, each element is [None,hidden_size*2]
            # 3.2) backward gru for sentence
            hidden_state_backward_sentences = self.gru_backward_sentence_level(sentence_representation)  # a list,length is sentence_length, each element is [None,hidden_size*2]
            # 3.3) concat forward hidden state and backward hidden state
            # below hidden_state_sentence is a list,len:sentence_length,element:[None,hidden_size*4]
            self.hidden_state_sentence = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in zip(hidden_state_forward_sentences, hidden_state_backward_sentences)]
            #print('self.hidden_state_sentence', len(self.hidden_state_sentence), self.hidden_state_sentence[0].get_shape())
        
        # 4.Sentence Attention
        document_representation = self.attention_sentence_level_per_label(self.hidden_state_sentence)  # shape:[num_classes,None,hidden_size*4]
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(document_representation,keep_prob=self.dropout_keep_prob)  # shape:[num_classes,None,hidden_size*4]
        # dropout some elements in the document_representation.
        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            h_drop_transposed = tf.transpose(self.h_drop, perm=[1,2,0]) # shape:[None,hidden_size*4,num_classes]
            logits = tf.multiply(h_drop_transposed, self.W_projection) # here is an element-wise multiplication: only the document representation for the label is multiplied by the projection weights for that label, shape:[None,hidden_size*4,num_classes]==tf.multiply([None,hidden_size*4,num_classes],[hidden_size*4,num_classes])
            logits = tf.reduce_sum(logits, axis=1) + self.b_projection # shape:[None,num_classes]
            #the two lines above calculates a dot product with adding bias between per-label document representations and per-label projection weights.            
            #logit = tf.matmul(self.h_drop, tf.expand_dims(W_project_unstacked[n],axis=1)) + b_projection_unstacked[n]  # shape:[None,self.num_classes]==tf.matmul([None,hidden_size*2],[hidden_size*2,self.num_classes])
            #print('self.h_drop:',self.h_drop)
            #print('tf.expand_dims(W_project_unstacked[n],axis=1)):',tf.expand_dims(W_project_unstacked[n],axis=1))
            #print('b_projection_unstacked[n]:',b_projection_unstacked[n])
            #print('logit:',logit)
            #logits.append(logit)
            #n=n+1    
        #logits = tf.stack(logits) #to test
        #logits = tf.transpose(tf.reduce_sum(logits,axis=2))
        #print('logits:',logits)
        return logits
    
    #unused, this is considered in the "def inference_per_label(self)" above.
    def inference_per_label_sent_only(self):
        """main computation graph here: 1.Word Encoder. 2.Word Attention. 3.Sentence Encoder 4.Sentence Attention 5.linear classifier"""
        # 1.Word Encoder
        # 1.1 embedding of words
        #print('before spliting', self.input_x.get_shape()) #shape (?,252)
        input_x = tf.split(self.input_x, self.num_sentences,axis=1)  # a list. length:num_sentences.each element is:[None,self.sequence_length/num_sentences]
        #print('before stacking', input_x.get_shape())
        input_x = tf.stack(input_x, axis=1)  # shape:[None,self.num_sentences,self.sequence_length/num_sentences]
        #print('after stacking', input_x.get_shape()) # shape (?,4,63)
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,input_x)  # [None,num_sentences,sentence_length,embed_size]
        #print('after embedding_lookup', self.embedded_words.get_shape()) # shape (?,4,63)
        embedded_words_reshaped = tf.reshape(self.embedded_words, shape=[-1, self.sequence_length,self.embed_size])  # [batch_size*num_sentences,sentence_length,embed_size]
        #print('after reshaping', embedded_words_reshaped.get_shape()) # shape (?,4,63)
        
        #before spliting (?, 252)
        #after stacking (?, 4, 63)
        #after embedding_lookup (?, 4, 63, 100)
        #after reshaping (?, 63, 100) [batch_size*num_sentences,sentence_length,embed_size]

        # 1.2 forward gru
        hidden_state_forward_list = self.gru_forward_word_level(embedded_words_reshaped)  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        # 1.3 backward gru
        hidden_state_backward_list = self.gru_backward_word_level(embedded_words_reshaped)  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        # 1.4 concat forward hidden state and backward hidden state. hidden_state: a list.len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        self.hidden_state = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in
                             zip(hidden_state_forward_list, hidden_state_backward_list)]  # hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
                             #self.hidden_state is a list.
                             
        # 2.Word Attention
        # for each sentence.
        sentence_representation = self.attention_word_level(self.hidden_state)  # output:[batch_size*num_sentences,hidden_size*2]
        sentence_representation = tf.reshape(sentence_representation, shape=[-1, self.num_sentences, self.hidden_size * 2])  # shape:[batch_size,num_sentences,hidden_size*2]
        #with tf.name_scope("dropout"):#TODO
        #    sentence_representation = tf.nn.dropout(sentence_representation,keep_prob=self.dropout_keep_prob)  # shape:[None,hidden_size*4]

        # 3.Sentence Encoder
        # 3.1) forward gru for sentence
        hidden_state_forward_sentences = self.gru_forward_sentence_level(sentence_representation)  # a list.length is sentence_length, each element is [None,hidden_size*2]
        # 3.2) backward gru for sentence
        hidden_state_backward_sentences = self.gru_backward_sentence_level(sentence_representation)  # a list,length is sentence_length, each element is [None,hidden_size*2]
        # 3.3) concat forward hidden state and backward hidden state
        # below hidden_state_sentence is a list,len:sentence_length,element:[None,hidden_size*4]
        self.hidden_state_sentence = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in zip(hidden_state_forward_sentences, hidden_state_backward_sentences)]
        #print('self.hidden_state_sentence', len(self.hidden_state_sentence), self.hidden_state_sentence[0].get_shape())
    
        # 4.Sentence Attention
        document_representation = self.attention_sentence_level_per_label_sent_only(self.hidden_state_sentence)  # shape:[num_classes,None,hidden_size*4]
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(document_representation,keep_prob=self.dropout_keep_prob)  # shape:[num_classes,None,hidden_size*4]
        # dropout some elements in the document_representation.
        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            h_drop_transposed = tf.transpose(self.h_drop, perm=[1,2,0]) # shape:[None,hidden_size*4,num_classes]
            logits = tf.multiply(h_drop_transposed, self.W_projection) # shape:[None,hidden_size*4,num_classes]==tf.multiply([None,hidden_size*4,num_classes],[hidden_size*4,num_classes])
            logits = tf.reduce_sum(logits, axis=1) + self.b_projection # shape:[None,num_classes]
            #the two lines above calculates a dot product with adding bias between per-label document representations and per-label projection weights.            
            #logit = tf.matmul(self.h_drop, tf.expand_dims(W_project_unstacked[n],axis=1)) + b_projection_unstacked[n]  # shape:[None,self.num_classes]==tf.matmul([None,hidden_size*2],[hidden_size*2,self.num_classes])
            #print('self.h_drop:',self.h_drop)
            #print('tf.expand_dims(W_project_unstacked[n],axis=1)):',tf.expand_dims(W_project_unstacked[n],axis=1))
            #print('b_projection_unstacked[n]:',b_projection_unstacked[n])
            #print('logit:',logit)
            #logits.append(logit)
            #n=n+1    
        #logits = tf.stack(logits) #to test
        #logits = tf.transpose(tf.reduce_sum(logits,axis=2))
        print('logits:',logits)
        return logits
        
    # loss for single-label classification    
    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                    logits=self.logits);  # sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            # print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss
    
    # loss for multi-label classification (JMAN-s)
    def loss_multilabel(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,
                                                             logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda #12 loss
            self.sim_loss = tf.constant(0., dtype=tf.float32)
            self.sub_loss = tf.constant(0., dtype=tf.float32)
            loss = self.loss_ce + self.l2_losses
        return loss
    
    # L_sim new: j,k per doc, \sum_d \sum_{j,k \in y_d} Sim_jk|R(S_dj)-R(S_dk)|
    def loss_multilabel_onto_new_sim_pair_diff_abs(self, label_sim_matrix, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            #print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            
            # only considering the similarity of co-occuring label in each labelset y_d.
            sig_output = tf.sigmoid(self.logits) # get s_d from l_d
            sig_list=tf.unstack(sig_output)
            
            partitions = tf.range(self.batch_size)
            num_partitions = self.batch_size
            label_list = tf.dynamic_partition(self.input_y_multilabel, partitions, num_partitions, name='dynamic_unstack')
            
            self.sim_loss = 0
            for i in range(len(sig_list)): # loop over d
                logit_vector = tf.expand_dims(sig_list[i],0) # s_d, shape [1,5196]
                #print("logit_vector:",logit_vector)
                
                label_vector = label_list[i] #y_d, shape [1,5196]
                #print("label_vector:",label_vector)
                
                #get an index vector from y_d
                label_index_2d = tf.where(label_vector)
                #gather the s_d_true from s_d: s_d_true means the s_d values for the true labels of document d.  
                s_d_true = tf.expand_dims(tf.gather_nd(logit_vector,label_index_2d),0)
                #calculate |R(S_dj)-R(S_dk)|
                pred_d_true = tf.round(s_d_true)
                pair_diff_abs_d = tf.abs(tf.transpose(pred_d_true) - pred_d_true)
                #gather the Sim_jk from Sim
                label_index = label_index_2d[:,-1]
                label_len = tf.shape(label_index)[0]
                A,B=tf.meshgrid(label_index,tf.transpose(label_index))
                ind_squ = tf.concat([tf.reshape(B,(-1,1)),tf.reshape(A,(-1,1))],axis=-1)
                label_sim_matrix_d = tf.reshape(tf.gather_nd(label_sim_matrix,ind_squ),[label_len,label_len])
                
                self.sim_loss = self.sim_loss + tf.reduce_sum(tf.multiply(label_sim_matrix_d,pair_diff_abs_d))
            self.sim_loss=(self.sim_loss/self.batch_size)*self.lambda_sim/2.0
            self.sub_loss = tf.constant(0., dtype=tf.float32)
            loss = self.loss_ce + self.l2_losses + self.sim_loss
        return loss
        
    # L_sim only: j,k per batch
    def loss_multilabel_onto_new_sim_per_batch(self, label_sim_matrix, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,
                                                             logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            #print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            
            # only considering the similarity of co-occuring label in each labelset y_d. 
            co_label_mat_batch = tf.matmul(tf.transpose(self.input_y_multilabel),self.input_y_multilabel,a_is_sparse=True,b_is_sparse=True) # input_y_multilabel is a matrix \in R^{|D|,|T|}
            co_label_mat_batch = tf.sign(co_label_mat_batch)
            label_sim_matrix = tf.multiply(co_label_mat_batch,label_sim_matrix) # only considering the label similarity of labels in the label set for this document (here is a batch).
            
            # sim-loss after sigmoid L_sim = sim(T_j,T_k)|s_dj-s_dk|^2
            sig_output = tf.sigmoid(self.logits) # self.logit is the matrix S \in R^{|D|,|T|}
            vec_square = tf.multiply(sig_output,sig_output) # element-wise multiplication 
            vec_square = tf.reduce_sum(vec_square,0) # an array of num_classes values {sum_d l_dj^2}_j
            vec_mid = tf.matmul(tf.transpose(sig_output),sig_output)
            vec_rows=tf.ones([tf.size(vec_square),1])*vec_square # copy the vector by it self to shape a square
            vec_columns=tf.transpose(vec_rows)
            vec_diff=vec_rows-2*vec_mid+vec_columns # (li-lj)^2=li^2-2lilj+lj^2 # vec_diff is now a matrix = {sum_d (l_di-l_dj)^2}_i,j
            vec_diff=tf.multiply(vec_diff,label_sim_matrix) #sim(T_i,T_j)*(li-lj)^2 # element-wise # using the label_sim_matrix
            #vec_diff=tf.multiply(vec_diff,co_label_mat_batch) # using only tag co-occurrence 
            vec_final=tf.reduce_sum(vec_diff)/2 # vec_diff is symmetric
            #vec_final=tf.reduce_sum(vec_diff)/2/self.num_classes/self.num_classes # vec_diff is symmetric
            self.sim_loss=(vec_final/self.batch_size)*self.lambda_sim
            
            self.sub_loss = tf.constant(0., dtype=tf.float32)
            loss = self.loss_ce + self.l2_losses + self.sim_loss
        return loss
    
    # sim-loss only: j,k per document - tensor operations only - requiring large GPU memory
    def loss_multilabel_onto_new_sim_per_doc_tensor(self, label_sim_matrix, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,
                                                             logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            
            # only considering the similarity of co-occuring label in each labelset y_d. 
            co_label_mat = tf.matmul(tf.expand_dims(self.input_y_multilabel,2),tf.expand_dims(self.input_y_multilabel,1)) # (128,5196,5196)
            label_sim_matrix = tf.multiply(co_label_mat,tf.expand_dims(label_sim_matrix,0))
            # sim-loss after sigmoid L_sim = sim(T_j,T_k)|s_dj-s_dk|^2
            sig_output = tf.sigmoid(self.logits) # get s_d from l_d
            vec_diff_squared = tf.square(tf.expand_dims(sig_output,1)-tf.expand_dims(sig_output,2)) # (128,5196,5196)
            
            vec_final = tf.reduce_sum(tf.multiply(label_sim_matrix,vec_diff_squared))/2.0
            self.sim_loss=(vec_final/self.batch_size)*self.lambda_sim
            
            self.sub_loss = tf.constant(0., dtype=tf.float32)
            loss = self.loss_ce + self.l2_losses + self.sim_loss
        return loss
    
    # sim-loss only: j,k per document - with for loop operations - requiring large GPU memory [not used]
    def loss_multilabel_onto_new_sim_per_doc_not_used(self, label_sim_matrix, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,
                                                             logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            #print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            
            # only considering the similarity of co-occuring label in each labelset y_d.
            sig_output = tf.sigmoid(self.logits) # get s_d from l_d
            logit_list=tf.unstack(sig_output)
            
            partitions = tf.range(self.batch_size)
            num_partitions = self.batch_size
            label_list = tf.dynamic_partition(self.input_y_multilabel, partitions, num_partitions, name='dynamic_unstack')
            
            self.sim_loss = 0
            for i in range(len(logit_list)):
                logit_vector = tf.expand_dims(logit_list[i],1)
                logit_list[i] = tf.multiply(logit_list[i],0)
                #print("logit_vector:",logit_vector)
                pair_diff = tf.transpose(logit_vector) - logit_vector # pair_diff: {l_di-l_dj}_i,j
                #print("pair_diff:",pair_diff)
                pair_diff_squared = tf.square(pair_diff) # pair_diff_squared: {|l_di-l_dj|^2}_i,j
                #print("pair_diff_squared:",pair_diff_squared)
                
                label_vector = label_list[i]
                label_list[i] = tf.multiply(label_list[i],0)
                #print("label_vector:",label_vector)
                label_co_doc = tf.matmul(tf.transpose(label_vector),label_vector)
                #print("label_co_doc:",label_co_doc)
                label_co_sim_doc = tf.multiply(label_co_doc,label_sim_matrix)
                #print("label_co_sim_doc:",label_co_sim_doc)
                pair_diff_weighted = tf.multiply(label_co_sim_doc,pair_diff_squared)
                #print("pair_diff_weighted:",pair_diff_weighted)
                self.sim_loss = self.sim_loss + tf.reduce_sum(pair_diff_weighted)
            self.sim_loss=(self.sim_loss/self.batch_size)*self.lambda_sim/2.0
            self.sub_loss = tf.constant(0., dtype=tf.float32)
            loss = self.loss_ce + self.l2_losses + self.sim_loss
        return loss
    
    # sim-loss only: j,k per document
    def loss_multilabel_onto_new_sim_per_doc(self, label_sim_matrix, l2_lambda=0.0001, dynamic_sem_l2=False):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            #print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            if dynamic_sem_l2:
                self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            else: # not adding sim and/or sem matrices into the l2 regularisation
                self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name and 'label_sim_mat' not in v.name]) * l2_lambda
            
            # only considering the similarity of co-occuring label in each labelset y_d.
            sig_output = tf.sigmoid(self.logits) # get s_d from l_d
            #sig_list=tf.unstack(sig_output) # this causes valueerror for dynamic dimensions: sig_output shape as (?,num_classes).
            
            partitions = tf.range(tf.shape(sig_output)[0])
            num_partitions = self.batch_size
            sig_list = tf.dynamic_partition(sig_output, partitions, num_partitions, name='dynamic_unstack_logits')
            label_list = tf.dynamic_partition(self.input_y_multilabel, partitions, num_partitions, name='dynamic_unstack_multilabels')
            
            self.sim_loss = 0
            for i in range(len(sig_list)): # loop over d
                #logit_vector = tf.expand_dims(sig_list[i],0) # s_d, shape [1,5196]
                logit_vector = sig_list[i]
                #print("logit_vector:",logit_vector)
                
                label_vector = label_list[i] #y_d, shape [1,5196]
                #print("label_vector:",label_vector)
                label_vector_bool = tf.cast(label_vector, tf.bool) 
                #get an index vector from y_d
                label_index_2d = tf.where(label_vector_bool)
                #gather the s_d_true from s_d: s_d_true means the s_d values for the true labels of document d.  
                s_d_true = tf.expand_dims(tf.gather_nd(logit_vector,label_index_2d),0)
                #calculate |s_dj-s_dk|^2
                pair_diff_squared_d = tf.square(tf.transpose(s_d_true) - s_d_true)
                #gather the Sim_jk from Sim
                label_index = label_index_2d[:,-1]
                label_len = tf.shape(label_index)[0]
                #ind_flat_lower = tf.tile(label_index,[label_len])
                #ind_mat = tf.reshape(ind_flat_lower,[label_len,label_len])
                #ind_flat_upper = tf.reshape(tf.transpose(ind_mat),[-1])
                #ind_squ = tf.transpose(tf.stack([ind_flat_upper,ind_flat_lower]))
                A,B=tf.meshgrid(label_index,tf.transpose(label_index))
                ind_squ = tf.concat([tf.reshape(B,(-1,1)),tf.reshape(A,(-1,1))],axis=-1)
                label_sim_matrix_d = tf.reshape(tf.gather_nd(label_sim_matrix,ind_squ),[label_len,label_len])
                
                self.sim_loss = self.sim_loss + tf.reduce_sum(tf.multiply(label_sim_matrix_d,pair_diff_squared_d))
            self.sim_loss=(self.sim_loss/self.batch_size)*self.lambda_sim/2.0
            self.sub_loss = tf.constant(0., dtype=tf.float32)
            loss = self.loss_ce + self.l2_losses + self.sim_loss
        return loss
    
    # L_sim and L_sub - per doc - L_sim as lambda_sim*|R(S_dj)-R(S_dk)|
    # label_sub_matrix: sub(T_j,T_k) \in {0,1} means whether T_j is a hyponym of T_k.
    def loss_multilabel_onto_new_simsub_pair_diff_abs(self, label_sim_matrix, label_sub_matrix, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            #print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            
            sig_output = tf.sigmoid(self.logits) # get s_d from l_d
            sig_list=tf.unstack(sig_output)
            
            partitions = tf.range(self.batch_size)
            num_partitions = self.batch_size
            label_list = tf.dynamic_partition(self.input_y_multilabel, partitions, num_partitions, name='dynamic_unstack')
            
            self.sim_loss = 0
            self.sub_loss = 0
            for i in range(len(sig_list)): # loop over d
                logit_vector = tf.expand_dims(sig_list[i],0) # s_d, shape [1,5196]
                #print("logit_vector:",logit_vector)
                
                label_vector = label_list[i] #y_d, shape [1,5196]
                #print("label_vector:",label_vector)
                
                #get an index vector from y_d
                label_index_2d = tf.where(label_vector)
                #gather the s_d_true from s_d: s_d_true means the s_d values for the true labels of document d.  
                s_d_true = tf.expand_dims(tf.gather_nd(logit_vector,label_index_2d),0)
                #calculate |R(S_dj)-R(S_dk)|
                pred_d_true = tf.round(s_d_true)
                pair_diff_abs_d = tf.abs(tf.transpose(pred_d_true) - pred_d_true)
                #calculate R(s_dj)(1-R(s_dk))
                pair_sub_d = tf.matmul(tf.transpose(pred_d_true),1-pred_d_true)
                
                #gather the Sim_jk from Sim and the Sub_jk from Sub
                label_index = label_index_2d[:,-1]
                label_len = tf.shape(label_index)[0]
                A,B=tf.meshgrid(label_index,tf.transpose(label_index))
                ind_squ = tf.concat([tf.reshape(B,(-1,1)),tf.reshape(A,(-1,1))],axis=-1)
                label_sim_matrix_d = tf.reshape(tf.gather_nd(label_sim_matrix,ind_squ),[label_len,label_len])
                label_sub_matrix_d = tf.reshape(tf.gather_nd(label_sub_matrix,ind_squ),[label_len,label_len])
                
                self.sim_loss = self.sim_loss + tf.reduce_sum(tf.multiply(label_sim_matrix_d,pair_diff_abs_d))
                self.sub_loss = self.sub_loss + tf.reduce_sum(tf.multiply(label_sub_matrix_d,pair_sub_d))
            self.sim_loss=(self.sim_loss/self.batch_size)*self.lambda_sim/2.0
            self.sub_loss=(self.sub_loss/self.batch_size)*self.lambda_sub/2.0
            
            loss = self.loss_ce + self.l2_losses + self.sim_loss + self.sub_loss
        return loss
        
    # L_sim and L_sub - per doc
    # label_sub_matrix: sub(T_j,T_k) \in {0,1} means whether T_j is a hyponym of T_k.
    def loss_multilabel_onto_new_simsub_per_doc(self, label_sim_matrix, label_sub_matrix, l2_lambda=0.0001, dynamic_sem_l2=False):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            #print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            if dynamic_sem_l2:
                self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            else: # not adding sim and/or sem matrices into the l2 regularisation
                self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name and 'label_sim_mat' not in v.name and 'label_sub_mat' not in v.name]) * l2_lambda
            
            sig_output = tf.sigmoid(self.logits) # get s_d from l_d
            #sig_list=tf.unstack(sig_output) # this causes valueerror for dynamic dimensions: sig_output shape as (?,num_classes).
            
            #partitions = tf.range(self.batch_size)
            #num_partitions = self.batch_size
            #print('sig_output.get_shape()',sig_output.get_shape()[0])
            #print('tf.shape(sig_output)[0]',tf.shape(sig_output)[0])
            #for the dynamic partition below, a good reference is https://stackoverflow.com/questions/45404056/tf-unstack-with-dynamic-shape 
            num_partitions = self.batch_size # this is the max number of partitions
            partitions = tf.range(tf.shape(sig_output)[0])
            sig_list = tf.dynamic_partition(sig_output, partitions, num_partitions, name='dynamic_unstack_logits')
            label_list = tf.dynamic_partition(self.input_y_multilabel, partitions, num_partitions, name='dynamic_unstack_multilabels')
            #print(len(sig_list),sig_list)
            #print(len(label_list),label_list)
            self.sim_loss = 0
            self.sub_loss = 0
            for i in range(len(sig_list)): # loop over d
                #logit_vector = tf.expand_dims(sig_list[i],0) # s_d, shape [1,5196]
                logit_vector = sig_list[i]
                #print("logit_vector:",logit_vector)
                
                label_vector = label_list[i] #y_d, shape [1,5196]
                #print("label_vector:",label_vector)
                label_vector_bool = tf.cast(label_vector, tf.bool) 
                #print("label_vector_bool:",label_vector_bool)
                
                #get an index vector from y_d
                label_index_2d = tf.where(label_vector_bool)
                
                #gather the s_d_true from s_d: s_d_true means the s_d values for the true labels of document d.  
                s_d_true = tf.expand_dims(tf.gather_nd(logit_vector,label_index_2d),0)
                #calculate |s_dj-s_dk|^2
                pair_diff_squared_d = tf.square(tf.transpose(s_d_true) - s_d_true)
                #calculate R(s_dj)(1-R(s_dk))
                pred_d_true = tf.round(s_d_true)
                pair_sub_d = tf.matmul(tf.transpose(pred_d_true),1-pred_d_true)
                
                #gather the Sim_jk from Sim and the Sub_jk from Sub
                label_index = label_index_2d[:,-1]
                label_len = tf.shape(label_index)[0]
                A,B=tf.meshgrid(label_index,tf.transpose(label_index))
                ind_squ = tf.concat([tf.reshape(B,(-1,1)),tf.reshape(A,(-1,1))],axis=-1)
                label_sim_matrix_d = tf.reshape(tf.gather_nd(label_sim_matrix,ind_squ),[label_len,label_len])
                label_sub_matrix_d = tf.reshape(tf.gather_nd(label_sub_matrix,ind_squ),[label_len,label_len])
                
                self.sim_loss = self.sim_loss + tf.reduce_sum(tf.multiply(label_sim_matrix_d,pair_diff_squared_d))
                self.sub_loss = self.sub_loss + tf.reduce_sum(tf.multiply(label_sub_matrix_d,pair_sub_d))
            self.sim_loss=(self.sim_loss/self.batch_size)*self.lambda_sim/2.0
            self.sub_loss=(self.sub_loss/self.batch_size)*self.lambda_sub/2.0
            
            loss = self.loss_ce + self.l2_losses + self.sim_loss + self.sub_loss
        return loss
        
    # L_sim and L_sub - per batch, used in the NAACL paper
    # label_sub_matrix: sub(T_j,T_k) \in {0,1} means whether T_j is a hypernym of T_k.
    def loss_multilabel_onto_new_simsub_per_batch(self, label_sim_matrix, label_sub_matrix, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            #print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            
            co_label_mat_batch = tf.matmul(tf.transpose(self.input_y_multilabel),self.input_y_multilabel,a_is_sparse=True,b_is_sparse=True)
            co_label_mat_batch = tf.sign(co_label_mat_batch)
            label_sim_matrix = tf.multiply(co_label_mat_batch,label_sim_matrix) # only considering the label similarity of labels in the label set for this document (batch of documents).
            label_sub_matrix = tf.multiply(co_label_mat_batch,label_sub_matrix)
            
            # the sim-loss after sigmoid
            sig_output = tf.sigmoid(self.logits)
            vec_square = tf.multiply(sig_output,sig_output)
            vec_square = tf.reduce_sum(vec_square,0) # an array of num_classes values {sum_d l_di}_i
            vec_mid = tf.matmul(tf.transpose(sig_output),sig_output)
            vec_rows=tf.ones([tf.size(vec_square),1])*vec_square
            vec_columns=tf.transpose(vec_rows)
            vec_diff=vec_rows-2*vec_mid+vec_columns # (li-lj)^2=li^2-2lilj+lj^2 # vec_diff is now a matrix = {sum_d (l_di-l_dj)^2}_i,j
            vec_diff=tf.multiply(vec_diff,label_sim_matrix) #sim(T_i,T_j)*(li-lj)^2 # element-wise # using the label_sim_matrix
            #vec_diff=tf.multiply(vec_diff,co_label_mat_batch) # using only tag co-occurrence 
            vec_final=tf.reduce_sum(vec_diff)/2 # vec_diff is symmetric
            #vec_final=tf.reduce_sum(vec_diff)/2/self.num_classes/self.num_classes # vec_diff is symmetric
            self.sim_loss=(vec_final/self.batch_size)*self.lambda_sim
            
            # the sub-loss after sigmoid 
            pred = tf.round(sig_output)
            pred_mat = tf.matmul(tf.transpose(pred),1-pred)
            sub_loss = tf.multiply(pred_mat,label_sub_matrix)
            self.sub_loss = self.lambda_sub * tf.reduce_sum(sub_loss) / 2. / self.batch_size
            
            loss = self.loss_ce + self.l2_losses + self.sim_loss + self.sub_loss
        return loss
    
    # L_sub only - per batch - used in the NAACL paper
    def loss_multilabel_onto_new_sub_per_batch(self, label_sub_matrix, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            #print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            
            ## sub_loss: matrix multiplication: only using the label relations in the label set, treating same in each batch.
            co_label_mat_batch = tf.matmul(tf.transpose(self.input_y_multilabel),self.input_y_multilabel,a_is_sparse=True,b_is_sparse=True)
            co_label_mat_batch = tf.sign(co_label_mat_batch)
            label_sub_matrix = tf.multiply(co_label_mat_batch,label_sub_matrix)
            
            # the sub-loss after sigmoid
            sig_output = tf.sigmoid(self.logits)
            pred = tf.round(sig_output)
            pred_mat = tf.matmul(tf.transpose(pred),1-pred)
            sub_loss = tf.multiply(pred_mat,label_sub_matrix)
            self.sub_loss = self.lambda_sub * tf.reduce_sum(sub_loss) / 2. / self.batch_size
            
            self.sim_loss = tf.constant(0., dtype=tf.float32)
            loss = self.loss_ce + self.l2_losses + self.sub_loss
        return loss
    
    # L_sub only - per document
    def loss_multilabel_onto_new_sub_per_doc(self, label_sub_matrix, l2_lambda=0.0001, dynamic_sem_l2=False):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,logits=self.logits);  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            #print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            self.loss_ce = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            if dynamic_sem_l2:
                self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            else: # not adding sim and/or sem matrices into the l2 regularisation
                self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name and 'label_sub_mat' not in v.name]) * l2_lambda
            
            ## sub_loss: matrix multiplication: only using the label relations in the label set, treating same in each batch.
            # only considering the similarity of co-occuring label in each labelset y_d.
            sig_output = tf.sigmoid(self.logits) # get s_d from l_d
            #sig_list=tf.unstack(sig_output) # this causes valueerror for dynamic dimensions: sig_output shape as (?,num_classes).
            
            partitions = tf.range(tf.shape(sig_output)[0])
            num_partitions = self.batch_size
            sig_list = tf.dynamic_partition(sig_output, partitions, num_partitions, name='dynamic_unstack_logits')
            label_list = tf.dynamic_partition(self.input_y_multilabel, partitions, num_partitions, name='dynamic_unstack_multilabels')
            
            self.sub_loss = 0
            for i in range(len(sig_list)): # loop over d
                #logit_vector = tf.expand_dims(sig_list[i],0) # s_d, shape [1,5196]
                logit_vector = sig_list[i]
                #print("logit_vector:",logit_vector)
                
                label_vector = label_list[i] #y_d, shape [1,5196]
                #print("label_vector:",label_vector)
                label_vector_bool = tf.cast(label_vector, tf.bool) 
                #get an index vector from y_d
                label_index_2d = tf.where(label_vector_bool)
                #gather the s_d_true from s_d: s_d_true means the s_d values for the true labels of document d.  
                s_d_true = tf.expand_dims(tf.gather_nd(logit_vector,label_index_2d),0)
                #calculate R(s_dj)(1-R(s_dk))
                pred_d_true = tf.round(s_d_true)
                pair_sub_d = tf.matmul(tf.transpose(pred_d_true),1-pred_d_true)
                #gather the Sub_jk from Sub
                label_index = label_index_2d[:,-1]
                label_len = tf.shape(label_index)[0]
                A,B=tf.meshgrid(label_index,tf.transpose(label_index))
                ind_squ = tf.concat([tf.reshape(B,(-1,1)),tf.reshape(A,(-1,1))],axis=-1)
                label_sub_matrix_d = tf.reshape(tf.gather_nd(label_sub_matrix,ind_squ),[label_len,label_len])
                
                self.sub_loss = self.sub_loss + tf.reduce_sum(tf.multiply(label_sub_matrix_d,pair_sub_d))
            
            self.sub_loss=(self.sub_loss/self.batch_size)*self.lambda_sub/2.0
            
            self.sim_loss = tf.constant(0., dtype=tf.float32)
            loss = self.loss_ce + self.l2_losses + self.sub_loss
        return loss
        
    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True) #decay of learning rate
        #self.learning_rate_=learning_rate
        #noise_std_dev = tf.constant(0.3) / (tf.sqrt(tf.cast(tf.constant(1) + self.global_step, tf.float32))) #gradient_noise_scale=noise_std_dev
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients) #using adam here. # gradient cliping is also applied.
        return train_op

    def gru_single_step_word_level(self, Xt, h_t_minus_1):
        """
        single step of gru for word level
        :param Xt: Xt:[batch_size*num_sentences,embed_size]
        :param h_t_minus_1:[batch_size*num_sentences,embed_size]
        :return:
        """
        # update gate: decides how much past information is kept and how much new information is added.
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1,
                                                                self.U_z) + self.b_z)  # z_t:[batch_size*num_sentences,self.hidden_size]
        # reset gate: controls how much the past state contributes to the candidate state.
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1,
                                                                self.U_r) + self.b_r)  # r_t:[batch_size*num_sentences,self.hidden_size]
        # candiate state h_t~
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h) +r_t * (tf.matmul(h_t_minus_1, self.U_h)) + self.b_h)  # h_t_candiate:[batch_size*num_sentences,self.hidden_size]
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate  # h_t:[batch_size*num_sentences,hidden_size]
        return h_t
    
    def gru_single_step_sentence_level(self, Xt,
                                       h_t_minus_1):  # Xt:[batch_size, hidden_size*2]; h_t:[batch_size, hidden_size*2]
        """
        single step of gru for sentence level
        :param Xt:[batch_size, hidden_size*2]
        :param h_t_minus_1:[batch_size, hidden_size*2]
        :return:h_t:[batch_size,hidden_size]
        """
        # update gate: decides how much past information is kept and how much new information is added.
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z_sentence) + tf.matmul(h_t_minus_1,
                                                                         self.U_z_sentence) + self.b_z_sentence)  # z_t:[batch_size,self.hidden_size*2]
        #print('z_t in gru_single_step_sentence_level', z_t.get_shape()) # z_t in gru_single_step_sentence_level (128, 200)                                                             
        # reset gate: controls how much the past state contributes to the candidate state.
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r_sentence) + tf.matmul(h_t_minus_1,
                                                                         self.U_r_sentence) + self.b_r_sentence)  # r_t:[batch_size,self.hidden_size*2]
        #print('r_t in gru_single_step_sentence_level', r_t.get_shape()) # r_t in gru_single_step_sentence_level (128, 200)                                                                             
        # candiate state h_t~
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h_sentence) + r_t * (
        tf.matmul(h_t_minus_1, self.U_h_sentence)) + self.b_h_sentence)  # h_t_candiate:[batch_size,self.hidden_size*2]
        #print('h_t_candiate in gru_single_step_sentence_level', h_t_candiate.get_shape()) # h_t_candiate in gru_single_step_sentence_level (128, 200)    
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate
        #print('h_t in gru_single_step_sentence_level', h_t.get_shape()) # h_t in gru_single_step_sentence_level (128, 200)            
        return h_t
        
    def gru_single_step_sentence_level_per_label(self, Xt,
                                       h_t_minus_1):  # Xt:[batch_size, hidden_size*2]; h_t:[batch_size, hidden_size*2]
        """
        single step of gru for sentence level
        :param Xt:[batch_size, hidden_size*2] or [num_classes, batch_size, hidden_size*2]
        :param h_t_minus_1:[batch_size, hidden_size*2] or [num_classes, batch_size, hidden_size*2]
        :return:h_t:[batch_size,hidden_size]
        """
        #expand dimension to the same rank for tf.matmul
        W_z_sentence = tf.tile(tf.expand_dims(self.W_z_sentence, axis=0),[self.num_classes,1,1])
        U_z_sentence = tf.tile(tf.expand_dims(self.U_z_sentence, axis=0),[self.num_classes,1,1])
        W_r_sentence = tf.tile(tf.expand_dims(self.W_r_sentence, axis=0),[self.num_classes,1,1])
        U_r_sentence = tf.tile(tf.expand_dims(self.U_r_sentence, axis=0),[self.num_classes,1,1])
        W_h_sentence = tf.tile(tf.expand_dims(self.W_h_sentence, axis=0),[self.num_classes,1,1])
        U_h_sentence = tf.tile(tf.expand_dims(self.U_h_sentence, axis=0),[self.num_classes,1,1])
        # update gate: decides how much past information is kept and how much new information is added.
        z_t = tf.nn.sigmoid(tf.matmul(Xt, W_z_sentence) + tf.matmul(h_t_minus_1,
                                                                         U_z_sentence) + self.b_z_sentence)  # z_t:[batch_size,self.hidden_size*2]
        #print('z_t in gru_single_step_sentence_level', z_t.get_shape()) # z_t in gru_single_step_sentence_level (128, 200)                                                             
        # reset gate: controls how much the past state contributes to the candidate state.
        r_t = tf.nn.sigmoid(tf.matmul(Xt, W_r_sentence) + tf.matmul(h_t_minus_1,
                                                                         U_r_sentence) + self.b_r_sentence)  # r_t:[batch_size,self.hidden_size*2]
        #print('r_t in gru_single_step_sentence_level', r_t.get_shape()) # r_t in gru_single_step_sentence_level (128, 200)                                                                             
        # candiate state h_t~
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, W_h_sentence) + r_t * (
        tf.matmul(h_t_minus_1, U_h_sentence)) + self.b_h_sentence)  # h_t_candiate:[batch_size,self.hidden_size*2]
        #print('h_t_candiate in gru_single_step_sentence_level', h_t_candiate.get_shape()) # h_t_candiate in gru_single_step_sentence_level (128, 200)    
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate # this is element-wise multiplication
        #print('h_t in gru_single_step_sentence_level', h_t.get_shape()) # h_t in gru_single_step_sentence_level (128, 200)            
        return h_t

    # forward gru for first level: word levels
    def gru_forward_word_level(self, embedded_words):
        """
        :param embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        :return:forward hidden state: a list.length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        """
        # split embedded_words
        embedded_words_splitted = tf.split(embedded_words, self.sequence_length,
                                           axis=1)  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,1,embed_size]
                                           # Now the sequence_length is the sentence_length
        #print('after splitting in gru', len(embedded_words_splitted), embedded_words_splitted[0].get_shape())                                   
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in
                                  embedded_words_splitted]  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,embed_size]
        # demension_1=embedded_words_squeeze[0].get_shape().dims[0]
        #h_t = tf.ones((self.batch_size * self.num_sentences,
        #               self.hidden_size))  #TODO self.hidden_size h_t =int(tf.get_shape(embedded_words_squeeze[0])[0]) # tf.ones([self.batch_size*self.num_sentences, self.hidden_size]) # [batch_size*num_sentences,embed_size]
        h_t = tf.ones_like(embedded_words_squeeze[0])
        h_t_forward_list = []
        for time_step, Xt in enumerate(embedded_words_squeeze):  # Xt: [batch_size*num_sentences,embed_size]
            h_t = self.gru_single_step_word_level(Xt,h_t)  # [batch_size*num_sentences,embed_size]<------Xt:[batch_size*num_sentences,embed_size];h_t:[batch_size*num_sentences,embed_size]
            h_t_forward_list.append(h_t)
        return h_t_forward_list  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]

    # backward gru for first level: word level
    def gru_backward_word_level(self, embedded_words):
        """
        :param   embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        :return: backward hidden state:a list.length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        """
        # split embedded_words
        embedded_words_splitted = tf.split(embedded_words, self.sequence_length,
                                           axis=1)  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,1,embed_size]
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in
                                  embedded_words_splitted]  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,embed_size]
        embedded_words_squeeze.reverse()  # it is a list,length is sentence_length, each element is [batch_size*num_sentences,embed_size]
        # demension_1=int(tf.get_shape(embedded_words_squeeze[0])[0]) #h_t = tf.ones([self.batch_size*self.num_sentences, self.hidden_size])
        #h_t = tf.ones((self.batch_size * self.num_sentences, self.hidden_size))
        h_t = tf.ones_like(embedded_words_squeeze[0])
        h_t_backward_list = []
        for time_step, Xt in enumerate(embedded_words_squeeze):
            h_t = self.gru_single_step_word_level(Xt, h_t)
            h_t_backward_list.append(h_t)
        h_t_backward_list.reverse() #ADD 2017.06.14
        return h_t_backward_list

    # forward gru for second level: sentence level
    def gru_forward_sentence_level(self, sentence_representation):
        """
        :param sentence_representation: [batch_size,num_sentences,hidden_size*2]
        :return:forward hidden state: a list,length is num_sentences, each element is [batch_size,hidden_size*2]
        """
        # split embedded_words
        sentence_representation_splitted = tf.split(sentence_representation, self.num_sentences,
                                                    axis=1)  # it is a list.length is num_sentences,each element is [batch_size,1,hidden_size*2]
        sentence_representation_squeeze = [tf.squeeze(x, axis=1) for x in
                                           sentence_representation_splitted]  # it is a list.length is num_sentences,each element is [batch_size, hidden_size*2]
        # demension_1 = int(tf.get_shape(sentence_representation_squeeze[0])[0]) #scalar: batch_size
        #h_t = tf.ones((self.batch_size, self.hidden_size * 2))  # TODO
        h_t = tf.ones_like(sentence_representation_squeeze[0])
        h_t_forward_list = []
        for time_step, Xt in enumerate(sentence_representation_squeeze):  # Xt:[batch_size, hidden_size*2]
            h_t = self.gru_single_step_sentence_level(Xt,
                                                      h_t)  # h_t:[batch_size,hidden_size*2]<---------Xt:[batch_size, hidden_size*2]; h_t:[batch_size, hidden_size*2]
            h_t_forward_list.append(h_t)
        return h_t_forward_list  # a list,length is num_sentences, each element is [batch_size,hidden_size*2]
    
    # forward gru for second level: sentence level
    def gru_forward_sentence_level_per_label(self, sentence_representation):
        """
        :param sentence_representation: [num_classes,batch_size,num_sentences,hidden_size*2]
        :return:forward hidden state: a list,length is num_sentences, each element is [num_classes,batch_size,hidden_size*2]
        """
        # split embedded_words
        sentence_representation_splitted = tf.split(sentence_representation, self.num_sentences,
                                                    axis=2)  # it is a list.length is num_sentences,each element is  [num_classes,batch_size,1,hidden_size*2]
        sentence_representation_squeeze = [tf.squeeze(x, axis=2) for x in
                                           sentence_representation_splitted]  # it is a list.length is num_sentences,each element is [num_classes,batch_size, hidden_size*2]
        # demension_1 = int(tf.get_shape(sentence_representation_squeeze[0])[0]) #scalar: batch_size
        #h_t = tf.ones((self.num_classes, self.batch_size, self.hidden_size * 2))
        h_t = tf.ones_like(sentence_representation_squeeze[0])
        h_t_forward_list = []
        for time_step, Xt in enumerate(sentence_representation_squeeze):  # Xt:[num_classes, batch_size, hidden_size*2]
            h_t = self.gru_single_step_sentence_level_per_label(Xt,
                                                      h_t)  # h_t:[num_classes,batch_size,hidden_size*2]<---------Xt:[num_classes, batch_size, hidden_size*2]; h_t:[batch_size, hidden_size*2] or [num_classes, batch_size, hidden_size*2]
            h_t_forward_list.append(h_t)
        return h_t_forward_list  # a list,length is num_sentences, each element is [num_classes, batch_size,hidden_size*2]
        
    # backward gru for second level: sentence level
    def gru_backward_sentence_level(self, sentence_representation):
        """
        :param sentence_representation: [batch_size,num_sentences,hidden_size*2]
        :return:forward hidden state: a list,length is num_sentences, each element is [batch_size,hidden_size]
        """
        # split embedded_words
        sentence_representation_splitted = tf.split(sentence_representation, self.num_sentences,
                                                    axis=1)  # it is a list.length is num_sentences,each element is [batch_size,1,hidden_size*2]
        sentence_representation_squeeze = [tf.squeeze(x, axis=1) for x in
                                           sentence_representation_splitted]  # it is a list.length is num_sentences,each element is [batch_size, hidden_size*2]
        sentence_representation_squeeze.reverse()
        # demension_1 = int(tf.get_shape(sentence_representation_squeeze[0])[0])  # scalar: batch_size
        #h_t = tf.ones((self.batch_size, self.hidden_size * 2))
        h_t = tf.ones_like(sentence_representation_squeeze[0])
        h_t_forward_list = []
        for time_step, Xt in enumerate(sentence_representation_squeeze):  # Xt:[batch_size, hidden_size*2]
            h_t = self.gru_single_step_sentence_level(Xt,h_t)  # h_t:[batch_size,hidden_size*2]<---------Xt:[batch_size, hidden_size*2]; h_t:[batch_size, hidden_size*2]
            h_t_forward_list.append(h_t)
        h_t_forward_list.reverse() #ADD 2017.06.14
        return h_t_forward_list  # a list,length is num_sentences, each element is [batch_size,hidden_size*2]
    
    # backward gru for second level: sentence level
    def gru_backward_sentence_level_per_label(self, sentence_representation):
        """
        :param sentence_representation: [num_classes, batch_size,num_sentences,hidden_size*2]
        :return:forward hidden state: a list,length is num_sentences, each element is [num_classes,batch_size,hidden_size]
        """
        # split embedded_words
        sentence_representation_splitted = tf.split(sentence_representation, self.num_sentences,
                                                    axis=2)  # it is a list.length is num_sentences,each element is [num_classes,batch_size,1,hidden_size*2]
        sentence_representation_squeeze = [tf.squeeze(x, axis=2) for x in
                                           sentence_representation_splitted]  # it is a list.length is num_sentences,each element is [num_classes,batch_size, hidden_size*2]
        sentence_representation_squeeze.reverse()
        # demension_1 = int(tf.get_shape(sentence_representation_squeeze[0])[0])  # scalar: batch_size
        #h_t = tf.ones((self.num_classes, self.batch_size, self.hidden_size * 2))
        h_t = tf.ones_like(sentence_representation_squeeze[0])
        h_t_forward_list = []
        for time_step, Xt in enumerate(sentence_representation_squeeze):  # Xt:[num_classes, batch_size, hidden_size*2]
            h_t = self.gru_single_step_sentence_level_per_label(Xt,h_t)  # h_t:[num_classes,batch_size,hidden_size*2]<---------Xt:[num_classes,batch_size, hidden_size*2]; h_t:[batch_size, hidden_size*2]
            h_t_forward_list.append(h_t)
        h_t_forward_list.reverse() #ADD 2017.06.14
        return h_t_forward_list  # a list,length is num_sentences, each element is [num_classes,batch_size,hidden_size*2]