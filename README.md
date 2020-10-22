# Explanable Automated Medical Coding

This project proposes an explanable automated medical coding approach based on Hierarchical Label-wise Attention Network and label embedding initialisation. The approach can be applied to multi-label text classification in any domains.

The detail explanation of the approach is in 
* Explainable Automated Coding of Clinical Notes using Hierarchical Label-Wise Attention Networks and Label Embedding Initialisation, submitted to *Journal of Biomedical Informatics*. Preprint to be available soon.

A part of the results (especially regarding label embedding initialisation) was orally and virtually presented in [HealTAC 2020](http://healtex.org/healtac-2020/programme/) with [slides](https://drive.google.com/file/d/1XIhuwMuelmsFYvsXYABr7NJjnJTULrez/view) available.

## Hierarchical Label-wise Attention Network
<p align="center">
    <img src="https://github.com/acadTags/Explainable-Automated-Medical-Coding/blob/master/HLAN/HLAN-architecture.PNG" width="600" title="Hierarchical Label-wise Attention Network">
</p>

The key computation graph is implemented in [```def inference_per_label(self)```](https://github.com/acadTags/Explainable-Automated-Medical-Coding/blob/cd9a360d5522d239d2ac5cef9a4ab507627bfa8d/HLAN/HAN_model_dynamic.py#L491) in ```./HLAN/HAN_model.py```.

# Requirements
* Python 3 (tested on versions 3.6.* and 3.7.*)
* Tensorflow 1.* (tested on versions 1.4.1, 1.8.0, and 1.14.0)
* [Numpy](http://www.numpy.org/)
* [scikit-learn](http://scikit-learn.github.io/stable) for implementing evaluation metrics
* [Gensim](https://radimrehurek.com/gensim/) for pre-training word and label embeddings with the word2vec algorithm
* [NLTK](https://www.nltk.org/) for tokenisation
* [Spacy](https://spacy.io/) for customised rule-based sentence parsing
* [TFLearn](http://tflearn.org/) for sequence padding

# Demo for Quick Start
* First, please ensure that you have requested the MIMIC-III dataset, see [the official page to request MIMIC-III](https://mimic.physionet.org/gettingstarted/access/).

* Second, download the files in ```checkpoints``` and ```cache_vocabulary_label_pik``` folders from Onedrive.

* Third, run the Jupyter Notebook demo [```demo_HLAN_viz.ipynb```](https://github.com/acadTags/ExplainableAutomated-Medical-Coding/blob/master/HLAN/demo_HLAN_viz.ipynb) and try with your own discharge summaries or those in the MIMIC-III dataset. By setting the ```to_input``` in the notebook as ```True```, the notebook will ask you to input or paste a discharge summary; otherwise, you can save your discharge summaries, each in a line, under the ```..\dataset\``` folder and replace the ```filename_to_predict``` to the filename (see ```Section 2.part2``` in the notebook). After running, the predictions are displayed with label-wise attention visualisations. The attention visualisations are further stored as ```.xlsx``` files in the ```..\explanation\``` folder. 

# Content
* ```./HLAN/HAN_train.py``` contains code for configuration and training
* ```./HLAN/HAN_model.py``` contains the computational graph, loss function and optimisation
* ```./HLAN/data_util_gensim.py``` contains code for input and target generation
* ```./HLAN/demo_HLAN_viz.ipynb``` and ```./HLAN/model_predict_util.py``` contains code for the demo based on Jupyter Notebook and the helper functions
* ```./embeddings``` contains self-trained word2vec embeddings: word embeddings and label embeddings
* ```./datasets``` contains the datasets used
* ```./checkpoints``` contains the checkpoints of HLAN, HA-GRU, and HAN models trained from the author on the MIMIC-III datasets
* ```./explanations``` contains the Excel sheets displaying the attention visualisation, generated after running the demo in ```./HLAN/demo_HLAN_viz.ipynb```
* ```./knowledge_bases``` contains knowledge sources used for label subsumption relations and the ICD code description files
* ```./cache_vocabulary_label_pik``` stores the cached .pik files about vocabularies and labels
* ```./results-HEALTAC 2020``` contains the CNN, CNN+att, Bi-GRU, BERT results with label embedding initilisation

# Key Configurations and Further Details

## Training the models

```--dataset``` is set to ```mimic3=ds-50``` for MIMIC-III-50, ```mimic-ds``` for MIMIC-III, and ```mimic-ds-shielding-th50``` for MIMIC-III-shielding.

To use label embedding initialisation (*+LE*), set ```--use_label_embedding``` to ```True```; otherwise, set it to ```False```.

The files under ```./embeddings``` can be downloaded from Onedrive.

All the ```--marking_id```s below are simply for better marking of the command, which will appear in the name of the output files, can be changed to other values and do not affect the running.

#### Run HLAN
To train with the MIMIC-III-50 dataset
```
python HAN_train.py --dataset mimic3-ds-50 --batch_size 32 --per_label_attention=True --per_label_sent_only=False --num_epochs 100 --report_rand_pred=False --running_times 1 --early_stop_lr 0.00002 --remove_ckpts_before_train=False --use_label_embedding=True --ckpt_dir checkpoint_HLAN+LE_50/ --use_sent_split_padded_version=False --marking_id 50-hlan --gpu=True
```

#### Run HA-GRU
This is by changing ```--per_label_sent_only``` to ```True``` while keeping  ```--per_label_attention``` as ```True```.

To train with the MIMIC-III-50 dataset
```
python HAN_train.py --dataset mimic3-ds-50 --batch_size 32 --per_label_attention=True --per_label_sent_only=True --num_epochs 100 --report_rand_pred=False --running_times 1 --early_stop_lr 0.00002 --remove_ckpts_before_train=False --use_label_embedding=True --ckpt_dir checkpoint_HAGRU+LE_50/ --use_sent_split_padded_version=False --marking_id 50-hagru --gpu=True
```

#### Run HAN
This is by changing ```--per_label_attention``` to ```False```. The ```--batch_size``` is changed to ```128``` for this model in the experiment.

To train with the MIMIC-III-50 dataset
```
python HAN_train.py --dataset mimic3-ds-50 --batch_size 128 --per_label_attention=False --per_label_sent_only=False --num_epochs 100 --report_rand_pred=False --running_times 1 --early_stop_lr 0.00002 --remove_ckpts_before_train=False --use_label_embedding=True --ckpt_dir checkpoint_HAN+LE_50/ --use_sent_split_padded_version=False --marking_id 50-han --gpu=True
```

#### Other Configurations
For all the models above, you can set the learning rate (```--learning_rate```), number of epochs (```--num_epochs```), early stop learning rate (```--early_stop_lr```), and other configurations when you run the command, or set those in the ```*_train.py``` files.

Check the full list of configurations in ```HAN_train.py```.

To view the changing of training loss and validation loss, replacing $PATH-logs$ to a real path.
```
tensorboard --logdir $PATH-logs$
```

## Label embedding initialisation
<p align="center">
    <img src="https://github.com/acadTags/Explainable-Automated-Medical-Coding/blob/master/results-HealTAC%202020/label-embedding-init-figure.PNG" width="300" title="Label Embedding Initialisation for Deep-Learning-Based Multi-Label Classification">
</p>

Key part of the implementation of label embedding initiailisation is in the two functions [```def assign_pretrained_label_embedding_per_label```](https://github.com/acadTags/Explainable-Automated-Medical-Coding/blob/cd9a360d5522d239d2ac5cef9a4ab507627bfa8d/HLAN/HAN_train.py#L759) (for HLAN and HA-GRU) and [```def assign_pretrained_label_embedding```](https://github.com/acadTags/Explainable-Automated-Medical-Coding/blob/cd9a360d5522d239d2ac5cef9a4ab507627bfa8d/HLAN/HAN_train.py#L720) (for HAN) in ```./HLAN/HAN_train.py```.

Besides, below is the implementation of label embedding initiailisation on top of the [```model.py```](https://github.com/jamesmullenbach/caml-mimic/blob/master/learn/models.py) from the caml-mimic GitHub project.
```
# based on https://github.com/jamesmullenbach/caml-mimic/blob/master/learn/models.py
def _code_emb_init(self, code_emb, code_list):
        # code_emb is a Gensim Word2Vec model loaded from pre-trained label embeddings
        # code_list is a list of code having the same order as in multi-hot representation (sorted by frequency from high to low)
        code_embs = Word2Vec.load(code_emb)
        # bound for random variables for Xavier initialisation.
        bound = np.sqrt(6.0) / np.sqrt(self.num_labels + code_embs.vector_size)  
        weights = np.zeros(self.classifier.weight.size())
        n_exist, n_inexist = 0, 0
        for i in range(self.num_labels):
            code = code_list[i]
            if code in code_embs.wv.vocab:
                n_exist = n_exist + 1
                vec = code_embs.wv[code]
                #normalise to unit length
                weights[i] = vec / float(np.linalg.norm(vec) + 1e-6) 
                #additional standardisation for BERT models:
                #standardise to the same as the originial initilisation in def _init_weights(self, module) in https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
                #weights[i] = stats.zscore(weights[i])*self.initializer_range # self.initializer_range = 0.02
                
            else:
                n_inexist = n_inexist + 1
                #using the original xavier uniform initialisation for CNN, CNN+att, and BiGRU
                weights[i] = np.random.uniform(-bound, bound, code_embs.vector_size);
                #or using the original normal distribution initialisation for BERT
                #weights[i] = np.random.normal(0, std, code_embs.vector_size);
        print("code exists embedding:", n_exist, " ;code not exist embedding:", n_inexist)
        
        # initialise label embedding for the weights in the final linear layer
        self.classifier.weight.data = torch.Tensor(weights).clone()
        print("final layer: code embedding initialised")
```

## Dataset and preprocessing
We used [the MIMIC-III dataset](https://mimic.physionet.org/) with the preprocessing steps from [caml-mimic](https://github.com/jamesmullenbach/caml-mimic) to generate the two dataset settings MIMIC-III and MIMIC-III-50. We also created a MIMIC-III-shielding dataset based on the NHS shielding ICD-10 codes.

## Pre-training of label embeddings
We used the Continous Bag-of-Words algorithm (CBoW) in Gensim word2vec (see [gensim.models.word2vec.Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec), on all label sets in the training data. Codes for training word and label embeddings are available in [```train_word_embedding.py```](https://github.com/acadTags/Explainable-Automated-Medical-Coding/blob/master/embeddings/train_word_embedding.py) and [```train_code_embedding.py```](https://github.com/acadTags/Explainable-Automated-Medical-Coding/blob/master/embeddings/train_code_embedding.py).

## Additional libraries for other models
For CNN, CNN+att, Bi-GRU models:
* PyTorch 0.3.0 with [caml-mimic](https://github.com/jamesmullenbach/caml-mimic)

For BERT models:
* PyTorch 1.0.0+ 
* [Huggingface Transformers](https://github.com/huggingface/transformers) for BERT training and BioBERT model conversion to PyTorch
* [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers) 0.20.2 for Multi-Label Classfication with BERT models
* [BioBERT](https://github.com/dmis-lab/biobert) for pre-trained BioBERT models.

## Other details 
* Using pre-trained BioBERT models: See answer from https://github.com/huggingface/transformers/issues/457#issuecomment-518403170.
* Training BERT for long documents: We adapted the sliding window approach from [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers) for multi-label classification. The idea is to treat a long document (discharge summaries in this project) as separate documents within the token length limit (sharing same set of labels) for training. During the testing stage, output averaged results of separated documents. The results of MIMIC-III-50 were based on this adaptation. The results of MIMIC-III were based on first 512 tokens only due to a memory usage above the 60G limit.

# Acknowledgement
* Our code is based on our previous implenmatation, [Automated-Social-Annotation](https://github.com/acadTags/Automated-Social-Annotation), which is based on [brightmart's implementation](https://github.com/brightmart/text_classification) of TextRNN and Hierarchical Attention Network under the MIT license.
* The MIMIC-III dataset is from https://mimic.physionet.org/ after request and training.
* Thanks for the kind answers from [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers).
