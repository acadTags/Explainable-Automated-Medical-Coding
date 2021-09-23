# Explainable Automated Medical Coding

This project proposes an explainable automated medical coding approach based on Hierarchical Label-wise Attention Network and label embedding initialisation. The approach can be applied to multi-label text classification in any domains.

Detailed explanation of the approach is in 
* Explainable Automated Coding of Clinical Notes using Hierarchical Label-Wise Attention Networks and Label Embedding Initialisation, accepted for *Journal of Biomedical Informatics*, Oct 2020. [Preprint](https://arxiv.org/abs/2010.15728) is available on ArXiv.

A part of the results (especially regarding label embedding initialisation) was orally and virtually presented in [HealTAC 2020](http://healtex.org/healtac-2020/programme/) with [slides](https://drive.google.com/file/d/1XIhuwMuelmsFYvsXYABr7NJjnJTULrez/view) available.

Update (6 Sep 2021): added `--do_hierarchical_evaluation` flag for hierarchical evaluation using [CoPHE](https://github.com/modr00cka/CoPHE).

## Hierarchical Label-wise Attention Network
<p align="center">
    <img src="https://github.com/acadTags/Explainable-Automated-Medical-Coding/blob/master/HLAN/HLAN-architecture.PNG" width="600" title="Hierarchical Label-wise Attention Network">
</p>

The key computation graph is implemented in [```def inference_per_label(self)```](https://github.com/acadTags/Explainable-Automated-Medical-Coding/blob/cd9a360d5522d239d2ac5cef9a4ab507627bfa8d/HLAN/HAN_model_dynamic.py#L491) in ```./HLAN/HAN_model.py```.

# Requirements
* Python 3 (tested on versions 3.6.* and 3.7.*)
* Tensorflow 1.* (tested on versions 1.4.1, 1.8.0, 1.14.0, and 1.15.5)
* [Numpy](http://www.numpy.org/)
* [scikit-learn](http://scikit-learn.github.io/stable) for implementing evaluation metrics
* [Gensim](https://radimrehurek.com/gensim/) 3.* (tested on 3.8.3) for pre-training word and label embeddings with the word2vec algorithm
* [NLTK](https://www.nltk.org/) for tokenisation
* [Spacy](https://spacy.io/) 2.3.2 (before 3.x) for customised rule-based sentence parsing
* [TFLearn](http://tflearn.org/) for sequence padding

# How to Train on New Data
* First, format your data where each line has the format `doc__label__labelA labelB labelC`, for details see [`datasets`](https://github.com/acadTags/Explainable-Automated-Medical-Coding/tree/master/datasets). The data can be either split to train\[-validation\]-test (each split as a single file) or without split (only one data file). 
* Second, prepare word embeddings and the optional label embeddings using Gensim package, using existing embeddings or those trained from your texts (e.g. using script in [`embeddings`](https://github.com/acadTags/Explainable-Automated-Medical-Coding/tree/master/embeddings), also see the [`notebook from caml-mimic`](https://github.com/jamesmullenbach/caml-mimic/blob/master/notebooks/dataproc_mimic_III.ipynb) for embedding from MIMIC-III. The trained embeddings on MIMIC-III are [downloadable](https://onedrive.live.com/?authkey=%21ACZVuCnEV2zDKow&id=22F95C44F607EC5B%21255141&cid=22F95C44F607EC5B)).
* Third, add a new data block ([`if FLAGS.dataset == "YOUR_DATASET_NAME":`](https://github.com/acadTags/Explainable-Automated-Medical-Coding/blob/master/HLAN/HAN_train.py#L142)) with variables specified in `HAN_train.py`. Please read closely the example code block and comments provided.
* Finally, run commands (e.g. `python HAN_train.py --dataset YOUR_DATASET_NAME`) with arguments, see details in [`Training the models`](https://github.com/acadTags/Explainable-Automated-Medical-Coding/blob/master/README.md#training-the-models).

# Jupyter Notebook Demo with MIMIC-III ICD Coding
* First, ensure that you have requested the MIMIC-III dataset, see [the official page to request MIMIC-III](https://mimic.physionet.org/gettingstarted/access/). Place the files ```D_ICD_DIAGNOSES.csv``` and ```D_ICD_PROCEDURES.csv``` under the ```knowledge_bases``` folder.

* Second, download the files in ```checkpoints```, ```cache_vocabulary_label_pik```, and ```embeddings``` folders from Onedrive ([link](https://onedrive.live.com/?authkey=%21ACZVuCnEV2zDKow&id=22F95C44F607EC5B%21255141&cid=22F95C44F607EC5B)).

* Third, run the Jupyter Notebook demo [```demo_HLAN_viz.ipynb```](https://github.com/acadTags/Explainable-Automated-Medical-Coding/blob/master/HLAN/demo_HLAN_viz.ipynb) and try with your own discharge summaries or those in the MIMIC-III dataset. By setting the ```to_input``` in the notebook as ```True```, the notebook will ask you to input or paste a discharge summary; otherwise, you can save your discharge summaries, each in a line, under the ```..\dataset\``` folder and replace the ```filename_to_predict``` to the filename (see ```Section 2.part2``` in the notebook). After running, the predictions are displayed with label-wise attention visualisations. The attention visualisations are further stored as ```.xlsx``` files in the ```..\explanation\``` folder. 

# Content
* ```./HLAN/HAN_train.py``` contains code for configuration and training
* ```./HLAN/HAN_model.py``` contains the computational graph, loss function and optimisation
* ```./HLAN/data_util_gensim.py``` contains code for input and target generation
* ```./HLAN/demo_HLAN_viz.ipynb``` and ```./HLAN/model_predict_util.py``` contains code for the demo based on Jupyter Notebook and the helper functions
* ```./HLAN/evaluation_setup.py``` and ```./HLAN/multi_level_eval.py``` contains code from [CoPHE](https://github.com/modr00cka/CoPHE) for the hierarchical evaluation of multi-label classification
* ```./embeddings``` contains self-trained word2vec embeddings: word embeddings and label embeddings
* ```./datasets``` contains the datasets used
* ```./checkpoints``` contains the checkpoints of HLAN, HA-GRU, and HAN models trained from the author on the MIMIC-III datasets
* ```./explanations``` contains the Excel sheets displaying the attention visualisation, generated after running the demo in ```./HLAN/demo_HLAN_viz.ipynb```
* ```./knowledge_bases``` contains knowledge sources used for label subsumption relations and the ICD code description files
* ```./cache_vocabulary_label_pik``` stores the cached .pik files about vocabularies and labels
* ```./results-HEALTAC 2020``` contains the CNN, CNN+att, Bi-GRU, BERT results with label embedding initilisation

# Key Configurations and Further Details

## Training the models

```--dataset``` is set to ```mimic3-ds-50``` for MIMIC-III-50, ```mimic-ds``` for MIMIC-III, and ```mimic-ds-shielding-th50``` for MIMIC-III-shielding.

To use label embedding initialisation (*+LE*), set ```--use_label_embedding``` to ```True```; otherwise, set it to ```False```.

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

By setting ```running_times``` as ```k```, it will report averaged results and standard deviations with ```k``` runs. For example, ```--running_times 10```.

For hierarchical evaluation results using [CoPHE](https://github.com/modr00cka/CoPHE), add the flag `--do_hierarchical_evaluation=True`.

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
* Fine-tuning BERT with long documents: We adapted the sliding window approach from [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers) for multi-label classification. The idea is to treat a long document (discharge summaries in this project) as separate documents within the token length limit (sharing the same set of labels) for training. During the testing stage, the BERT model outputs the averaged prediction score of the separated documents. The results of MIMIC-III-50 were based on this adaptation. The results of MIMIC-III were based on first 512 tokens only due to a memory usage above the 60G limit.

# Acknowledgement
* Our code is based on our previous implementation, [Automated-Social-Annotation](https://github.com/acadTags/Automated-Social-Annotation), which is based on [brightmart's implementation](https://github.com/brightmart/text_classification) of TextRNN and Hierarchical Attention Network under the MIT license.
* The MIMIC-III dataset is from https://mimic.physionet.org/ after request and training.
* The hierarchical evaluation is based on [CoPHE](https://github.com/modr00cka/CoPHE), consented by the author modr00cka.
* Thanks for the kind answers from [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers).
