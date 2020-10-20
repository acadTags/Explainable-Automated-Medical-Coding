# label-embedding-initialisation

This project proposes a label embedding initialisation approach to improve multi-label classification (and especially for automated medical coding).

<p align="center">
    <img src="https://github.com/acadTags/Explainable-Automated-Medical-Coding/blob/master/results-HealTAC%202020/label-embedding-init-figure.PNG" width="400" title="Label Embedding Initialisation for Deep-Learning-Based Multi-Label Classification">
</p>

Key part of the implementation of label embedding initiailisation:
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

# Results
See the folders for detailed results (mean and standard deviation)of various Micro-averaged, Macro-averaged, and example-based metrics) of each model. Brief results are in the figures below, where ```+LE``` means the model with label embedding initialisation.

<p align="center">
    <img src="https://github.com/Explainable-Automated-Medical-Coding/blob/master/results-HealTAC%202020/mimic-iii-results.JPG" width="435" title="Results of the MIMIC-III dataset">
    <img src="https://github.com/Explainable-Automated-Medical-Coding/blob/master/results-HealTAC%202020/mimic-iii-50%20results.JPG" width="435" title="Results of the MIMIC-III-50 dataset">
</p>

# Requirements
* Python 3.6.*
* PyTorch 0.3.0 with [caml-mimic](https://github.com/jamesmullenbach/caml-mimic) for CNN,BiGRU,CNN+att models for CNN,BiGRU,CNN+att models
* PyTorch 1.0.0+ for BERT models
* [Huggingface Transformers](https://github.com/huggingface/transformers) for BERT training and BioBERT model conversion to PyTorch
* [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers) 0.20.2 for Multi-Label Classfication with BERT models
* [Gensim](https://radimrehurek.com/gensim/) for pre-training label embeddings with the word2vec algorithm
* [BioBERT](https://github.com/dmis-lab/biobert) for pre-trained BioBERT models.

# Dataset and preprocessing
We used [the MIMIC-III dataset](https://mimic.physionet.org/) with the preprocessing steps from [caml-mimic](https://github.com/jamesmullenbach/caml-mimic) to generate the two dataset settings MIMIC-III and MIMIC-III-50.

# Pre-training of label embeddings
We used the continous bag-of-words algorithm (cbow) in Gensim word2vec (see [gensim.models.word2vec.Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec), on all label sets in the training data.

# Other details 
* Using pre-trained BioBERT models: See answer from https://github.com/huggingface/transformers/issues/457#issuecomment-518403170.
* Training BERT for long documents: We adapted the sliding window approach from [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers) for multi-label classification. The idea is to treat a long document (discharge summaries in this project) as separate documents within the token length limit (sharing same set of labels) for training. During the testing stage, output averaged results of separated documents. The results of MIMIC-III-50 were based on this adaptation. The results of MIMIC-III were based on first 512 tokens only due to a memory usage above the 60G limit.

# Acknowledgement
* The MIMIC-III dataset is from https://mimic.physionet.org/ after request and training.
* Thanks for the kind answers from [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers).
