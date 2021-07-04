#train word embedding from the full discharge summaries in MIMIC-III.
#here we used the preprocessed discharge summary file disch_full.csv from CAML-MIMIC-III
#see In [8] in https://github.com/jamesmullenbach/caml-mimic/blob/master/notebooks/dataproc_mimic_III.ipynb to generate disch_full.csv

import sys
sys.path.append('../')
import csv
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
#from constants import MIMIC_3_DIR

dim = 100
MIMIC_3_DIR = '.' # define your disch_full.csv file path here

def check(rows):
    for row in rows:
       assert len(row) == 5

mimiciii_preprocessed_path = '%s/disch_full.csv' % MIMIC_3_DIR #to learn code embedding from training data labels

# initializing the titles and rows list 
fields = [] 
rows = [] 
       
# reading csv file 
with open(mimiciii_preprocessed_path, 'r', encoding='utf-8', errors='ignore') as csvfile: 
    # creating a csv reader object 
    csvreader = csv.reader(csvfile) 
      
    # extracting field names through first row 
    fields = next(csvreader)
    
    # extracting each data row one by one 
    for row in csvreader:
        rows.append(row) 

    # check whether the row lengths are correct
    check(rows)
    
    # get total number of rows 
    print("Total no. of rows: %d"%(csvreader.line_num)) 

data = [[n for n in row[4].split(' ')] for row in rows]

print (data[0], '\n', '\n', data[1])

path = get_tmpfile("%s/word-emb-mimic3-%s.model" % (MIMIC_3_DIR,dim)) # tr: from training data only
model = Word2Vec(data, size=dim, window=5, min_count=0, workers=4) #using cbow (sg=0 as default)
#class gensim.models.word2vec.Word2Vec(sentences=None, corpus_file=None, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=<built-in function hash>, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=(), max_final_vocab=None)
model.save("%s/word-emb-mimic3-tr-%s.model" % (MIMIC_3_DIR,dim))

#code for quick testing
model = Word2Vec.load("%s/word-emb-mimic3-%s.model" % (MIMIC_3_DIR,dim))
vector = model.wv['oncology']
print(vector)
print('vocab size:',len(model.wv.vocab)) # vocab size: 150854 for disch_full.csv