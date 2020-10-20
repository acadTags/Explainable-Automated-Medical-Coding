The folder presents results of the BERT ([BioBERT](https://github.com/dmis-lab/biobert)) model on the MIMIC-III and MIMIC-III-50 datasets.

# Files
MIMIC-III-50 results are in ```bert-50.csv``` and ```bert-50-ce.csv```.

MIMIC-III results are in ```bert-50-full.csv``` and ```bert-full-ce.csv```.

The files with ```ce``` include results with label embedding initialisation (or "code embedding").

Every row after the header in each file contains training and testing results of a random run. 

The last row is the averaged results of 5 runs with standard deviation.

# Metrics
The metrics partly follow the study in (Mullenbach et al., 2018)

* Testing results:

  Macro-averaging accuracy, prevision, recall, f1 and AUC: ```acc_macro```, ```prec_macro```, ```rec_macro```, ```f1_macro```, ```auc_macro```
  
  Micro-averaging accuracy, prevision, recall, f1 and AUC: ```acc_micro```, ```prec_micro```, ```rec_micro```, ```f1_micro```, ```auc_micro```

  Example-based precision, recall, and f1 at k: ```prec_at_k```, ```rec_at_k```, ```f1_at_k```
  
  Example-based accuracy, precision, recall, and f1 with 0.5 as threshold: ```acc```, ```prec```, ```rec```, ```f1```
  
  Hamming loss: ```hamming_loss```
  
  Label ranking average precision: ```LRAP```

[1] Mullenbach J, Wiegreffe S, Duke J, Sun J, Eisenstein J. Explainable Prediction of Medical Codes from Clinical Text. InProceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers) 2018 Jun (pp. 1101-1111).

Reference for the BioBERT model:

Lee J, Yoon W, Kim S, Kim D, Kim S, So CH, Kang J. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics. 2020 Feb 15;36(4):1234-40.
