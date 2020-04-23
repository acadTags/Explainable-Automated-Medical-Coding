The folder presents results of the CNN+att model (or called CAML in the study Mullenbach et al., 2018) on the MIMIC-III and MIMIC-III-50 datasets.

# Files
MIMIC-III-50 results are in ```caml_result_50.xlsx``` and ```caml_ce_result_50.xlsx```.

MIMIC-III results are in ```caml_result_full.xlsx``` and ```caml_ce_result_full.xlsx```.

where the files with ```ce``` include results with label embedding initialisation (or "code embedding").

Every row after the header in each file contains training and testing results of a random run. 

The last row is the averaged results of 10 runs with standard deviation.

# Metrics
The metrics follow the study in (Mullenbach et al., 2018)

* Validation results:

  Macro-averaging accuracy, prevision, recall, f1 and AUC: ```acc_macro```, ```prec_macro```, ```rec_macro```, ```f1_macro```, ```auc_macro```
  
  Micro-averaging accuracy, prevision, recall, f1 and AUC: ```acc_micro```, ```prec_micro```, ```rec_micro```, ```f1_micro```, ```auc_micro```

  Example-based precision, recall, and f1 at k: ```rec_at_k```, ```prec_at_k```, ```f1_at_k```
  
  Loss: ```loss_dev```

* Testing results:
  
  Macro-averaging accuracy, prevision, recall, f1 and AUC: ```acc_macro_te```, ```prec_macro_te```, ```rec_macro_te```, ```f1_macro_te```, ```auc_macro_te```	
  
  Micro-averaging accuracy, prevision, recall, f1 and AUC: ```acc_micro_te```, ```prec_micro_te```, ```rec_micro_te```, ```f1_micro_te```, ```auc_micro_te```	
 
  Example-based precision, recall, and f1 at k: ```rec_at_k_te```, ```prec_at_k_te```, ```f1_at_k_te```
  
  Loss: ```loss_test_te```

[1] Mullenbach J, Wiegreffe S, Duke J, Sun J, Eisenstein J. Explainable Prediction of Medical Codes from Clinical Text. InProceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers) 2018 Jun (pp. 1101-1111).
