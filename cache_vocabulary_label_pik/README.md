Keep this folder in the program. The vocabulary indexes, label indexes, label similarity matrix and label subsumption matrix for each dataset are generated here after the first run. The vocabulary and label index ```.pik``` files are also available at Onedrive, [link](https://onedrive.live.com/?authkey=%21ACZVuCnEV2zDKow&id=22F95C44F607EC5B%21255141&cid=22F95C44F607EC5B).

For MIMIC-III-50, the relevant files to be automatically generated to this folder are:
* mimic3-ds-50-HAN_word_vocabulary.pik, the word-to-index and index-to-word dictionaries
* mimic3-ds-50-HAN_label_vocabulary.pik, the label-to-index and index-to-label dictionaries
* mimic3-ds-50_label_sim_0.pik, the label similarity matrix (not used in this study, but required for the program)
* icd9-50_label_sub.pik, the label subsumption matrix (not used in this study, but required for the program)

The vocabulary and label index files in the folder are required to run the Jupyter Notebook demo [```demo_HLAN_viz.ipynb```](https://github.com/acadTags/ExplainableAutomated-Medical-Coding/blob/master/HLAN/demo_HLAN_viz.ipynb).
