# Datasets

Datasets were originally from MIMIC-III and the data pre-processed using the script provided from the [caml-mimic](https://github.com/jamesmullenbach/caml-mimic) Github reporitory.

To further preprocess the data suitable for this program, please ensure that:
* Every line of the data contain a document (e.g. a discharge summary) and its labels (e.g. ICD-9 codes). The document and the labels are separated by a ```__label__``` sign. Labels are separated by a white space.

This data format is the same as in the ```bibsonomy_preprocessed_merged_final.txt``` from the GitHub project [Automated-Social-Annotation](https://github.com/acadTags/Automated-Social-Annotation/tree/master/datasets).

The MIMIC-III and MIMIC-III-50 are the same documents and data splits as in [caml-mimic](https://github.com/jamesmullenbach/caml-mimic) (Mullenbach et al., 2018).

The MIMIC-III-shielding dataset was created by 
* (1) Selecting the ICD-9 codes which appeared no less than 50 times and those matched to the [ICD-10 codes provided by the NHS for shielding patients identification during the COVID-19 pandemic](https://digital.nhs.uk/binaries/content/assets/website-assets/services/high-risk-shielded-patient-list-identification-methodology/spl-icd10-opcs4-disease-groups-v2.0.xlsx). There were finally 20 ICD-9 codes.
* (2) Selecting the documents (in each of the train, validation, and test set in the MIMIC-III dataset preprocessed from [caml-mimic](https://github.com/jamesmullenbach/caml-mimic)) which contain at least one of the shielding-related ICD-9 codes. The data split was kept as the same as in MIMIC-III after this filtering.

We provided the list of ICD-9 codes of MIMIC-III-50 and MIMIC-III-shielding in the supplementary material (Table S1-S2) of [the paper](https://arxiv.org/abs/2010.15728).
