# Learning to Defer with Uncertainty
This repository contains implementation of the Learning to Defer with Uncertainty (LDU) algorithm, an approach which considers the model's predictive uncertainty when identifying the patient group to be evaluated by human experts. By identifying patients for whom the uncertainty of computer-aided diagnosis is estimated to be high and defers them for evaluation by human experts, the LDU algorithm can be used to mitigate the risk of erroneous computer-aided diagnoses in clinical
settings.

## Publication
“Incorporating Uncertainty in Learning to Defer Algorithms for Safe Computer-Aided Diagnosis” http://arxiv.org/abs/2108.07392

## Diagnostic Tasks
The LDU algorithm was tested on three diagnostic tasks using different types of medical data:<br />
(1) diagnosis of myocardial infarction using free-text discharge summaries from the MIMIC-III database.<br />
(2) diagnosis of any comorbidities (positive Charlson Index) using structured hospital records from the Heritage Health dataset.<br />
(3) diagnosis of pleural effusion and diagnosis of pneumothorax using chest x-ray images from the MIMIC-CXR database.<br />

## Data Sources
Discharge summaries from the MIMIC-III database: https://physionet.org/content/mimiciii/1.4/ <br />
Heritage Health dataset:  https://www.kaggle.com/c/hhp <br />
Chest x-ray images from the MIMIC-CXR database: https://physionet.org/content/mimic-cxr/2.0.0/ <br />

## Acknowledgement
If you found this repository useful, please consider citing our publication at http://arxiv.org/abs/2108.07392<br />

## References
[1] Mulyar A, Schumacher E, Rouhizadeh M, Dredze M. Phenotyping of Clinical Notes with Improved Document Classification Models Using Contextualized Neural Language Models. October 01, 2019:[arXiv:1910.13664 p.]. BERT Long Document Classification github repository: https://github.com/AndriyMulyar/bert_document_classification/blob/e9d9cd4dc810630f05661f777923632e3d8fe097/bert_document_classification/document_bert.py<br />
[2] Alsentzer E, Murphy JR, Boag W, Weng W-H, Jin D, Naumann T, et al. Publicly Available Clinical
BERT Embeddings. April 01, 2019: [arXiv:1904.03323 p.]. https://huggingface.co/emilyalsentzer/Bio_Discharge_Summary_BERT<br />
[3] Dissez G, Duboc G. CheXpert : A Large Chest X-Ray Dataset and Competition. https://github.com/gaetandi/cheXpert<br />



