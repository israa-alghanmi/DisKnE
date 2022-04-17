# Disease Knowledge Evaluation - DisKnE
This repository contains Python scripts for evaluating and constructing the Disease Knowledge Evaluation benchmark (DisKnE) introduced in the paper: [__Probing Pre-Trained Language Models for Disease Knowledge__.](https://aclanthology.org/2021.findings-acl.266.pdf)
___

## DisKnE Construction
#### Requirements

License and access to the following resources are needed to reconstruct the dataset:
##### 1. Datasets
* [MedNLI](https://physionet.org/content/mednli/1.0.0/)
* [MEDIQA-NLI](https://physionet.org/content/mednli-bionlp19/1.0.1/)
##### 2. [UMLS](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html): We use umls-2020AA-full version.
#### DisKnE annotation files
Each annotation file represents a variant of DisKnE, to assess medical or terminological knowledge: 
* Medical-similar, terminological-similar: Negative examples are selected as being the most similar to the target disease from a pre-defined list.
* Medical-random, terminological-random: Negative examples are selected at random from a pre-defined list. 
#### Installing the required packages
``` 
pip install requirements.txt
``` 

#### Required Arguments

``` 
usage: generate.py [-h] --train TRAINING_PATH --dev DEV_PATH --test1
                   TESTING_PATH --test2 TESTING2_PATH --test2_gt
                   TESTING2_GT_PATH --umls UMLS_PATH --annotaion_f ANNOTAION_F
                   --dest CONSTRUCTED_FILE

setup - Arguments get parsed via --commands

required arguments:
  --train TRAINING_PATH
                        The path of MedNLI training file
  --dev DEV_PATH        The path of MedNLI dev file
  --test1 TESTING_PATH  The path of MedNLI testing file
  --test2 TESTING2_PATH
                        The path of MEDIQA-NLI
  --test2_gt TESTING2_GT_PATH
                        The path of MEDIQA-NLI ground truth file
  --umls UMLS_PATH      UMLS file path
  --annotaion_f ANNOTAION_F
                        Annotaion file path
  --dest CONSTRUCTED_FILE
                        The path to store the constructed dataset
``` 

#### Example
``` 
python "generate.py" --train "./mli_train_v1.jsonl" --dev  "./mli_dev_v1.jsonl" --test1 "./mli_test_v1.jsonl" --test2 "./mednli_bionlp19_shared_task.jsonl" --test2_gt "./mednli_bionlp19_shared_task_ground_truth.csv" --annotaion_f "./DisknE_similar_annotation.csv" --dest "./DisknE_medical_similar.csv" --umls "./umls-2020AA-full.zip"
``` 


#### Training-Test Splits

``` 
usage: splits.py [-h] [--data_path DATA_PATH] [--umls_path UMLS_PATH]
                 [--pym_path PYM_PATH] [--type TYPE] [--category CATE]
                 [--save_dir SAVE_DIR]

Experiment setup - Arguments get parsed via --commands

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Required- DisknE main dataset path
  --umls_path UMLS_PATH
                        Required- UMLS file path
  --type TYPE           Required- For medical knowledge choose either
                        "similar" or "random", for terminological knowledge:
                        "terminological_similar" or "terminological_random"
  --category CATE       Required only for medical knowledge in which a
                        category need to be specified: "symptoms_diseases" or
                        "treatments_diseases" or "tests_diseases"
                        "procedure_diseases", for terminological skip.
  --save_dir SAVE_DIR   Required- Directory to store DisknE diseases
                        splits
``` 

#### Example 
```
python "splits.py" --data_path "./DisknE_medical_similar.csv" --umls_path "./umls-2020AA-full.zip" --type 'similar' --category 'symptoms_diseases' --save_dir "./DisKnE"
```

## Evaluation


``` 
usage: evaluate.py [-h] [--data_dir DATA_DIR] [--BERT_path BERT_PATH]
                   [--seed_val SEED_VAL]

Experiment setup - Arguments get parsed via --commands

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Required- Directory path for DisknE diseases splits
  --BERT_path BERT_PATH
                        Required- Path to model
  --seed_val SEED_VAL   Required- Seed value

``` 

### Example
```
python "evaluate.py" --data_dir "./DisKnE" --BERT_path "emilyalsentzer/Bio_ClinicalBERT" --seed_val 12345
``` 

___
## Citation
``` 
@inproceedings{alghanmi-etal-2021-probing,
    title = "Probing Pre-Trained Language Models for Disease Knowledge",
    author = "Alghanmi, Israa  and
      Espinosa Anke, Luis  and
      Schockaert, Steven",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.266",
    doi = "10.18653/v1/2021.findings-acl.266",
    pages = "3023--3033",
}

``` 
