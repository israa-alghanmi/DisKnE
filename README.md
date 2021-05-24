# Disease Knowledge Evaluation - DisKnE
This repository contains Python scripts for evaluating and constructing the Disease Knowledge Evaluation benchmark (DisKnE) introduced in the paper: __Probing Pre-Trained Language Models for Disease Knowledge__.
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
``` python
pip install requirements.txt
``` 

#### Required Arguments

``` python
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
``` python
python "generate.py" --train "./mli_train_v1.jsonl" --dev  "./mli_dev_v1.jsonl" --test1 "./mli_test_v1.jsonl" --test2 "./mednli_bionlp19_shared_task.jsonl" --test2_gt "./mednli_bionlp19_shared_task_ground_truth.csv" --annotaion_f "./DisknE_similar_annotation.csv" --dest "./DisknE_medical_similar.csv" --umls "./umls-2020AA-full.zip"
``` 


## Evaluation


``` python
usage: evaluation.py [-h] [--data_path DATA_PATH] [--umls_path UMLS_PATH]
                     [--pym_path PYM_PATH] [--type TYPE] [--category CATE]
                     [--seed_val SEED_VAL] [--input_type INPUT_TYPE]

Experiment setup - Arguments get parsed via --commands

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Required- DisknE dataset path
  --umls_path UMLS_PATH
                        Required- UMLS file path
  --pym_path PYM_PATH   Required- Path to store pym file, needed to import
                        UMLS in Python
  --type TYPE           Required- For medical knowledge choose either:
                        1.similar 2.random, for terminological knowledge:
                        1.terminological_similar 2. terminological_random
  --category CATE       Required only for medical knowledge where
                        a category need to be specified: 1.symptoms_diseases
                        2.treatments_diseases 3.tests_diseases
                        4.procedure_diseases, for terminological skip.
  --seed_val SEED_VAL   Required- Seed value
  --input_type INPUT_TYPE
                        Optional- Use for running the canonicalized
                        hypothesis-only baseline: disease_only
``` 

### Examples
* #### Medical-random for tests to diseases category
``` python
python "evaluation.py" --data_path "./DisknE_medical_random.csv" --umls_path "./umls-2020AA-full.zip" --type "random" --category "tests_diseases" --seed_val=12345
``` 

* #### Terminological-similar 
``` python
python "evaluation.py" --data_path "./DisknE_medical_random.csv" --umls_path "./umls-2020AA-full.zip" --type "terminological_similar" --seed_val=12345
``` 
___
<!--- ## Citation-->
