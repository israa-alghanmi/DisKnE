# Disease Knowledge Evaluation - DisKnE
This repository contains Python scripts for evaluating and constructing the Disease Knowledge Evaluation benchmark (DisKnE) proposed in the paper: Probing Pre-Trained Language Models for Disease Knowledge.
___

## DisKnE Construction
### Requirements:
License and access to the following resources are needed to reconstruct the dataset:
#### 1. Datasets
* [MedNLI](https://physionet.org/content/mednli/1.0.0/)
* [MEDIQA-NLI](https://physionet.org/content/mednli-bionlp19/1.0.1/)
#### 2. [UMLS](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html)
We use umls-2020AA-full version.
#### 3. DisKnE annotation files
Each annotation file represents a variant of DisKnE, to assess medical or terminological knowledge: 
* Medical-similar, terminological-similar: Negative examples are selected as being most similar to the target disease from a pre-defined list.
* Medical-random, terminological-random: Negative examples are selected at random from a pre-defined list. 
#### 4. Installing the required packages
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
python3 "generate.py" --train "./mli_train_v1.jsonl" --dev  "./mli_dev_v1.jsonl" --test1 "./mli_test_v1.jsonl" --test2 "./mednli_bionlp19_shared_task.jsonl" --test2_gt "./mednli_bionlp19_shared_task_ground_truth.csv" --annotaion_f "./DisknE_similar_annotation.csv" --dest "./DisknE_medical_similar.csv" --umls "./umls-2020AA-full.zip"
``` 

___
## Citation
