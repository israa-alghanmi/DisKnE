import json
import pandas as pd
from owlready2 import *
from owlready2.pymedtermino2 import *
from owlready2.pymedtermino2.umls import *
import numpy as np
import argparse

def parse_arguments():
    """Read arguments from a command line."""
    parser = argparse.ArgumentParser(description='setup - Arguments get parsed via --commands')
    parser.add_argument('--train', dest='training_path', type=str, default="./mli_train_v1.jsonl",
        help='The path of MedNLI dataset training file')
    parser.add_argument('--dev', dest='dev_path', type=str, default="./mli_dev_v1.jsonl",
        help='The path of MedNLI dataset dev file ')
    parser.add_argument('--test1', dest='testing_path', type=str, default="./mli_test_v1.jsonl",
        help='The path of MedNLI dataset testing file ')
    parser.add_argument('--test2', dest='testing2_path', type=str, default="./mednli_bionlp19_shared_task.jsonl", help='The path of MEDIQA-NLI')
    parser.add_argument('--test2_gt', dest='testing2_gt_path', type=str, default="./mednli_bionlp19_shared_task_ground_truth.csv'", help='The path of MEDIQA-NLI ground truth file')
    parser.add_argument('--umls', dest='umls_path', type=str, default="./umls-2020AA-full.zip",
        help='UMLS file path')
    parser.add_argument('--annotaion_f', dest='annotaion_f', type=str, default="./DisknE_terminological_random_annotation.csv",
        help='Annotaion file path')
    parser.add_argument('--dest', dest='constructed_file', type=str, default="./DisknE_terminological_random.csv",
        help='The path to store the constructed dataset')
    args = parser.parse_args()
    return args



def load_jsonl(input_path) -> list:
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data



def get_df_from_json(data):
    sentence1=[]
    sentence2=[]
    pairID=[]
    gold_label=[] 
    for i in range (0,len(data)):
            sentence1.append(data[i]['sentence1'])
            sentence2.append(data[i]['sentence2'])
            pairID.append(data[i]['pairID'])
            gold_label.append(data[i]['gold_label'])
    d = {'sentence1':sentence1,'sentence2':sentence2,'pairID':pairID,'gold_label':gold_label }
    df = pd.DataFrame.from_dict(d)
    return df


def construct_dataset(df,annotaion_df):
    annotaion_df = annotaion_df.replace(np.nan, '', regex=True)
    annotaion_df['P']=''
    annotaion_df['Disease']=''
    for i in range(0,len(annotaion_df)):
        if ',ng' not in annotaion_df['pair_id'][i]:
            row=df.loc[df['pairID'] == annotaion_df['pair_id'][i]]
            if len(row)>0 :
                annotaion_df['P'][i]=str(row['sentence1'].iloc[0])
                annotaion_df['Disease'][i]=str(row['sentence2'].iloc[0])[int(annotaion_df['start_index'][i]):int(annotaion_df['end_index'][i]+1)].lower()
                annotaion_df['Disease'][i]=str(annotaion_df['Disease'][i]).strip()
        if ',ng' in annotaion_df['pair_id'][i]:   
            pair_id=annotaion_df['pair_id'][i].replace(',ng','')
            row=df.loc[df['pairID'] == pair_id]
            if len(row)>0 :
                annotaion_df['P'][i]=str(row['sentence1'].iloc[0])
                df_CUI=annotaion_df['CUI'][i]
                concept = CUI2[df_CUI]
                concept = concept >> SNOMEDCT_US
                concept=concept.pop()
                annotaion_df['Disease'][i]=str(concept.label.first()).lower()
    return annotaion_df
    

if __name__ == '__main__':  
    
    args = parse_arguments()
    
    #<<-------------Prepare MEDIQA-NLI --------------->>
    test2_data= load_jsonl(args.testing2_path)
    test2_labels= pd.read_csv(args.testing2_gt_path)
    for i in range( 0,len(test2_labels)):
        for j in range (0,len(test2_data)):
            if test2_labels['pair_id'][i] == test2_data[j]['pairID']:
                   test2_data[j]['gold_label']= test2_labels['label'][i]
    test2_df= get_df_from_json(test2_data)
    #<<-----------Load Data----------------->>
    training_data= load_jsonl(args.training_path)
    dev_data= load_jsonl(args.dev_path)
    test_data= load_jsonl(args.testing_path)
    training_df= get_df_from_json(training_data)
    dev_df= get_df_from_json(dev_data)
    test_df= get_df_from_json(test_data)
    frames = [training_df, dev_df, test_df,test2_df]
    df= pd.concat(frames)
    #<<-------------Load UMLS--------------->>
    default_world.set_backend(filename = "pym3.sqlite3")
    #import_umls(args.umls_path, terminologies = ["ICD10", "SNOMEDCT_US", "CUI"])
    #default_world.save()
    PYM = get_ontology("http://PYM/").load()
    CUI2 = PYM["CUI"]
    SNOMEDCT_US = PYM["SNOMEDCT_US"]
    #<<-----------construct the dataset----------------->>
    annotaion_df= pd.read_csv(args.annotaion_f)
    new_df=construct_dataset(df,annotaion_df)
    new_df.to_csv(f'{args.constructed_file}',index=False)        

            