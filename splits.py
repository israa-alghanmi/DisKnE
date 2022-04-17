import sys
import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import time
import datetime
import random
from owlready2 import *
from owlready2.pymedtermino2 import *
from owlready2.pymedtermino2.umls import *
from sklearn.utils import shuffle
import gc 




def parse_arguments():
    """Read arguments from a command line."""    
    parser = argparse.ArgumentParser(description='Experiment setup - Arguments get parsed via --commands')
    parser.add_argument('--data_path', dest='data_path', type=str, default="",
     help='Required- DisknE main dataset path')
    parser.add_argument('--umls_path', dest='umls_path', type=str, default="./umls-2020AA-full.zip",
     help='Required- UMLS file path')
    parser.add_argument('--type', dest='type', type=str, default="similar",
     help='Required- For medical knowledge choose either "similar" or "random", for terminological knowledge: "terminological_similar" or "terminological_random"')
    parser.add_argument('--category', dest='cate', type=str, default="",
     help='Required only for medical knowledge in which a category need to be specified: "symptoms_diseases" or "treatments_diseases" or "tests_diseases" "procedure_diseases", for terminological skip. ')
    parser.add_argument('--save_dir', dest='save_dir', type=str, default="",
     help='Required- Directory to store DisknE diseases splits')
    args = parser.parse_args() 
    return args

def decode_label(label):
    decode_map = {"non-entailment":0,"entailment":1}
    return decode_map[label]


def read_csv_to_df(input_path):
    df= pd.read_csv(input_path)
    df.label= df.label.apply(lambda x: decode_label(x))
    return df
    

def prepare_splits(concept,matches,CUI):

    all_data= read_csv_to_df(args.data_path)
    num_classes= len(set(all_data['label']))    
    new_test_indexes=[]    
    remove_list=[] 
    
    
    for i in range (0,len(all_data)):
        try:
            #--Find exact matches of the target disease anywhere in the dataset (including the premise), remove resulted examples from the main dataset
            if  str(all_data['CUI'][i])==CUI  or str(all_data['original_cui'][i])==CUI or any(x in str(all_data['P'][i]).lower() for x in matches):
                remove_list.append(i)

            #if no category specified
            if args.cate=='' and str(all_data['original_cui'][i])==CUI:
                      new_test_indexes.append(i)  
            else:
                
                if all_data['category'][i]==args.cate and str(all_data['original_cui'][i])==CUI:
                    new_test_indexes.append(i)  
        except: continue    
    #<---------------------------------------------------->
    # <--Test set-->
    new_test_df= pd.DataFrame(columns=all_data.columns)        
    for i in new_test_indexes:
        new_test_df= new_test_df.append(all_data.iloc[i],ignore_index=True)
    new_test_df= new_test_df.drop_duplicates(subset=['P', 'Disease'], keep='first',ignore_index=True)
    new_test_df = shuffle(new_test_df,random_state=84)
    #<---------------------------------------------------->
    #<--- Training and vaildation sets-->
    for i in remove_list:
        all_data=all_data.drop(i)
    all_data['Disease']=all_data['Disease'].map(lambda x: x if type(x)!=str else x.lower())
    all_data['Disease']=all_data['Disease'].map(lambda x: x if type(x)!=str else x.strip())
    all_data=all_data.drop_duplicates(subset=['P', 'Disease'], keep='first',ignore_index=True)
    all_data = shuffle(all_data, random_state=42)
    training_df, dev_df= train_test_split(all_data, test_size=0.10, shuffle=False)
    print(str(disease) ,' Done...', flush=True)
    return new_test_df,training_df,dev_df
    
    

if __name__ == '__main__':  
    
    if torch.cuda.is_available():        
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    args = parse_arguments()
    
    #<-------- PyMedTermino SETUP -------->
    
    default_world.set_backend(filename = "./pym.sqlite3")
    # If running for a second time, comment the follwing two lines as the pym file has already been created in the specified path
    import_umls(args.umls_path, terminologies = ["ICD10", "SNOMEDCT_US", "CUI"])
    default_world.save()
    PYM = get_ontology("http://PYM/").load()
    CUI2 = PYM["CUI"]
    SNOMEDCT_US = PYM["SNOMEDCT_US"]
    

    
    # Considered diseases for Medical knowledge
    if args.type=='random' or args.type=='similar':
        diseases=['C0010054', 'C0018802', 'C0948089','C0038454', 'C0020538','C0018799',
                  'C0027051','C0018801','C0042029','C0024115','C0023890','C0022661',
                  'C0020456','C0032285','C0027765','C0002871','C1145670','C0035078',
                  'C0034063','C0876973','C0024117','C0003507','C0018790','C0242966',
                  'C0155626','C0149871','C0035204','C0023895','C0022660','C0020473',
                  'C0017168','C0746731','C1561643','C0403447','C0275518','C0085096',
                  'C0041912','C0034065','C0022658','C0022116','C0018939','C0011334',
                  'C0008350','C0008311','C0007785','C0004364','C0003962','C0017178',
                  'C1410927','C1260873','C0878544','C0856747','C0746883','C0745136',
                  'C0742078','C0729552','C0585105','C0546817','C0458219','C0340643',
                  'C0340288','C0340285','C0272285','C0264733','C0264714','C0264490',
                  'C0263854','C0239946','C0206062','C0162871','C0162429','C0155668',
                  'C0151740','C0085605','C0085581','C0042373','C0041952','C0040997',
                  'C0040583','C0040558','C0037278','C0037199','C0032227','C0031350',
                  'C0031039','C0030469','C0030305','C0028754','C0027868','C0025517',
                  'C0023212','C0023211','C0022638','C0021390','C0020676','C0020443',
                  'C0020305','C0020258','C0019372','C0019064','C0017160','C0015544',
                  'C0014868','C0014335','C0014179','C0012569','C0011880','C0011860',
                  'C0011854','C0011175','C0011168','C0009319','C0008325','C0007286',
                  'C0007282','C0007193','C0007177','C0004623','C0004610','C0003869',
                  'C0003504','C0002965','C0002940','C0001175','C0004153']

    # Considered diseases for terminological knowledge   
    if args.type=='terminological_similar' or args.type=='terminological_random':
        diseases=['C0001339','C0002871','C0002940','C0002965','C0003486','C0003504',
              'C0003507','C0003864','C0004096','C0004364','C0004763','C0007194',
              'C0007273', 'C0007282','C0007766','C0007787','C0008325','C0008350',
              'C0009450','C0010054','C0010709','C0011168','C0011849','C0011854',
              'C0011860','C0011880','C0012813','C0012819','C0013182','C0014009',
              'C0014118','C0014544','C0015230','C0017152','C0017168','C0018099',
              'C0018790','C0018799','C0018801','C0018802','C0019196','C0019348',
              'C0020456','C0020538','C0020542','C0020676','C0021843','C0022104',
              'C0022660','C0022661','C0023890','C0023895','C0024115','C0024117',
              'C0024131','C0025517','C0026267','C0026269','C0027051','C0027765',
              'C0030283','C0030305','C0030312','C0032227','C0032285','C0032290',
              'C0032326','C0032371','C0034063','C0034155','C0035021','C0035078',
              'C0035439','C0037199','C0037315','C0038449','C0038454','C0038833',
              'C0040583','C0042029','C0042345','C0085615','C0149721','C0149801',
              'C0149871','C0149931','C0150055','C0151517','C0152020','C0152025',
              'C0155626','C0155909','C0162316','C0162871','C0235325','C0235329',
              'C0235480','C0236073','C0242231','C0262655','C0333559','C0398623',
              'C0403447','C0442874','C0457949','C0494475','C0520679','C0558382',
              'C0565599','C0587044','C0679247','C0694549','C0742758','C0876973',
              'C0949083','C1561643']



    # Get a list of all diseases as mentioned in MedNLI filtered dataset
    org_df=read_csv_to_df(args.data_path) 
    org_df =org_df[~org_df.pair_id.str.contains(',neg')]
    df2 = pd.DataFrame({'CUI':org_df.CUI.unique()})
    df2['Disease'] = [list(set(org_df['Disease'].loc[org_df['CUI'] == x['CUI']]))  for _, x in df2.iterrows()]

    # --generate train, validation and test splits per disease
    for disease in diseases:
    
        try:
            concept = CUI2[str(disease)]           
            matches= []
            matched= list(df2.loc[df2.CUI==str(concept.name)]['Disease'])[0]
            for x in matched:
                matches.append(x.lower())
            cn = concept >> SNOMEDCT_US
            if len(cn) != 0:
                cn2=cn.pop()
            for x in concept.synonyms:
                matches.append(x.lower())

        except: continue
       
        new_test_df,training_df,dev_df=prepare_splits(concept,matches,str(disease))
       
        #Consider diseases with at-least two positive examples
        if list(new_test_df['label'].values).count(1) < 2: continue
        
        with open(args.save_dir+"/diseases_list.txt", "a") as myfile:
          myfile.write('_{}'.format(concept.label[0])+' \n') 
         
        # CSV files for each disease        
        training_df.to_csv(args.save_dir+'/DisKnE_train_{}.csv'.format(concept.label[0]), index=False) 
        dev_df.to_csv(args.save_dir+'/DisKnE_dev_{}.csv'.format(concept.label[0]), index=False)
        new_test_df.to_csv(args.save_dir+'/DisKnE_test_{}.csv'.format(concept.label[0]), index=False)
          
