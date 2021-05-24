import sys
import argparse
import json
import pandas as pd
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report,f1_score,accuracy_score
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
from torch.optim import Adam




def parse_arguments():
    """Read arguments from a command line."""
    parser = argparse.ArgumentParser(description='Experiment setup - Arguments get parsed via --commands')
    parser.add_argument('--data_path', dest='data_path', type=str, default="",
        help='Required- DisknE dataset path')
    parser.add_argument('--umls_path', dest='umls_path', type=str, default="./umls-2020AA-full.zip",
        help='Required- UMLS file path')
    parser.add_argument('--pym_path', dest='pym_path', type=str, default="./pym.sqlite3",
        help='Required- Path to store pym file, needed to import UMLS in Python')
    parser.add_argument('--type', dest='type', type=str, default="similar", help='Required- For medical knowledge choose either: 1.similar 2.random, for terminological knowledge: 1.terminological_similar 2. terminological_random')
    parser.add_argument('--category', dest='cate', type=str, default="", help='Required only when evaluating medical knowledge where a category need to be specified: 1.symptoms_diseases 2.treatments_diseases 3.tests_diseases 4.procedure_diseases, for terminological skip. ')
    parser.add_argument('--seed_val', dest='seed_val', type=int, default="42", help='Required- Seed value')
    parser.add_argument('--input_type', dest='input_type', type=str, default="", help='Optional- use when evaluating the canonicalized hypothesis-only baseline: disease_only ')
    args = parser.parse_args()
    return args

def decode_label(label):
    decode_map = {"non-entailment":0,"entailment":1}
    return decode_map[label]


def read_csv_to_df(input_path):
    df= pd.read_csv(input_path)
    df.label= df.label.apply(lambda x: decode_label(x))
    return df
    

def get_x_y(df,dt,CUI):

    text=df['P'] + '[SEP]'+ df['Disease'].map(lambda x: x if type(x)!=str else x.lower())
    
    if args.input_type=='disease_only':
        text=df['Disease'].map(lambda x: x if type(x)!=str else x.lower())  
    
    text=text.map(lambda x: x if type(x)!=str else x.strip()) 
    y=df['label']
              
    return text,y



def get_inputID(bert_path,sentences):
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    input_ids = []
    attention_masks = []
    token_type_ids=[]
    for sent in sentences:
        encoded_sent = tokenizer(sent, pad_to_max_length=True, add_special_tokens = True, truncation='longest_first', max_length=256)
        input_ids.append(encoded_sent.input_ids)
        attention_masks.append(encoded_sent.attention_mask)
        token_type_ids.append(encoded_sent.token_type_ids)               
    return input_ids, attention_masks, token_type_ids





def prepare_data(bert_path,concept,matches,CUI):   
    
    
    all_data= read_csv_to_df(args.data_path)
    num_classes= len(set(all_data['label']))    
    new_test_indexes=[]    
    remove_list=[]
    ## Find matches of the target disease in the dataset,
    ## remove matches from the main df, add to the test set
    for i in range (0,len(all_data)):
        try:
            if  str(all_data['CUI'][i])==CUI  or str(all_data['original_cui'][i])==CUI or any(x in str(all_data['P'][i]).lower() for x in matches):
                remove_list.append(i)
            if args.type=='terminological_similar' or args.type=='terminological_random':
                if str(all_data['original_cui'][i])==CUI:
                    new_test_indexes.append(i)  
            else:
                if all_data['category'][i]==args.cate and str(all_data['original_cui'][i])==CUI:
                    new_test_indexes.append(i)  
        except: continue
    
    
    new_test_df= pd.DataFrame(columns=all_data.columns)    
    for i in new_test_indexes:
        new_test_df= new_test_df.append(all_data.iloc[i],ignore_index=True)
    
    ## Test data
    new_test_df= new_test_df.drop_duplicates(subset=['P', 'Disease'], keep='first',ignore_index=True)
    new_test_df = shuffle(new_test_df,random_state=84)
    xTest,y_test=get_x_y(new_test_df,'new_test_df',CUI)

    
    for i in remove_list:
        all_data=all_data.drop(i)

    all_data['Disease']=all_data['Disease'].map(lambda x: x if type(x)!=str else x.lower())
    all_data['Disease']=all_data['Disease'].map(lambda x: x if type(x)!=str else x.strip())
    if args.input_type=='disease_only':
        for i in xTest:
          all_data=all_data[~all_data.Disease.str.contains(str(i))]

    all_data=all_data.drop_duplicates(subset=['P', 'Disease'], keep='first',ignore_index=True)
    all_data = shuffle(all_data, random_state=42)
    
    ##########################################
    
    #Train and vaildation data
    X,y=get_x_y(all_data,'all_data',CUI)
    xTrain, xValid, y_train, y_valid = train_test_split(X, y, test_size=0.10, shuffle=False)
    y_train= y_train.reset_index()
    y_train= y_train['label']
    y_valid= y_valid.reset_index()
    y_valid= y_valid['label']
    
    ##########################################
    
    
    train_input_ids, train_attention_masks, train_token_type_ids= get_inputID(bert_path, xTrain.values)
    valid_input_ids,vaild_attention_masks, valid_token_type_ids= get_inputID(bert_path, xValid.values)
    test_input_ids,test_attention_masks, test_token_type_ids= get_inputID(bert_path, xTest.values)
    
    return xTest,xTrain,xValid, num_classes,train_input_ids, train_attention_masks,train_token_type_ids,y_train,valid_input_ids,vaild_attention_masks,valid_token_type_ids,y_valid,test_input_ids,test_attention_masks,test_token_type_ids, y_test
    

def get_pytorch_tensors(train_input_ids, train_attention_masks,train_token_type_ids,y_train,valid_input_ids,vaild_attention_masks,valid_token_type_ids,y_valid,test_input_ids,test_attention_masks,test_token_type_ids,y_test):
    
    train_inputs = torch.tensor(train_input_ids)
    validation_inputs = torch.tensor(valid_input_ids)  
    prediction_inputs = torch.tensor(test_input_ids)
     
    train_labels = torch.tensor(y_train)
    validation_labels = torch.tensor(y_valid)
    prediction_labels = torch.tensor(y_test)
    
    train_masks = torch.tensor(train_attention_masks)
    validation_masks = torch.tensor(vaild_attention_masks)
    prediction_masks = torch.tensor(test_attention_masks)
    
    train_token_type_ids=torch.tensor(train_token_type_ids)
    valid_token_type_ids=torch.tensor(valid_token_type_ids)
    test_token_type_ids=torch.tensor(test_token_type_ids)
       
    
    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_token_type_ids, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks,valid_token_type_ids, validation_labels)
    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    # Create the DataLoader for our testing set.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks,test_token_type_ids, prediction_labels)
    prediction_sampler = RandomSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    
    
    return train_data,train_sampler,train_dataloader,validation_data,validation_sampler,validation_dataloader,prediction_data,prediction_sampler,prediction_dataloader

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def training():

    
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs   
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)     
    os.environ['PYTHONHASHSEED'] = str(seed_val) 
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
       

    
    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    no_improvement=0
    
    # For each epoch...
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.
    
        print("\n")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        total_loss = 0
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device).long()
            b_input_mask = batch[1].to(device).long()
            b_token_type_ids = batch[2].to(device).long()
            b_labels = batch[3].to(device).long()
            model.zero_grad()        
            outputs = model(b_input_ids, 
                        token_type_ids=b_token_type_ids, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)            
        loss_values.append(avg_train_loss)
    
        print("\n")
        print("  Average training loss: {0:.8f}".format(avg_train_loss))
        torch.cuda.empty_cache()
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
    
        print("\n")
        print("Running Validation...")
        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
    
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device).long()
            b_input_mask = batch[1].to(device).long()
            b_token_type_ids = batch[2].to(device).long()
            b_labels = batch[3].to(device).long()

            with torch.no_grad():        
                outputs = model(b_input_ids, 
                                token_type_ids=b_token_type_ids, 
                                attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            
            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy
    
            # Track the number of batches
            nb_eval_steps += 1
    
        # Report the final accuracy for this validation run.
        print("Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        if epoch_i!=0:
          if loss_values[epoch_i-1] < loss_values[epoch_i] or loss_values[epoch_i-1] == loss_values[epoch_i]:
            no_improvement=no_improvement+1
        if no_improvement >1:
          break
    print("")
    print("Training complete!")
    print("")
    gc.collect()
    torch.cuda.empty_cache()
    



def testing():    
    model.eval()    
    predictions , true_labels = [], []
    
    # Predict 
    for batch in prediction_dataloader:
        b_input_ids = batch[0].to(device).long()
        b_input_mask = batch[1].to(device).long()
        b_token_type_ids = batch[2].to(device).long()
        b_labels = batch[3].to(device).long()
    
        with torch.no_grad():
          outputs = model(b_input_ids, token_type_ids=b_token_type_ids, 
                          attention_mask=b_input_mask)
    
        logits = outputs[0]
        
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        predictions.append(logits)
        true_labels.append(label_ids)
    
    print('Testing DONE.')
    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    print(classification_report(flat_true_labels, flat_predictions))
    

    gc.collect()
    torch.cuda.empty_cache()

 
if __name__ == '__main__':  
    
    if torch.cuda.is_available():        
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    args = parse_arguments()
    
    #<-------- PyMedTermino SETUP -------->
    
    default_world.set_backend(filename = args.pym_path)
    # If running for a second time, comment the follwing two lines as the pym file has already been created in the specified path
    import_umls(args.umls_path, terminologies = ["ICD10", "SNOMEDCT_US", "CUI"])
    default_world.save()
    PYM = get_ontology("http://PYM/").load()
    CUI2 = PYM["CUI"]
    SNOMEDCT_US = PYM["SNOMEDCT_US"]
    
    ##########################################
    
    # Considered diseases for Medical knowledge
    if args.type=='random' or args.type=='similar' or args.type=='random_matched' or args.type=='similar_matched':
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
  
    
    ##########################################
    
    # Get a list of all diseases as mentioned in MedNLI filtered dataset
    org_df=read_csv_to_df(args.data_path) 
    org_df =org_df[~org_df.pair_id.str.contains(',neg')]
    df2 = pd.DataFrame({'CUI':org_df.CUI.unique()})
    df2['Disease'] = [list(set(org_df['Disease'].loc[org_df['CUI'] == x['CUI']]))  for _, x in df2.iterrows()]
 
    ##########################################
    #<-- Hyper-parameters --> 
    lr= 2e-5
    epochs = 8
    seed_val=args.seed_val
    counter = 0
    batch_size=8
    berts= ["emilyalsentzer/Bio_ClinicalBERT", 
            'bert-base-cased',
            #'./biobert_v1.0_pubmed_pmc_local',
            "scibert_scivocab_cased"]  
    
    #We run BioBERT locally, downloaded from: https://github.com/dmis-lab/biobert , we use biobert_v1.0_pubmed_pmc version 
    ##########################################
    
    for BERT_path in berts:
    # Test by disease --- generate train, validation and test splits per disease
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
         
          xTest,xTrain,xValid, num_classes,train_input_ids, train_attention_masks,train_token_type_ids,y_train,valid_input_ids,vaild_attention_masks,valid_token_type_ids,y_valid,test_input_ids,test_attention_masks,test_token_type_ids, y_test=prepare_data(BERT_path,concept,matches,str(disease))
       
          #Consider diseases with at-least two positive examples
          if list(y_test).count(1) < 2: continue
          
          ############################################  
          
                 
          print('<---------------------------------------->') 
          print('BERT path: ', BERT_path)
          print('seed value: ', seed_val)
          print('Disease:', concept.label[0])
          print('CUI:', concept.name)
          print("Matches:", matches)
          print('length of new test set', len(xTest))
          print('length of training: ', len(xTrain))
          print('length of dev: ', len(xValid))
          print('<---------------------------------------->') 

          
          train_data,train_sampler,train_dataloader,validation_data,validation_sampler,validation_dataloader,prediction_data,prediction_sampler,prediction_dataloader= get_pytorch_tensors(train_input_ids, train_attention_masks,train_token_type_ids,y_train,valid_input_ids,vaild_attention_masks,valid_token_type_ids,y_valid,test_input_ids,test_attention_masks,test_token_type_ids,y_test)
   
          model = BertForSequenceClassification.from_pretrained(BERT_path, num_labels = num_classes)
          model.to(device)
          optimizer = Adam(model.parameters(), lr = lr)
          
          
          # Source code for LM fine-tuning and testing are mainly from https://mccormickml.com/2019/07/22/BERT-fine-tuning/
          print('<---------------- Train ------------------>')             
          training()
          print('<---------------- TEST ------------------>')     
          testing()
          print('<---------------------------------------->')  
          del model
          del optimizer
          gc.collect()
          torch.cuda.empty_cache()
          
    