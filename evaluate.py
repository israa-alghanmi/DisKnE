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
from datasets import Dataset



def parse_arguments():
    """Read arguments from a command line."""
    parser = argparse.ArgumentParser(description='Experiment setup - Arguments get parsed via --commands')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default="",
        help='Required- Directory path for DisknE diseases splits')
    parser.add_argument('--BERT_path', dest='BERT_path', type=str, default="",
        help='Required- Path to BERT')
    parser.add_argument('--seed_val', dest='seed_val', type=int, default="42",
     help='Required- Seed value')
    args = parser.parse_args()
    return args


def preprocess_df(df):
    df = df.replace(np.nan, '', regex=True)
    my_dict = {"P": list(df['P']), 'Disease': list(df['Disease']), 'labels':list(df['label']) }
    dataset_=Dataset.from_dict(my_dict)
    dataset_=dataset_.map(tokenize_function, batched=True)
    return dataset_ 
    


def tokenize_function(example):
    return tokenizer(example["P"], example["Disease"], truncation=True,pad_to_max_length=True, max_length=256)
    


def training():

    
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
       
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)     
    # This training code is based on the `run_glue.py` script here: https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128, https://mccormickml.com/2019/07/22/BERT-fine-tuning/
       
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
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            total_loss += outputs.loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)            
        
    
        print("\n")
        print("  Average training loss: {0:.8f}".format(avg_train_loss))

        # ========================================
        #               Validation
        # ========================================
    
        print("\n")
        model.eval()

        eval_loss = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        total_eval_loss = 0
    
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
        
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = batch["labels"].to('cpu').numpy()
            total_eval_loss += outputs.loss
            nb_eval_steps += 1
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        loss_values.append(avg_val_loss)
        
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
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
    
        logits = outputs.logits
        
        logits = logits.detach().cpu().numpy()
        label_ids = batch["labels"].to('cpu').numpy()
        
        predictions.append(logits)
        true_labels.append(label_ids)
    
    print('Testing DONE.')
    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    #print(classification_report(flat_true_labels, flat_predictions))
    
    #F1 score for the positive class
    postives_indices = [i for i, elem in enumerate(flat_true_labels) if 1==elem]
    true_labels=[1 for d in postives_indices]
    predictions=[flat_predictions[d] for d in postives_indices]
    f1= f1_score(true_labels, predictions, average='macro')
    
    
    print('F1 score',f1)
    gc.collect()
    torch.cuda.empty_cache()
    return f1

 
if __name__ == '__main__':  
    
    if torch.cuda.is_available():        
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    args = parse_arguments()
    
    seed_val=args.seed_val
    os.environ['PYTHONHASHSEED'] = str(seed_val) 
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   
    
    
    #-- Hyper-parameters 
    lr= 2e-5
    epochs = 8
    batch_size=8
    BERT_path= args.BERT_path
    diseases_list_f=args.data_dir+'/diseases_list.txt'
    
    #--read DisKne 
    with open(diseases_list_f) as file:
        diseases_list = file.readlines()
        diseases_list = [line.rstrip() for line in diseases_list]
    
    diseases_list=list(set(diseases_list))
    all_f1=[]
    for disease in diseases_list:
        print(disease)
        training_df= pd.read_csv(args.data_dir+'/DisKnE_train{}.csv'.format(disease))
        dev_df= pd.read_csv(args.data_dir+'/DisKnE_dev{}.csv'.format(disease))
        test_df= pd.read_csv(args.data_dir+'/DisKnE_test{}.csv'.format(disease))
        tokenizer = AutoTokenizer.from_pretrained(BERT_path)        
        #--prepare datasets
        train_tokenized_datasets=preprocess_df(training_df)
        dev_tokenized_datasets=preprocess_df(dev_df)
        test_tokenized_datasets=preprocess_df(test_df)
        train_tokenized_datasets3 = train_tokenized_datasets.remove_columns(["P", "Disease"])
        train_tokenized_datasets3.set_format("torch")
        dev_tokenized_datasets3 = dev_tokenized_datasets.remove_columns(["P", "Disease"])
        dev_tokenized_datasets3.set_format("torch")
        test_tokenized_datasets3 = test_tokenized_datasets.remove_columns(["P", "Disease"])
        test_tokenized_datasets3.set_format("torch")
        #--prepare dataloaders
        train_dataloader=DataLoader(train_tokenized_datasets3, shuffle=True, batch_size=batch_size)
        validation_dataloader=DataLoader(dev_tokenized_datasets3, shuffle=True, batch_size=batch_size) 
        prediction_dataloader=DataLoader(test_tokenized_datasets3, shuffle=True, batch_size=batch_size)

        model = BertForSequenceClassification.from_pretrained(BERT_path, num_labels=2)
        model.to(device)
        optimizer = Adam(model.parameters(), lr = lr)
        
        print('<-- Training -->')             
        training()
        print('<--TEST-->')     
        f1=testing()
        all_f1.append(f1)
        print('<---------------------------------------->')  
        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
    
    print("Average F1 :{}".format(sum(all_f1) / len(all_f1)))   
    
        
    
