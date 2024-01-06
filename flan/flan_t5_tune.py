import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, T5Tokenizer, T5ForConditionalGeneration
from transformers import LlamaForCausalLM, LlamaTokenizer
import pickle
# from datasets import load_metric
from torch.utils.data import Dataset, DataLoader
import warnings
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
import time

warnings.filterwarnings("ignore")

from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.meteor_score import meteor_score
from rouge_score.rouge_scorer import RougeScorer

# bleu = load_metric('bleu')

import random
import os
import numpy as np


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(1)


print("starting")

model_path = 'google/flan-t5-xl'

tokenizer = T5Tokenizer.from_pretrained(model_path, model_max_length = 512, truncation_side = 'left')
model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


with open('./final_data/text/final_train_input.p', 'rb') as f:
    train_input = pickle.load(f)

with open('./final_data/text/final_train_output.p', 'rb') as f:
    train_output = pickle.load(f)

with open('./final_data/text/final_valid_input.p', 'rb') as f:
    valid_input = pickle.load(f)

with open('./final_data/text/final_valid_output.p', 'rb') as f:
    valid_output = pickle.load(f)

print("train input 0 : ", train_input[0])

print('len(train_input[0]) : ', len(train_input[0]))
print('len(train_output[0]) : ', len(train_output[0]))
print('len(valid_input[0]) : ', len(valid_input[0]))
print('len(valid_output[0]) : ', len(valid_output[0]))


train_input_ids = tokenizer(train_input, padding = True, truncation = True, return_tensors = 'pt')['input_ids']
train_attention_mask = tokenizer(train_input, padding = True, truncation = True, return_tensors = 'pt')['attention_mask']
train_output_ids = tokenizer(train_output, padding = True, truncation = True, return_tensors = 'pt')['input_ids']


valid_input_ids = tokenizer(valid_input, padding = True, truncation = True, return_tensors = 'pt')['input_ids']
valid_attention_mask = tokenizer(valid_input, padding = True, truncation = True, return_tensors = 'pt')['attention_mask']
valid_output_ids = tokenizer(valid_output, padding = True, truncation = True, return_tensors = 'pt')['input_ids']



print("model loaded")

print("train input ids shape : ", train_input_ids.shape)
print("valid input ids shape : ", valid_input_ids.shape)




class PlantDataset(Dataset):

    def __init__(self, input_ids, attention_mask, output_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.output_ids = output_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.output_ids[idx]



train_dataloader = DataLoader(PlantDataset(train_input_ids, train_attention_mask, train_output_ids), batch_size=2, shuffle=True)
valid_dataloader = DataLoader(PlantDataset(valid_input_ids, valid_attention_mask, valid_output_ids), batch_size=2, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
print("model loaded")

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)


def train_epoch(model, dataloader):
    model.train()
    epoch_train_loss = 0.0

    for step, batch in enumerate(tqdm(dataloader)):
        
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)

        input_ids, atention_mask, output_ids = batch

        # output_ids[output_ids == tokenizer.pad_token_id] = -100

        outputs = model(input_ids = input_ids,
                        attention_mask = atention_mask,
                        labels = output_ids)

        loss = outputs['loss']

        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        
        

        

    print("Epoch train loss : ", epoch_train_loss)

    

def valid_epoch(model, dataloader):
    model.eval()
    valid_loss = 0.0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            batch = tuple(t.to(device) for t in batch)

            input_ids, atention_mask, output_ids = batch

            # output_ids[output_ids == tokenizer.pad_token_id] = -100

            outputs = model(input_ids = input_ids,
                            attention_mask = atention_mask,
                            labels = output_ids)

            loss = outputs['loss']

            valid_loss += loss.item()

    print("valid loss : ", valid_loss)    

    return valid_loss

    




def train_valid(model, train_loader, valid_loader):
    min_val_loss = 1e6

    for epoch in range(5):
        # start_time = time.time()
        print("Epoch number : ", epoch)
        train_epoch(model, train_loader)
        valid_loss = valid_epoch(model, valid_loader)

        if(valid_loss < min_val_loss):
            print("Saving model\n")
            print('min val loss : ', min_val_loss)
            print('valid loss : ', valid_loss)
            min_val_loss = valid_loss
            model.save_pretrained(f"./save_model/flan_t5/flan_t5_epoch_{epoch}")
            print('model saved\n')






train_valid(model, train_dataloader, valid_dataloader)

















