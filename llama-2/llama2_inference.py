import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
import pickle
# from datasets import load_metric
from torch.utils.data import Dataset, DataLoader
import warnings
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from evaluate import load
bertscore = load("bertscore")

warnings.filterwarnings("ignore")

from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.meteor_score import meteor_score
from rouge_score.rouge_scorer import RougeScorer

# bleu = load_metric('bleu')

import random
import os
import numpy as np
# model = AutoModelForCausalLM.from_pretrained("facebook/opt-13b", torch_dtype=torch.float16).cuda()
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b", use_fast=False)

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

model_name = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_name, model_max_length = 2048, truncation_side = 'left')



tokenizer.pad_token = tokenizer.eos_token



model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)




with open('./final_data/text/final_train_input.p', 'rb') as f:
    train_input = pickle.load(f)

with open('./final_data/text/final_train_output.p', 'rb') as f:
    train_output = pickle.load(f)

with open('./final_data/text/final_valid_input.p', 'rb') as f:
    valid_input = pickle.load(f)

with open('./final_data/text/final_valid_output.p', 'rb') as f:
    valid_output = pickle.load(f)


with open('./final_data/text/final_test_input.p', 'rb') as f:
    test_input = pickle.load(f)

with open('./final_data/text/final_test_output.p', 'rb') as f:
    test_output = pickle.load(f)


print("train input 0 : ", train_input[0])

print('len(train_input[0]) : ', len(train_input[0]))
print('len(train_output[0]) : ', len(train_output[0]))
print('len(valid_input[0]) : ', len(valid_input[0]))
print('len(valid_output[0]) : ', len(valid_output[0]))
print('len(test_input[0] : ', len(test_input[0]))
print('len(test_output[0]) : ', len(test_output[0]))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def get_scores(reference_list: list,
               hypothesis_list: list):
    count=0
    met=0
    bleu_1=0
    bleu_2=0
    bleu_3=0
    bleu_4=0
    rouge1=0
    rouge2=0
    rougel = 0
    weights_1 = (1./1.,)
    weights_2 = (1./2. , 1./2.)
    weights_3 = (1./3., 1./3., 1./3.)
    weights_4 = (1./4., 1./4., 1./4., 1./4.)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])

   

    bert_precision_score = 0.0
    bert_recall_score = 0.0
    bert_f1_score = 0.0

    for reference, hypothesis in list(zip(reference_list, hypothesis_list)):
        scores = rouge_scorer.score(reference, hypothesis)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougel += scores['rougeL'].fmeasure

        bert_results = bertscore.compute(predictions=[hypothesis], references=[reference], model_type = 'microsoft/deberta-xlarge-mnli')
        bert_precision = bert_results['precision']
        bert_recall = bert_results['recall']
        bert_f1 = bert_results['f1']

        bert_precision_score += sum(bert_precision)
        bert_recall_score += sum(bert_recall)
        bert_f1_score += sum(bert_f1)

        # met += meteor_score([reference], hypothesis)

        reference = reference.split()
        hypothesis = hypothesis.split()
        bleu_1 += sentence_bleu([reference], hypothesis, weights_1) 
        bleu_2 += sentence_bleu([reference], hypothesis, weights_2)
        bleu_3 += sentence_bleu([reference], hypothesis, weights_3)
        bleu_4 += sentence_bleu([reference], hypothesis, weights_4)
        count += 1

    return {
        "rouge_1": rouge1*100/count,
        "rouge_2": rouge2*100/count,
        "rouge_L": rougel*100/count,
        "bleu_1": bleu_1*100/count,
        "bleu_2": bleu_2*100/count,
        "bleu_3": bleu_3*100/count,
        "bleu_4": bleu_4*100/count,
        'bert_precision' : bert_precision_score*100/count,
        'bert_recall' : bert_recall_score*100/count,
        'bert_f1' : bert_f1_score*100/count

        # "meteor": met*100/count,
    }  



save_path = "./save_model/llama2/llamma2_epoch_1"

model = PeftModel.from_pretrained(model, save_path)
model = model.to(device)


def return_generate(data_input):

    generated_list_with_input = []
    generated_list = []
    for i in range(len(data_input)):
        print(i)
        data = data_input[i]
        inputs = tokenizer(data, return_tensors='pt').to(device)
        input_length = inputs.input_ids.shape[1]
        print('inputs.input_ids.shape : ', inputs.input_ids.shape)
        # outputs = model.generate(
        #     **inputs, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True
        # )
        # outputs = model.generate(
        #     input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'], max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True
        # )
        outputs = model.generate(
            input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'], max_new_tokens=30, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True
        )
        print("output sequences shape : ", outputs.sequences.shape)
        # token = outputs.sequences[0, input_length:]
        token = outputs.sequences[0, :]
        token2 = outputs.sequences[0, input_length:]
        response = tokenizer.decode(token, skip_special_tokens = True)
        response2 = tokenizer.decode(token2, skip_special_tokens = True)
        generated_list_with_input.append(response)
        generated_list.append(response2)
        print()

    return generated_list_with_input, generated_list    


def print_metric(metric_dict):
    print("rouge 1 : ", metric_dict['rouge_1'])
    print("rouge 2 : ", metric_dict['rouge_2'])
    print("rouge L : ", metric_dict['rouge_L'])

    print("\n\n-----------------------\n\n")
    print("Bleu 1 : ", metric_dict['bleu_1'])
    print("Bleu 2 : ", metric_dict['bleu_2'])
    print("Bleu 3 : ", metric_dict['bleu_3'])
    print("Bleu 4 : ", metric_dict['bleu_4'])
    print('\n\n')

       
    print("Bert precision : ", metric_dict['bert_precision'])
    print('Bert recall : ', metric_dict['bert_recall'])
    print('Bert F1 : ', metric_dict['bert_f1'])
    print('\n\n')



# train_generated_list_with_input, train_generated_list = return_generate(train_input)
# valid_generated_list_with_input, valid_generated_list = return_generate(valid_input)
test_generated_list_with_input, test_generated_list = return_generate(test_input)

# print("\nTrain metrics")
# train_metric_dict = get_scores(reference_list=train_output, hypothesis_list=train_generated_list)
# print_metric(train_metric_dict)

# print("\n-------------------\n")
# print("Valid metrics")
# valid_metric_dict = get_scores(reference_list=valid_output, hypothesis_list=valid_generated_list)
# print_metric(valid_metric_dict)

print("\n-------------------\n")
print("test metrics")
test_metric_dict = get_scores(reference_list=test_output, hypothesis_list=test_generated_list)
print_metric(test_metric_dict)



# with open('./generated_response/fine_tune/vicuna/train_generated_list.p', 'wb') as f:
#     pickle.dump(train_generated_list, f)

# with open('./generated_response/fine_tune/vicuna/train_generated_list_with_input.p', 'wb') as f:
#     pickle.dump(train_generated_list_with_input, f)


# with open('./generated_response/fine_tune/vicuna/valid_generated_list.p', 'wb') as f:
#     pickle.dump(valid_generated_list, f)

# with open('./generated_response/fine_tune/vicuna/valid_generated_list_with_input.p', 'wb') as f:
#     pickle.dump(valid_generated_list_with_input, f)


# with open('./generated_response/fine_tune/vicuna/test_generated_list.p', 'wb') as f:
#     pickle.dump(test_generated_list, f)

# with open('./generated_response/fine_tune/vicuna/test_generated_list_with_input.p', 'wb') as f:
#     pickle.dump(test_generated_list_with_input, f)

with open('./generated_response/fine_tune/llama2/test_generated_list_30_tok.p', 'wb') as f:
    pickle.dump(test_generated_list, f)

with open('./generated_response/fine_tune/llama2/test_generated_list_with_input_30_tok.p', 'wb') as f:
    pickle.dump(test_generated_list_with_input, f)



print('\n')
print("input_prompt idx 1 : ", test_input[1])
print('\n')

print("gold response idx 1 : ", test_output[1])
print('\n')

print("generated output : ", test_generated_list[1])
print('\n')

print("generated output : ", test_generated_list_with_input[1])
print('\n')







