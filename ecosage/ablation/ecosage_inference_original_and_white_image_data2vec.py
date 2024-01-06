import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

# from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
import pickle
import requests
from PIL import Image
from tqdm import tqdm
from evaluate import load
bertscore = load("bertscore")

from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.meteor_score import meteor_score
from rouge_score.rouge_scorer import RougeScorer
from transformers import AutoImageProcessor, Data2VecVisionModel

image_processor = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
image_model = Data2VecVisionModel.from_pretrained("facebook/data2vec-vision-base")

import numpy as np
import os

import torch
from transformers import AutoProcessor

image_processor  = AutoProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
# image_processor 

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

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


# @registry.register_model("mini_gpt4")
class MiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4.yaml",
    }

    def __init__(
        self,
        # vit_model="eva_clip_g",
        # q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        # img_size=224,
        # drop_path_rate=0,
        # use_grad_checkpoint=False,
        # vit_precision="fp16",
        # freeze_vit=True,
        # freeze_qformer=True,
        # num_query_token=32,
        image_model = image_model,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        # print('Loading VIT')
        print('Loading image model')
        self.image_model = image_model

        for name, param in self.image_model.named_parameters():
                param.requires_grad = False

        # self.visual_encoder, self.ln_vision = self.init_vision_encoder(
        #     vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        # )
        # if freeze_vit:
        #     for name, param in self.visual_encoder.named_parameters():
        #         param.requires_grad = False
        #     self.visual_encoder = self.visual_encoder.eval()
        #     self.visual_encoder.train = disabled_train
        #     for name, param in self.ln_vision.named_parameters():
        #         param.requires_grad = False
        #     self.ln_vision = self.ln_vision.eval()
        #     self.ln_vision.train = disabled_train
        #     logging.info("freeze vision encoder")
        # print('Loading VIT Done')

        # print('Loading Q-Former')
        # self.Qformer, self.query_tokens = self.init_Qformer(
        #     num_query_token, self.visual_encoder.num_features
        # )
        # self.Qformer.cls = None
        # self.Qformer.bert.embeddings.word_embeddings = None
        # self.Qformer.bert.embeddings.position_embeddings = None
        # for layer in self.Qformer.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None
        # self.load_from_pretrained(url_or_filename=q_former_model)

        # if freeze_qformer:
        #     for name, param in self.Qformer.named_parameters():
        #         param.requires_grad = False
        #     self.Qformer = self.Qformer.eval()
        #     self.Qformer.train = disabled_train
        #     self.query_tokens.requires_grad = False
        #     logging.info("freeze Qformer")
        # print('Loading Q-Former Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        # special_tokens_dict = {'additional_special_tokens': ['[CONTEXT]', '[USER]', '[AGENT]', '[EXAMPLE]']}
        # num_added_toks = self.llama_tokenizer.add_special_tokens(special_tokens_dict)      

        if self.low_resource:
            # self.llama_model = LlamaForCausalLM.from_pretrained(
            #     llama_model,
            #     torch_dtype=torch.float16,
            #     load_in_8bit=True,
            #     device_map={'': device_8bit}
            # )
            mdl = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
            # mdl = get_peft_model(mdl, peft_config)
            # save_path = "./save_model/minigpt_decoder_weights/epoch_1"
            save_path = "./save_model/minigpt_decoder_weights/original_and_white/epoch_data2vec_1"
            # save_path = '/mnt/Data/mohit_prj22/plant_assistant_experiment/save_model/vicuna/vicuna_epoch_1'

            mdl = PeftModel.from_pretrained(mdl, save_path)
            self.llama_model = mdl
        else:
            # self.llama_model = LlamaForCausalLM.from_pretrained(
            #     llama_model,
            #     torch_dtype=torch.float16,
            # )
            mdl = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )
            # mdl = get_peft_model(mdl, peft_config)
            # save_path = "./save_model/minigpt_decoder_weights/epoch_1"
            save_path = "./save_model/minigpt_decoder_weights/original_and_white/epoch_data2vec_1"
            # save_path = '/mnt/Data/mohit_prj22/plant_assistant_experiment/save_model/vicuna/vicuna_epoch_1'

            mdl = PeftModel.from_pretrained(mdl, save_path)
            self.llama_model = mdl
        # self.llama_model.resize_token_embeddings(len(self.tokenizer))
        print("specials tokens : ", self.tokenizer.all_special_tokens)
        print('llama model \n : ', self.llama_model)
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            768, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image):
        device = image.device
        # print("device : ", device)
        # device = torch.device('cpu')
        if self.low_resource:
            print("Inside low resource")
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            # self_vis_enc = self.visual_encoder(image)
            
            # # image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            # image_embeds = self.ln_vision(self_vis_enc).to(device)
            # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            # query_output = self.Qformer.bert(
            #     query_embeds=query_tokens,
            #     encoder_hidden_states=image_embeds,
            #     encoder_attention_mask=image_atts,
            #     return_dict=True,
            # )
            self_vis_out = self.image_model(image)
            # inputs_llama = self.llama_proj(query_output.last_hidden_state)
            # print(self_vis_out.last_hidden_state.shape)
            inputs_llama = self.llama_proj(self_vis_out.last_hidden_state) 
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    # # def forward(self, samples):
    def generate_function(self, text_input, image):
        
        # text_list = ["Describe it", "What's in the image"]
        # text_list = ["Describe it"]

        # seg_tokens = [
        #     self.llama_tokenizer(
        #         seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
        #     # only add bos to the first seg
        #     for i, seg in enumerate(text_list)
        #      ]

        image_val = image_processor(image, return_tensors = 'pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), torch.float16)

        image_pixel = image_val['pixel_values']
        # image_pixel = image_val['pixel_values']
        # print("image pixel dtype : ", image_pixel.dtype)

        # img_embeds, atts_img = self.encode_img(image_val)
        # print(1/0)
        img_embeds, atts_img = self.encode_img(image_pixel)
        
        # print("image embeds shape : ", img_embeds.shape)
        # print("img embeds : ", img_embeds[0, 0, :5])
        # image_embeds = img_embeds
        img_list = [img_embeds]
        # prompt = '###Human: <Img><ImageHere></Img> Please provide a detailed description of the picture. ###Assistant:'
        # prompt = '###User: <Img><ImageHere></Img> What could cause six trunks to grow from a single tree?. ###Assistant:'
        # prompt = 'You are givena part of conversation between USER and AGENT. Your task is to complete the conversation on behalf of the AGENT. Conversation <Img><ImageHere></Img> ' + text_input
        # prompt = text_input 
        # if(white_image == True):
        #         prompt = 'Image of a plant <Img><ImageHere></Img> ' + text_input
        # else:
        prompt = text_input
        list_of_words = prompt.split()
        if('<Img><ImageHere></Img>' not in list_of_words):
                prompt = 'Image of a plant <Img><ImageHere></Img> ' + text_input       
        # # print('prompt : ', prompt)

        # index = text_input.find('</CONTEXT>')

        # print('text input : ', text_input)
        # print()

        # if(index == -1):
        #     prompt = 'Image of plant <Img><ImageHere></Img> ' + text_input

        # else:
        #     prompt = text_input[:index + 10] +  '<Img><ImageHere></Img> ' + text_input[index + 10 : ]


        
        prompt_segs = prompt.split('<ImageHere>')
        # print('prompt segs : ', prompt_segs)
        # print('len prompt segs : ', len(prompt_segs))
        # print('len img list : ', len(img_list))
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."



        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        # print('seg tokens len : ', len(seg_tokens))
        # print('seg tokens shape : ', seg_tokens.shape)

        # print("seg tokens shape : ",seg_tokens[0].shape)
        # print("seg tokens type : ", type(seg_tokens))
        # print('len seg tokens : ', len(seg_tokens))

        # for j in range(len(seg_tokens)):
        #     cnt = 0
        #     cnt += seg_tokens[j].shape[1]
        #     print("seg count : ", seg_tokens[j].shape)

        # img_list = [img_e for img_e in img_embeds]

        # seg_embs = [self.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        seg_embs = [self.llama_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        # print('seg embs [0] shape : ', seg_embs[0].shape)
        # print("len seg embs : ", len(seg_embs))
        # # print("seg  shape : ", seg_embs.shape )
        # print("seg 0 shape : ", seg_embs[0].shape )
        # print("seg 00 shape : ", seg_embs[0][0].shape )


        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        # mixed_embs = []
        # mixed_embs = img_list
        # mixed_embs.extend(seg_embs)
        # mixed_embs = torch.cat(mixed_embs, dim=1)
        # return mixed_embs

        # mixed_embs = [emb for pair in zip(seg_embs[:-1], img_embeds) for emb in pair] + [seg_embs[-1]]
        # print("len mixed embs : ",len(mixed_embs))
        # print("mixed embs  00 : ", mixed_embs[0][0].shape)
        # print("mixed embs 01 : ", mixed_embs[0][1])
        mixed_embs = torch.cat(mixed_embs, dim=1)

        

        # print('mixed embs shape : ', mixed_embs.shape)

        # print('img embeds shape : ', img_embeds.shape)
        # print('seg embs 0 shape : ', seg_embs[0].shape)
        # mixed_embs2 = torch.cat([img_embeds, seg_embs[0]], dim=1)


        # print('mixed embs 2 shape : ', mixed_embs2.shape)

        # print("all special tokens : ", self.llama_tokenizer.all_special_tokens)

            # generate_embed = img_embeds
        # outputs1 = self.llama_model.generate(
        #     inputs_embeds=mixed_embs,
        #     max_new_tokens=80,
        #         )
        outputs1 = self.llama_model.generate(
            inputs_embeds=mixed_embs,
            max_new_tokens=30,
                )        

        output_token = outputs1[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]  

        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        # output_text = output_text.split('###')[0]  # remove the stop sign '###'       

        # print("Output text 1 : ", output_text)
        # print("Output text 1 : ", output_text[0])
        # print("Output text 2 : ", output_text[1])
        # print('\n')
        # print('generated txt : \n', output_text)
        # print('\n')
        return output_text

    def generate_function2(self, text_input):

        tok = self.llama_tokenizer(text_input, return_tensors="pt").to(self.device).input_ids
        # print("\nText present\n")
        # print('tok shape : ', tok.shape)
        # tok_embed = self.llama_model.model.embed_tokens(tok)
        tok_embed = self.llama_model.base_model.model.model.embed_tokens(tok)
        # print('tok_embed shape :', tok_embed.shape )
        # print('\n\n')

        outputs1 = self.llama_model.generate(
            inputs_embeds=tok_embed,
            max_new_tokens=30,
                )

        output_token = outputs1[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]  

        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        # output_text = output_text.split('###')[0]  # remove the stop sign '###'       

        # print("Output text 1 : ", output_text)
        # print("Output text 1 : ", output_text[0])
        # print("Output text 2 : ", output_text[1])
        # print('\n')
        # print('generated txt : \n', output_text)
        # print('\n')
        return output_text

    # def forward(self, samples):
    # def generate_function(self, text_input, image):
        
    #     # text_list = ["Describe it", "What's in the image"]
    #     # text_list = ["Describe it"]

    #     # seg_tokens = [
    #     #     self.llama_tokenizer(
    #     #         seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
    #     #     # only add bos to the first seg
    #     #     for i, seg in enumerate(text_list)
    #     #      ]

    #     image_val = image_processor(image, return_tensors = 'pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), torch.float16)

    #     image_pixel = image_val['pixel_values']
    #     # image_pixel = image_val['pixel_values']
    #     # print("image pixel dtype : ", image_pixel.dtype)

    #     # img_embeds, atts_img = self.encode_img(image_val)
    #     # print(1/0)
    #     img_embeds, atts_img = self.encode_img(image_pixel)
        
    #     # print("image embeds shape : ", img_embeds.shape)
    #     # print("img embeds : ", img_embeds[0, 0, :5])
    #     # image_embeds = img_embeds
    #     img_list = [img_embeds]
    #     # prompt = '###Human: <Img><ImageHere></Img> Please provide a detailed description of the picture. ###Assistant:'
    #     # prompt = '###User: <Img><ImageHere></Img> What could cause six trunks to grow from a single tree?. ###Assistant:'
    #     # prompt = '###[CONTEXT][USER]: <Img><ImageHere></Img> What could cause six trunks to grow from a single tree?.[USER][CONTEXT] ###[AGENT]:'
    #     prompt = text_input
    #     # print('prompt : ', prompt)
        
    #     prompt_segs = prompt.split('<ImageHere>')
    #     # print('prompt segs : ', prompt_segs)
    #     assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    #     seg_tokens = [
    #         self.llama_tokenizer(
    #             seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
    #         # only add bos to the first seg
    #         for i, seg in enumerate(prompt_segs)
    #     ]
    #     # print('seg tokens len : ', len(seg_tokens))
    #     # print('seg tokens shape : ', seg_tokens.shape)

    #     # print("seg tokens shape : ",seg_tokens[0].shape)
    #     # print("seg tokens type : ", type(seg_tokens))
    #     # print('len seg tokens : ', len(seg_tokens))

    #     # for j in range(len(seg_tokens)):
    #     #     cnt = 0
    #     #     cnt += seg_tokens[j].shape[1]
    #     #     print("seg count : ", seg_tokens[j].shape)

    #     # img_list = [img_e for img_e in img_embeds]

    #     seg_embs = [self.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    #     # print("len seg embs : ", len(seg_embs))
    #     # # print("seg  shape : ", seg_embs.shape )
    #     # print("seg 0 shape : ", seg_embs[0].shape )
    #     # print("seg 00 shape : ", seg_embs[0][0].shape )


    #     mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    #     # mixed_embs = torch.cat(mixed_embs, dim=1)
    #     # return mixed_embs

    #     # mixed_embs = [emb for pair in zip(seg_embs[:-1], img_embeds) for emb in pair] + [seg_embs[-1]]
    #     # print("len mixed embs : ",len(mixed_embs))
    #     # print("mixed embs  00 : ", mixed_embs[0][0].shape)
    #     # print("mixed embs 01 : ", mixed_embs[0][1])
    #     mixed_embs = torch.cat(mixed_embs, dim=1)

        

    #     # print('mixed embs shape : ', mixed_embs.shape)

    #     # print('img embeds shape : ', img_embeds.shape)
    #     # print('seg embs 0 shape : ', seg_embs[0].shape)
    #     # mixed_embs2 = torch.cat([img_embeds, seg_embs[0]], dim=1)


    #     # print('mixed embs 2 shape : ', mixed_embs2.shape)

    #     # print("all special tokens : ", self.llama_tokenizer.all_special_tokens)

    #         # generate_embed = img_embeds
    #     # outputs1 = self.llama_model.generate(
    #     #     inputs_embeds=mixed_embs,
    #     #     max_new_tokens=80,
    #     #         )
    #     outputs1 = self.llama_model.generate(
    #         inputs_embeds=mixed_embs,
    #         max_new_tokens=30,
    #             )        

    #     output_token = outputs1[0]
    #     if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
    #             output_token = output_token[1:]
    #     if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
    #             output_token = output_token[1:]  

    #     output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
    #     # output_text = output_text.split('###')[0]  # remove the stop sign '###'       

    #     # print("Output text 1 : ", output_text)
    #     # print("Output text 1 : ", output_text[0])
    #     # print("Output text 2 : ", output_text[1])
    #     return output_text

    # def generate_function2(self, text_input):

    #     tok = self.llama_tokenizer(text_input, return_tensors="pt").to(self.device).input_ids
    #     print("\nText present\n")
    #     print('tok shape : ', tok.shape)
    #     tok_embed = self.llama_model.model.embed_tokens(tok)
    #     print('tok_embed shape :', tok_embed.shape )
    #     print('\n\n')

    #     outputs1 = self.llama_model.generate(
    #         inputs_embeds=tok_embed,
    #         max_new_tokens=80,
    #             )

    #     output_token = outputs1[0]
    #     if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
    #             output_token = output_token[1:]
    #     if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
    #             output_token = output_token[1:]  

    #     output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
    #     # output_text = output_text.split('###')[0]  # remove the stop sign '###'       

    #     # print("Output text 1 : ", output_text)
    #     # print("Output text 1 : ", output_text[0])
    #     # print("Output text 2 : ", output_text[1])
    #     return output_text


    def forward(self, text_input, image,  generate):
        print("generate : ", generate)
        if(generate):
            if(image == None):
                return self.generate_function2(text_input)
            else:
                return self.generate_function(text_input, image)

        if(image == None):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            img_embeds = torch.zeros(1,32,4096).to(device)
            atts_img = torch.zeros(1,32).to(device)
        else:    
            # image = samples["image"]
            image = image
            # print("Samples : ", samples['text_input'])
            # print("image shape : ",image.shape)
            image_val = image_processor(image, return_tensors = 'pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), torch.float16)

            
            image_pixel = image_val['pixel_values']
            # image_pixel = image_val['pixel_values']
            # print("image pixel dtype : ", image_pixel.dtype)

            # img_embeds, atts_img = self.encode_img(image_val)
            # print(1/0)
            img_embeds, atts_img = self.encode_img(image_pixel)

        # print('img embeds shape : ', img_embeds.shape)
        
        # if hasattr(samples, 'question_split'):  # VQA dataset
        #     print('VQA Batch')
        #     vqa_prompt = '###Human: <Img><ImageHere></Img> '
        #     img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        # elif self.prompt_list:
        #     prompt = random.choice(self.prompt_list)
        #     img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

        self.llama_tokenizer.padding_side = "right"

        # text = [t + self.end_sym for t in samples["text_input"]]
        # text = [t + self.end_sym for t in "Dinasours here"]
        # text = "Dinasours here" + self.end_sym
        text = text + self.end_sym

        # print("text : \n", text)
        # print()

        # print("Line 209")

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        # ).to(image.device)
        ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # print("to regress tokens : ", to_regress_tokens)

        # print("to regress token decode : ", self.llama_tokenizer.batch_decode(to_regress_tokens['input_ids']))

        # print("Line 218")

        # print('to regress tokens shape : ', to_regress_tokens.input_ids.shape)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )



        # print("Line 226")

        # print("att img shape : ", atts_img.shape)

        # empty_targets = (
        #     torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
        #                dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
        # )
        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).fill_(-100)  # plus one for bos
        )
        # print("empty target shape : ", empty_targets.shape)
        # print("targets shape : ", targets.shape)
        targets = torch.cat([empty_targets, targets], dim=1)


        # print("Line 234")

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        # bos_embeds = self.llama_model.model.embed_tokens(bos)
        bos_embeds = self.llama_model.base_model.model.model.embed_tokens(bos)
        # print('bos embeds shape : ', bos_embeds.shape)
        # print(1/0)
        atts_bos = atts_img[:, :1]

        # print("Line 243")

        # to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        to_regress_embeds = self.llama_model.base_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        # print("Line 249")

        # print("input embeds shape : ", inputs_embeds.shape)
        # print("attention_mask shape : ", attention_mask.shape)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        



        # print("Line 262")

        loss = outputs.loss

        # return {"loss": loss}
        return loss

    

vit_model = "eva_clip_g"
q_former_model = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth'
img_size = 224
num_query_token = 32
llama_model = '/mnt/Data/mohit_prj22/MiniGPT-4/weights/vicuna_llama_7b/'

drop_path_rate = 0
use_grad_checkpoint = False
vit_precision = 'fp16'
freeze_vit = True
freeze_qformer = True
low_resource = False
device_8bit = 0

prompt_path = 'prompts/alignment.txt'
prompt_template = '###Human: {} ###Assistant:'
max_txt_len = 160
end_sym = '###'

model = MiniGPT4(
            # vit_model=vit_model,
            # q_former_model=q_former_model,
            # img_size=img_size,
            # drop_path_rate=drop_path_rate,
            # use_grad_checkpoint=use_grad_checkpoint,
            # vit_precision=vit_precision,
            # freeze_vit=freeze_vit,
            # freeze_qformer=freeze_qformer,
            # num_query_token=num_query_token,
            image_model= image_model,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

ckpt_path = '/mnt/Data/mohit_prj22/MiniGPT-4/weights/checkpoints_aligned_vicuna_7b/pretrained_minigpt4_7b.pth'
if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
print("\n---------------\n")
print("model loaded")
print(model)

# model.llama_proj.load_state_dict(torch.load('./save_model/llama_proj_epoch_1'))
model.llama_proj.load_state_dict(torch.load('./save_model/original_and_white/llama_proj_data2vec_epoch_1'))

# url = 'https://media.newyorker.com/cartoons/63dc6847be24a6a76d90eb99/master/w_1160,c_limit/230213_a26611_838.jpg'
# image = Image.open('image_exp/tree.jpg').convert('RGB')  
# image1 = Image.open('image_exp/r2.jpg').convert('RGB')  
# image2 = Image.open('image_exp/cat3.jpg').convert('RGB')  
# print("read image")

# image_list = [image1, image2]

# image_list = image

model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


# loss = model(image, generate = True)
# model(image_list, generate = True)
# # print("Loss : ", loss['loss'])



# with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/image/final_valid_input.p', 'rb') as f:
#     valid_input = pickle.load(f)

# with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/image/final_valid_output.p', 'rb') as f:
#     valid_output = pickle.load(f)

with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/image_test_input_mod.p', 'rb') as f:
    test_input = pickle.load(f)

with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/image/final_test_output.p', 'rb') as f:
    test_output = pickle.load(f)


# with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/valid_image_loc_list.p', 'rb') as f:
#     valid_image_loc_list = pickle.load(f)

with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/test_image_loc_list.p', 'rb') as f:
    test_image_loc_list = pickle.load(f)




# with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/valid_present_list.p', 'rb') as f:
#     valid_present_list = pickle.load(f)

with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/test_present_list.p', 'rb') as f:
    test_present_list = pickle.load(f)



# print('test input 0 :', test_input[0])
# print('\n')
# print('test output 0 : ', test_output[0])   
# print('test output 00 : ', test_output[0][0]) 


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






# def return_generate(data_input, image_loc_list, present_list):

#     generated_list_with_input = []
#     generated_list = []
#     for i in range(len(data_input)):
#         print(i)
#         data = data_input[i][0]
#         image_loc1 = image_loc_list[i] + '.jpg'
#         image_loc2 = '/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/mapped_folder/' + image_loc1
#         present = present_list[i]
#         # image_list = []
#         if(present == True):
#             image = Image.open(image_loc2).convert('RGB')  
#             # image_list.append(image)
#         else:
#             image = None


#         outputs = model(data, image, generate = True)
#         generated_list.append(outputs)
#         # print("output sequences shape : ", outputs.sequences.shape)
#         # token = outputs.sequences[0, input_length:]
#         # token = outputs.sequences[0, :]
#         # token2 = outputs.sequences[0, input_length:]
#         # response = tokenizer.decode(token, skip_special_tokens = True)
#         # response2 = tokenizer.decode(token2, skip_special_tokens = True)
#         # generated_list_with_input.append(response)
#         # generated_list.append(response2)
#         print()

#     # return generated_list_with_input, generated_list   
#     return generated_list     
image_only_input_idx = []
image_only_output = []
def return_generate(data_input, image_loc_list, present_list):

    generated_list_with_input = []
    generated_list = []
    for i in range(len(data_input)):
        print(i)
        data = data_input[i]
        image_loc1 = image_loc_list[i] + '.jpg'
        image_loc2 = '/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/mapped_folder/' + image_loc1
        present = present_list[i]
        white_image=False
        # image_list = []
        if(present == True):
            image = Image.open(image_loc2).convert('RGB')  
            # image_list.append(image)
            # outputs = model(data, image, generate = True)
            # # generated_list.append(outputs)
            # image_only_input_idx.append(i)
            # image_only_output.append(outputs)
            # image = Image.open('white_image.jpeg').convert('RGB')
        else:
            # image = None
            image = Image.open('white_image.jpeg').convert('RGB')    
            # white_image=True          


        outputs = model(data, image, generate = True)
        generated_list.append(outputs)
        # print("output sequences shape : ", outputs.sequences.shape)
        # token = outputs.sequences[0, input_length:]
        # token = outputs.sequences[0, :]
        # token2 = outputs.sequences[0, input_length:]
        # response = tokenizer.decode(token, skip_special_tokens = True)
        # response2 = tokenizer.decode(token2, skip_special_tokens = True)
        # generated_list_with_input.append(response)
        # generated_list.append(response2)
        print()

    # return generated_list_with_input, generated_list   
    return generated_list
         


def print_metric(metric_dict):
    print("rouge 1 : ", metric_dict['rouge_1'])
    print("rouge 2 : ", metric_dict['rouge_2'])
    print("rouge L : ", metric_dict['rouge_L'])

    print("\n\n-----------------------\n\n")
    print("Bleu 1 : ", metric_dict['bleu_1'])
    print("Bleu 2 : ", metric_dict['bleu_2'])
    print("Bleu 3 : ", metric_dict['bleu_3'])
    print("Bleu 4 : ", metric_dict['bleu_4'])

    print("\n\n-----------------------\n\n")
    print("Bert precision : ", metric_dict['bert_precision'])
    print('Bert recall : ', metric_dict['bert_recall'])
    print('Bert F1 : ', metric_dict['bert_f1'])
    print("\n\n-----------------------\n\n")



# test_generated_list_with_input, test_generated_list = return_generate(test_input, image_loc_list, present_list)
# valid_generated_list = return_generate(valid_input, valid_image_loc_list, valid_present_list)
# test_generated_list = return_generate(test_input, test_image_loc_list, test_present_list)

# with open('./save_model/test_generated_list_80_tok.p', 'wb') as f:
#     pickle.dump(test_generated_list, f)

# print("\n-------------------\n")
# print("valid metrics")
# valid_metric_dict = get_scores(reference_list=valid_output, hypothesis_list=valid_generated_list)
# print_metric(valid_metric_dict)
# print("\n-------------------\n")

# print("\n-------------------\n")
# print("test metrics")

test_text_input = []
for j in test_input:
    test_text_input.append(j[0])


test_text_output = []
for j in test_output:
    test_text_output.append(j[0])

print('test text input[0] : ', test_text_input[0])    
print('test text output[0] : ', test_text_output[0])


test_input2 = []
test_output2 = []
test_idx =[]

for i,j in enumerate(test_text_output):
    
    temp_input = test_text_input[i]
    temp_output = test_text_output[i]

    if(len(temp_output)>0):
        test_idx.append(i)
        test_input2.append(temp_input)
        test_output2.append(temp_output)

print('\n---------------------------------\n')
print('new test input len : ', len(test_input2))
print('new_test_output len : ', len(test_output2))
print('\n---------------------------------\n')

      

test_generated_list = return_generate(test_text_input, test_image_loc_list, test_present_list)
# test_generated_list = return_generate(test_input2, test_image_loc_list, test_present_list)

with open('./save_model/test_generated_list_new_original_and_white_image_30_tok.p', 'wb') as f:
    pickle.dump(test_generated_list, f)

# with open('./save_model/proper_test_input.p', 'wb') as f:
#     pickle.dump(test_input2, f)
# with open('./save_model/proper_test_output.p', 'wb') as f:
#     pickle.dump(test_output2, f)
# with open('./save_model/proper_test_index.p', 'wb') as f:
#     pickle.dump(test_idx, f)  

print("\n-------------------\n")
print("test metrics")
test_metric_dict = get_scores(reference_list=test_text_output, hypothesis_list=test_generated_list)
# test_metric_dict = get_scores(reference_list=test_output2, hypothesis_list=test_generated_list)
image_only_gold_response = []

# for i in range(len(test_output2)):
#     if(test_present_list[i] == True):
#         image_only_gold_response.append(test_output2[i])

# # test_metric_dict = get_scores(reference_list=test_output2, hypothesis_list=image_only_output)
# test_metric_dict = get_scores(reference_list=image_only_gold_response, hypothesis_list=image_only_output)
print_metric(test_metric_dict)
print("\n-------------------\n")

# with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/zero_shot_data_image/zero_shot_prompt_input_image.p', 'rb') as f:
#     test_input2 = pickle.load(f)

# with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/zero_shot_data_image/zero_shot_prompt_output_image.p', 'rb') as f:
#     test_output2 = pickle.load(f)

# with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/zero_shot_data_image/test_image_loc_list.p', 'rb') as f:
#     image_loc_list2 = pickle.load(f)

# with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/zero_shot_data_image/test_present_list_image.p', 'rb') as f:
#     present_list2 = pickle.load(f)

# test_generated_list = return_generate(test_input2, image_loc_list2, present_list2)



# print("\n-------------------\n")
# print("test metrics")
# test_metric_dict = get_scores(reference_list=test_output2, hypothesis_list=test_generated_list)
# print_metric(test_metric_dict)
# print("\n-------------------\n")





print("input_prompt idx 1 : ", test_text_input[1])
print("\n-----------------------\n")

print("gold response idx 1 : ", test_text_output[1])
print("\n-----------------------\n")

print("generated output : ", test_generated_list[1])
print("\n-----------------------\n")

# print("input_prompt idx 1 : ", test_input2[7])
# print("\n-----------------------\n")

# print("gold response idx 1 : ", test_output2[7])
# print("\n-----------------------\n")

# print("generated output : ", test_generated_list[7])
# print("\n-----------------------\n")

