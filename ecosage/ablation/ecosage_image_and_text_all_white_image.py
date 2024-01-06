import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

# from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import pickle
import requests
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import torch
from transformers import AutoProcessor

image_processor  = AutoProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
# image_processor 

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

bos_token_embedding = None

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
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
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

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        print('Loading Q-Former Done')
        self.bos_token_embedding = None
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False, model_max_length = 512, truncation = 'left')
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
            mdl = get_peft_model(mdl, peft_config)
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
            mdl = get_peft_model(mdl, peft_config)
            self.llama_model = mdl
        # self.llama_model.resize_token_embeddings(len(self.tokenizer))
        print("specials tokens : ", self.tokenizer.all_special_tokens)
        print('llama model \n : ', self.llama_model)
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
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
            self_vis_enc = self.visual_encoder(image)
            # image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_embeds = self.ln_vision(self_vis_enc).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before,             
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=340,
                add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=80,
                add_special_tokens=False).to(img_embeds.device)
            # p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_before_embeds = self.llama_model.base_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            
            # p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.base_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

            # print('p_before token shape : ', p_before_tokens.shape)
            # print('p_before embeds shape : ', p_before_embeds.shape)

            # print('p_after token shape : ', p_after_tokens.shape)
            # print('p after embeds shape : ', p_after_embeds.shape)

            # print('img_embeds shape : ', img_embeds.shape)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            # print('wrapped img_embeds shape : ', wrapped_img_embeds.shape)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    # def forward(self, samples):
    def generate_function(self, image):
        
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
        
        print("image embeds shape : ", img_embeds.shape)
        print("img embeds : ", img_embeds[0, 0, :5])
        # image_embeds = img_embeds
        img_list = [img_embeds]
        # prompt = '###Human: <Img><ImageHere></Img> Please provide a detailed description of the picture. ###Assistant:'
        # prompt = '###User: <Img><ImageHere></Img> What could cause six trunks to grow from a single tree?. ###Assistant:'
        prompt = '###[CONTEXT][USER]: <Img><ImageHere></Img> What could cause six trunks to grow from a single tree?.[USER][CONTEXT] ###[AGENT]:'
        print('prompt : ', prompt)
        
        prompt_segs = prompt.split('<ImageHere>')
        print('prompt segs : ', prompt_segs)
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        print('seg tokens len : ', len(seg_tokens))
        # print('seg tokens shape : ', seg_tokens.shape)

        # print("seg tokens shape : ",seg_tokens[0].shape)
        print("seg tokens type : ", type(seg_tokens))
        # print('len seg tokens : ', len(seg_tokens))

        for j in range(len(seg_tokens)):
            cnt = 0
            cnt += seg_tokens[j].shape[1]
            print("seg count : ", seg_tokens[j].shape)

        # img_list = [img_e for img_e in img_embeds]

        # seg_embs = [self.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        seg_embs = [self.llama_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        print("len seg embs : ", len(seg_embs))
        # print("seg  shape : ", seg_embs.shape )
        print("seg 0 shape : ", seg_embs[0].shape )
        print("seg 00 shape : ", seg_embs[0][0].shape )


        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        # mixed_embs = torch.cat(mixed_embs, dim=1)
        # return mixed_embs

        # mixed_embs = [emb for pair in zip(seg_embs[:-1], img_embeds) for emb in pair] + [seg_embs[-1]]
        print("len mixed embs : ",len(mixed_embs))
        print("mixed embs  00 : ", mixed_embs[0][0].shape)
        # print("mixed embs 01 : ", mixed_embs[0][1])
        mixed_embs = torch.cat(mixed_embs, dim=1)

        

        print('mixed embs shape : ', mixed_embs.shape)

        print('img embeds shape : ', img_embeds.shape)
        print('seg embs 0 shape : ', seg_embs[0].shape)
        # mixed_embs2 = torch.cat([img_embeds, seg_embs[0]], dim=1)


        # print('mixed embs 2 shape : ', mixed_embs2.shape)

        print("all special tokens : ", self.llama_tokenizer.all_special_tokens)

            # generate_embed = img_embeds
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

        print("Output text 1 : ", output_text)
        # print("Output text 1 : ", output_text[0])
        # print("Output text 2 : ", output_text[1])
        return output_text

    def forward(self, text, image, output_text, generate):
        # print("generate : ", generate)
        if(generate):
            return self.generate_function(image)
        # image_presence = True
        # if(image == None):
        if(image == None):    
            print("\n-------------------Image is None-------------------\n")
            
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # img_embeds = torch.zeros(1,32,4096).to(device)
            # atts_img = torch.zeros(1,32).to(device)
            # image_presence = False

            self.llama_tokenizer.padding_side = "right"

            input_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=400,
                add_special_tokens=False
            # ).to(image.device)
            ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))    

            to_regress_tokens = self.llama_tokenizer(
            output_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=80,
            add_special_tokens=False
            # ).to(image.device)
            ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))      


            input_text_embeds = self.llama_model.base_model.model.model.embed_tokens(input_tokens.input_ids)

            to_regress_embeds = self.llama_model.base_model.model.model.embed_tokens(to_regress_tokens.input_ids)


            empty_targets = (
            torch.ones([input_tokens.attention_mask.shape[0], input_tokens.attention_mask.shape[1]],
                       dtype=torch.long).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).fill_(-100)  # plus one for bos
            )

            targets_text = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )

            inputs_embeds = torch.cat([input_text_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([input_tokens.attention_mask, to_regress_tokens.attention_mask], dim=1)
            targets = torch.cat([empty_targets, targets_text], dim=1)
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

            # if(white_image == True):
            #     prompt = 'Image of a plant <Img><ImageHere></Img> ' + text
            # else:
            prompt = text
            list_of_words = prompt.split()
            if('<Img><ImageHere></Img>' not in list_of_words):
                prompt = 'Image of a plant <Img><ImageHere></Img> ' + text

            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)
            # print('img embeds shape : ', img_embeds.shape)
            self.llama_tokenizer.padding_side = "right"

            to_regress_tokens = self.llama_tokenizer(
            output_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=80,
            add_special_tokens=False
            # ).to(image.device)
            ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).fill_(-100)  # plus one for bos
            )

            targets_text = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )

            bos = torch.ones((1,1), dtype = torch.long).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            # print('bos token id : ', self.llama_tokenizer.bos_token_id)                 
            # bos_embeds = self.llama_model.model.embed_tokens(bos)

            if(self.bos_token_embedding == None):
                self.bos_token_embedding = self.llama_model.base_model.model.model.embed_tokens(bos)
                # bos_embeds = self.llama_model.base_model.model.model.embed_tokens(bos)
            bos_embeds = self.bos_token_embedding 

            atts_bos = atts_img[:, :1]

            to_regress_embeds = self.llama_model.base_model.model.model.embed_tokens(to_regress_tokens.input_ids)

            inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)
            targets = torch.cat([empty_targets, targets_text], dim=1)




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
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
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
# print(model)

# url = 'https://media.newyorker.com/cartoons/63dc6847be24a6a76d90eb99/master/w_1160,c_limit/230213_a26611_838.jpg'
# image = Image.open('image_exp/tree.jpg').convert('RGB')  
# image1 = Image.open('image_exp/r2.jpg').convert('RGB')  
# image2 = Image.open('image_exp/cat3.jpg').convert('RGB')  
# print("read image")

# image_list = [image1, image2]

# image_list = image

model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

cnt = 0
for name, param in model.named_parameters():
    print("Count : ", cnt, " name : ", name)
    if('lora' in name):
        param.requires_grad = True
    # elif(cnt>=1187):
    #     param.requires_grad = False
    cnt+=1

for name, param in model.named_parameters():
    if(param.requires_grad == True):
        print(" name : ", name)

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('\n' + 'num_trainable_params = ' + str(num_trainable_params) + '\n')
# print(1/0)
# loss = model(image, generate = True)
# model(image_list, generate = True)
# # print("Loss : ", loss['loss'])

with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/image_train_input_mod.p', 'rb') as f:
    train_input = pickle.load(f)

with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/image/final_train_output.p', 'rb') as f:
    train_output = pickle.load(f)    

with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/image_valid_input_mod.p', 'rb') as f:
    valid_input = pickle.load(f)

with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/image/final_valid_output.p', 'rb') as f:
    valid_output = pickle.load(f) 


with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/train_image_loc_list.p', 'rb') as f:
    train_image_loc_list = pickle.load(f)

with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/valid_image_loc_list.p', 'rb') as f:
    valid_image_loc_list = pickle.load(f)    

with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/train_present_list.p', 'rb') as f:
    train_present_list = pickle.load(f)

with open('/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/valid_present_list.p', 'rb') as f:
    valid_present_list = pickle.load(f)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

def train_epoch(train_input, train_output):
    model.train()
    epoch_train_loss = 0.0
    for j in tqdm(range((len(train_input)))):
        # if(j == 680):
        #     continue        
        # print('train input index : ', j)
        optimizer.zero_grad()
        text = train_input[j][0]
        image_loc1 = train_image_loc_list[j] + '.jpg'
        image_loc2 = '/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/mapped_folder/' + image_loc1
        present = train_present_list[j]
        # image_list = []
        white_image = False
        if(present == True):
            # image = Image.open(image_loc2).convert('RGB')  
            image = Image.open('white_image.jpeg').convert('RGB')
            # image_list.append(image)
            train_output_text = train_output[j][0]
            if(len(train_output_text) == 0):
               continue
            
            # train_loss = model(text, image, train_output_text, white_image, generate = False)
            train_loss = model(text, image, train_output_text, generate = False)


        else:
            # image = None
            image = Image.open('white_image.jpeg').convert('RGB')  
            # white_image = True
            if(len(train_output_text) == 0):
               continue            
            # train_loss = model(text, image, train_output_text, white_image, generate = False)
            train_loss = model(text, image, train_output_text, generate = False)

        train_loss.backward()
        epoch_train_loss += train_loss.item()
        optimizer.step()
        # train_output_text = train_output[j][0]
 
        # if(len(train_output_text) == 0):
        #     continue
        # train_loss = model(text, image, train_output_text, white_image, generate = False)
        # train_loss.backward()
        # epoch_train_loss += train_loss.item()
        # optimizer.step()
    print('epoch train loss : ', epoch_train_loss)    


def valid_epoch(valid_input, valid_output):
    epoch_valid_loss = 0.0
    with torch.no_grad():
        model.eval()
        for j in tqdm(range((len(valid_input)))):

            text = valid_input[j][0]
            image_loc1 = valid_image_loc_list[j] + '.jpg'
            image_loc2 = '/mnt/Data/mohit_prj22/plant_assistant_experiment/final_data/mapped_folder/' + image_loc1
            present = valid_present_list[j]
            white_image = False
            # image_list = []
            if(present == True):
                # image = Image.open(image_loc2).convert('RGB')
                image = Image.open('white_image.jpeg').convert('RGB')  
                # image_list.append(image)
                valid_output_text = valid_output[j][0]

                if(len(valid_output_text) == 0):
                    continue
                # valid_loss = model(text, image, valid_output_text, white_image, generate = False)
                valid_loss = model(text, image, valid_output_text, generate = False)

            else:
                # image = None
                image = Image.open('white_image.jpeg').convert('RGB')  
                # white_image = True
                # continue

                valid_output_text = valid_output[j][0]

                if(len(valid_output_text) == 0):
                    continue

                valid_loss = model(text, image, valid_output_text, generate = False)

            epoch_valid_loss += valid_loss.item()

        return epoch_valid_loss   

min_val_loss = 1e6
for epoch in range(5):
    

    print("Epoch : ", epoch)

    print("Train epoch")
    train_epoch(train_input, train_output)
   
    print("Valid epoch")
    # with torch.no_grad():
    valid_loss = valid_epoch(valid_input, valid_output)

    print('valid loss : ', valid_loss)
    if(valid_loss< min_val_loss):
        print('min val loss : ', min_val_loss)
        min_val_loss = valid_loss
        model.llama_model.save_pretrained(f'./save_model/minigpt_decoder_weights/white_image/epoch_{epoch}')
        torch.save(model.llama_proj.state_dict(), f'./save_model/white_image/llama_proj_epoch_{epoch}')
        print('model saved\n')