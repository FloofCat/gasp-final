import torch
from .logging import Logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import os
import anthropic

# import google.generativeai as genai
# from google.api_core.exceptions import ResourceExhausted
# genai.configure(api_key="*************************************************")

import os
os.environ["OPENAI_API_KEY"] = "*************************************************"
os.environ["ANTHROPIC_API"] = "*************************************************"
import backoff 
import openai
from openai import OpenAI

class BlackBox:
    def __init__(self, config):
        self.config = config        
        self.seed = self.config["seed"]
        self.blackbox_name = self.config["black_box_model"]["model"]
        # self.blackbox_path = self.config["black_box_model"]["model_path"]
        self.provider = self.config["black_box_model"]["provider"]
        self.device = self.config["black_box_model"]["device"]
        
        self.temperature = self.config["black_box_params"]["temperature"]
        self.top_p = self.config["black_box_params"]["top_p"]
        self.max_length = self.config["black_box_params"]["max_length"]
        
        self.logger = Logging(self.config["black_box_model"]["logging_file"])
        print("Class: BlackBox Initialized")

        temp_path = {"falcon": "./model-cache/falcon-7b/", 
                    "llama3.1": "./model-cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/",
                    "llama3": "./model-cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/",
                    "llama2": "./model-cache/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/",
                    "mistral": "./model-cache/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db/"}

        self.blackbox_path = temp_path[self.blackbox_name]
        # self.load_model()
        # self.client = OpenAI()
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API"),
        )
        
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.blackbox_path,
                                                        torch_dtype=torch.float16,
                                                        trust_remote_code=True).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.blackbox_path,
                                                        trust_remote_code=True,
                                                        use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("[BlackBox] Model Loaded")

    def query(self, prompt):
        chat = [
            {"role": "system", "content": "You are a AI assistant that is designed to answer user's queries."},
            {"role": "user", "content": prompt}
        ]
        
        # @backoff.on_exception(backoff.expo, BaseException)
        def completions_with_backoff(chat):
            # completion = self.client.chat.completions.create(
            #     model="gpt-3.5-turbo-0125",
            #     messages=chat
            # )
            
            # return completion.choices[0].message.content
            
            try:
                message = self.client.messages.create(
                    model="claude-3-haiku-202403",
                    messages=chat,
                    max_tokens=1024
                )
                return message.choices[0].text
            except Exception as e:
                print(e)
                return None
            
        
        llm_response = completions_with_backoff(chat)
        
        # @backoff.on_exception(backoff.expo, BaseException)
        # def completions_with_backoff(prompt):
        #     response = self.model.generate_content(prompt)
            
        #     return response.text
        
        
        # llm_response = completions_with_backoff(prompt)
        
        # formatted_chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # inputs = self.tokenizer(formatted_chat, return_tensors='pt', add_special_tokens=False, padding=True)
        # inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}
        
        # outputs = self.model.generate(**inputs, max_length=self.max_length, temperature=self.temperature, top_p=self.top_p, do_sample=True)
        
        # llm_response = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        
        # self.logger.log(["PROMPT: " + prompt, "BLACKBOX-RESPONSE: " + llm_response])
        return llm_response