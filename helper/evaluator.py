import torch
from .logging import Logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

class Evaluator:
    def __init__(self, config):
        self.question_ids = ["1.b", "2.b", "3.b", "4.b", "5.b", "6.b", "7.b", "8.b", "9.b", "10.b", "11.b", "12.b", "13.b", "14.b", "15.b", "16.b", "17.b", "18.b", "19.b"]
        self.config = config
        self.seed = self.config['seed']
        self.eval_path = self.config["task_msg"]
        self.model_name = self.config['evaluator_model']['model']
        self.model_path = self.config['evaluator_model']['model_path']
        self.provider = self.config['evaluator_model']['provider']
        self.device = self.config['evaluator_model']['device']

        self.temperature = self.config['evaluator_params']['temperature']
        self.top_p = self.config['evaluator_params']['top_p']
        self.max_length = self.config['evaluator_params']['max_length']

        self.logger = Logging(self.config['evaluator_model']['logging_file'])
        print("Class: Evaluator Initialized")

        # Load the evaluator task
        self.load_task()
        self.load_model()
    
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                        torch_dtype=torch.float16,
                                                        trust_remote_code=True).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                        trust_remote_code=True,
                                                        use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pipe = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=self.device)

    def load_task(self):
        with open(self.eval_path, 'r') as f:
            self.eval_msg = f.read()

    def evaluate(self, response):
        chat = {"role": "user", "content": self.eval_msg + "\nAI: " + response}
        
        llm_response = self.pipe(chat,
                                max_length=self.max_length,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                return_prompt=True)[0]["generated_text"][-1]["content"]
                
        net_score = 0
        
        for line in llm_response.split("\n"):
            for qid in self.question_ids:
                if qid in line: 
                    score = float(line.split(" ")[1].strip())
                    net_score += score
                    break
        
        self.logger.log(["AI: " + response, "EVALUATOR: " + llm_response, "SCORE: " + str(net_score / len(self.question_ids))])
        return net_score / len(self.question_ids)


