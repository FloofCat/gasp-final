import tqdm
import pandas as pd
from .logging import Logging
import time

class Inference:
    def __init__(self, config, suffix_llm, lbo):
        self.suffix_llm = suffix_llm
        self.config = config
        self.lbo = lbo
        self.embeddings = self.lbo.embeddings

        self.logger = Logging(self.config["inference_logs"])
        
        self.epoches = self.config["epochs"]
        self.inference_csv = self.config["infer_save"]
        
        self.prompts = []
        self.chosen = []
        self.rejected = []
        print("Class: Inference Initialized")

    def generate_prompt(self, goal, sr, custom):
        startTime = time.time()
        suffixes, _ = self.suffix_llm.generate_suffix(goal)
        endTime = time.time() - startTime

        suffix = suffixes[0].strip()
        # Remove period if it's the last character
        suffix = suffix[:-1] if suffix[-1] == '.' else suffix
        
        # If first character of the suffix is a dot, no need to add a space
        if suffix[0] == '.':
            prompt = goal + suffix
        else:
            prompt = goal + " " + suffix

        response = self.lbo.blackbox.query(prompt)

        # Grade with evaluator
        score_custom = None
        score_sr = None
        if custom:
            score_custom = self.lbo.evaluator.evaluate(prompt, response)
        
        if sr:
            score_sr = self.lbo.evaluator.evaluate_strongreject(prompt, response)

        return prompt, response, score_custom, score_sr, endTime
     
    def generate_data(self, data):
        for i in tqdm.tqdm(range(data.shape[0])):
            epoch = 0
            
            goal = data['goal'].iloc[i]
            goal = goal.strip()
            
            while(epoch < self.epoches):
                epoch += 1
                suffixes, output_string = self.suffix_llm.generate_suffix(goal)
                
                embeddings = self.embeddings.get_embeddings(suffixes)
                
                # If no embeddings are found
                if(embeddings.shape[0] - 1 <= 0):
                    continue
                
                reduced_embeddings = self.embeddings.dimensionality_reduction(embeddings)
                
                # Making the mappings in lower dimension for LBO
                mappings = {}
                for j, suffix in enumerate(suffixes):
                    mappings[tuple(reduced_embeddings[j])] = suffix
                    
                prompt, score, _, expected_string, _ = self.lbo.lbo(goal, mappings)
                
                goal = prompt
                print(f"Epoch: {epoch} | Prompt: {prompt} | Score: {score}")
                
                self.prompts.append(prompt)
                self.chosen.append(expected_string)
                self.rejected.append(output_string)
                
                if score < 1:                
                    self.logger.log(["PROMPT: " + prompt, "CHOSEN: " + expected_string, "REJECTED: " + output_string])
                    break
        self.to_csv()
    
    def to_csv(self):
        df = pd.DataFrame()
        df['prompt'] = self.prompts
        df['chosen'] = self.chosen
        df['rejected'] = self.rejected
        
        df.to_csv("./gasp-llama2/logs/inference.csv", index=False)
                    
                    