import tqdm
import pandas as pd
from logging import Logger

class Inference:
    def __init__(self, config, suffix_llm, lbo):
        self.suffix_llm = suffix_llm
        self.config = config
        self.lbo = lbo
        self.embeddings = self.lbo.embeddings

        self.logger = Logger(self.config["inference_logs"])
        
        self.epoches = self.config["epochs"]
        self.inference_csv = self.config["infer_save"]
        
        self.prompts = []
        self.chosen = []
        self.rejected = []
        print("Class: Inference Initialized")
        
    def generate_data(self, data):
        for i in tqdm(len(data)):
            last_score = 2
            epoch = 0
            searches = 0
            
            goal = data['goal'].iloc[i]
            goal = goal.strip()
            
            while(epoch < self.epoches):
                suffixes, output_string = self.suffix_llm.generate_suffix(goal)
                
                embeddings = self.embeddings.get_embeddings(suffixes)
                
                # If no embeddings are found
                if(embeddings.shape[0] - 1 == 0):
                    continue
                
                reduced_embeddings = self.embeddings.dimensionality_reduction(embeddings)
                
                # Making the mappings in lower dimension for LBO
                mappings = {}
                for j, suffix in enumerate(suffixes):
                    mappings[tuple(reduced_embeddings[j])] = suffix
                    
                prompt, score, _, expected_string = self.lbo.lbo(goal, mappings, last_score, searches)
                
                if prompt == None:
                    print("Failed to find a better suffix for epoch: ", epoch)
                    searches += 1
                    continue
                
                searches = 0
                epoch += 1
                
                goal = prompt
                last_score = score
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
        
        df.to_csv("./gasp-final/logs/inference.csv", index=False)
                    
                    