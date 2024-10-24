class Tester:
    def __init__(self, config, suffix_llm):
        self.config = config
        self.suffix_llm = suffix_llm
        self.dataset_name = config["eval_dataset"]["name"]
        self.dataset_path = config["eval_dataset"]["data_path"]
        
        print("Class: Tester Initialized")
    
    def load_dataset(self):
        with open(self.dataset_path, 'r') as f:
            self.dataset = f.readlines()
        
        print("[Tester] Dataset Loaded")
        
    def evaluate(self):
        self.load_dataset()
        self.suffix_llm.load_orpo_model()
        
        
        
    
    
    