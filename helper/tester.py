class Tester:
    def __init__(self, config):
        self.config = config
        self.dataset_name = config["eval_dataset"]["name"]
        self.dataset_path = config["eval_dataset"]["data_path"]
        
        print("Class: Tester Initialized")
    
    def load_dataset(self):
        with open(self.dataset_path, 'r') as f:
            self.dataset = f.readlines()
        
        print("[Tester] Dataset Loaded")
    
    