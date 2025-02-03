import json

class Config:
    def __init__(self):
        self.suffix_cfg = self.load_config("./gasp-reminder-2/config/suffix_llm.json")
        self.evaluator_cfg = self.load_config("./gasp-reminder-2/config/evaluator.json")
        self.data_cfg = self.load_config("./gasp-reminder-2/config/data.json")
        self.blackbox_cfg = self.load_config("./gasp-reminder-2/config/blackbox.json")
        print("Class: Config Initialized")

    def load_config(self, config_file):
        with open(config_file) as f:
            config = json.load(f)
        
        return config