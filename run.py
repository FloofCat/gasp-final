import wandb
wandb.init(mode="disabled")

import argparse
from helper.blackbox import BlackBox
from helper.config import Config
from helper.inference import Inference
from helper.evaluator import Evaluator
from helper.lbo import LBO
from helper.orpo import ORPO
from helper.suffixllm import SuffixLLM

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pre-train')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    print("Performing task: ", args.task)

    # Initialize all necessary classes
    config = Config()
    suffix_llm = SuffixLLM(config.suffix_cfg, config.data_cfg)
    
    # Start by training the SuffixLLM
    if args.task == 'train':
        suffix_llm.train_model()
        
        print("Training complete!")
        
    elif args.task == 'eval':
        # You need to train the SuffixLLM first before evaluating
        evaluator = Evaluator(config.evaluator_cfg)
        blackbox = BlackBox(config.blackbox_cfg)
        suffix_llm.setup_inference()
        
        # Initialize LBO and Inference.
        lbo = LBO(config.data_cfg, suffix_llm.model, suffix_llm.tokenizer, blackbox, evaluator)
        orpo = ORPO(config.suffix_cfg, suffix_llm, config.blackbox_cfg["black_box_model"]["model"])

        inference = Inference(config.data_cfg, suffix_llm, lbo, orpo)
        inference.evaluate()
        
        orpo.save()        
        print("Evaluation complete!")

# Run.
main()




