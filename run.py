import wandb
wandb.init(mode="disabled")

import argparse
from helper.blackbox import BlackBox
from helper.config import Config
from helper.inference import Inference
from helper.evaluator import Evaluator
from helper.tester import Tester
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
    if args.task == 'pre-train':
        evaluator = Evaluator(config.evaluator_cfg)
        blackbox = BlackBox(config.blackbox_cfg)

        suffix_llm.train_model()

        # Once trained, we need to set it to eval mode.
        suffix_llm.setup_inference()

        # Initialize necessary classes for ORPO
        lbo = LBO(config.data_cfg, suffix_llm.model, suffix_llm.tokenizer, blackbox, evaluator)
        inference = Inference(config.data_cfg, suffix_llm, lbo)

        # Generate the data via LBO
        inference.generate_data(suffix_llm.retraining_data)

        print("Pre-training complete!")
        
    elif args.task == 'train':
        orpo = ORPO(config.suffix_cfg, suffix_llm, config.blackbox_cfg["black_box_model"]["model"])
        orpo.train(config.data_cfg["infer_save"])
        
        print("Training complete!")

    elif args.task == 'eval':
        blackbox = BlackBox(config.blackbox_cfg)
        
        tester = Tester(config.data_cfg, suffix_llm, blackbox)
        tester.test()
        
        print("Evaluation complete!")

# Run.
main()




