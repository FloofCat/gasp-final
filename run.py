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
    parser.add_argument('--task', type=str, default='all')

    args = parser.parse_args()
    return args

def train(suffix_llm):
    suffix_llm.train_model()

    print("Pre-training complete!")

def test(config, suffix_llm, evaluator, blackbox):
    # Load the trained model
    # suffix_llm.setup_inference()

    lbo = LBO(config.data_cfg, suffix_llm.model, suffix_llm.tokenizer, blackbox, evaluator)
    inference = Inference(config.data_cfg, suffix_llm, lbo)
    orpo = ORPO(config.suffix_cfg, suffix_llm, config.blackbox_cfg["black_box_model"]["model"])

    orpo.load_models()

    tester = Tester(config.data_cfg, suffix_llm, inference, orpo)
    tester.eval_orpo()
    
    print("Evaluation complete!")

def main():
    args = get_args()
    print("Performing task: ", args.task)

    # Initialize all necessary classes
    config = Config()
    suffix_llm = SuffixLLM(config.suffix_cfg, config.data_cfg)
    evaluator = Evaluator(config.evaluator_cfg)
    blackbox = BlackBox(config.blackbox_cfg)

    # Start by training the SuffixLLM
    if args.task == 'all':
        train(suffix_llm)
        test(config, suffix_llm, evaluator, blackbox)

    elif args.task == 'train':
        train(suffix_llm)

    elif args.task == 'eval':
        test(config, suffix_llm, evaluator, blackbox)

# Run.
main()