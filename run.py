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
    parser.add_argument('--task', type=str, default='train')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    print("Performing task: ", args.task)

    # Initialize all necessary classes
    config = Config()
    blackbox = BlackBox(config.blackbox_cfg)
    suffix_llm = SuffixLLM(config.suffix_cfg, config.data_cfg)
    evaluator = Evaluator(config.evaluator_cfg)

    # Start by training the SuffixLLM
    if args.task == 'train':
        suffix_llm.train_model()

        # Once trained, we need to set it to eval mode.
        suffix_llm.setup_inference()

        # Initialize necessary classes for ORPO
        lbo = LBO(config.lbo_cfg, suffix_llm.model, suffix_llm.tokenizer, blackbox, evaluator)
        inference = Inference(config.inference_cfg, suffix_llm, lbo)

        # Generate the data via LBO
        inference.generate_data(suffix_llm.retraining_data)

        # Train the ORPO model now
        orpo = ORPO(config.suffix_cfg, suffix_llm)
        orpo.train(config.data_cfg["infer_save"])

        print("Training complete!")

# Run.
main()




