"""
ML Pipeline
"""
import argparse
import src.basic_clean
#import src.train_test_model
#import src.check_score
import logging

'''
Main script to execute different ML steps

Define different steps necessary for model training and testing
- preprocessing/basic cleaning
- train/test model
- eval metrics
'''

def execute(args):
    """
    Execute the pipeline
    """
    logging.basicConfig(level=logging.INFO)

    if args.mlstep == "all" or args.mlstep == "basic_clean":
        logging.info("Basic cleaning has begun")
        src.basic_clean.execute_basic_cleaning()

    if args.mlstep == "all" or args.mlstep == "train_test_model":
        logging.info("Train/Test model has begun")
        src.train_test_model.train_test_model()

    if args.mlstep == "all" or args.mlstep == "eval_metrics":
        logging.info("Eval metrics has begun")
        src.eval_metrics.eval_metrics()


if __name__ == "__main__":
    """
    Main entrypoint
    """
    parser = argparse.ArgumentParser(description="ML pipeline")

    parser.add_argument(
        "--mlstep",
        type=str,
        choices=["basic_clean",
                 "train_test_model",
                 "eval_metrics",
                 "all"],
        default="all",
        help="ML pipeline steps: basic_clean, train_test_model, eval_metrics, all"
    )

    main_args = parser.parse_args()

    execute(main_args)