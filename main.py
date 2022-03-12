"""
ML Pipeline
"""
import argparse
import src.basic_clean
import src.model_functions
import src.check_slice_performance
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

    if args.mlstep == "all" or args.mlstep == "model_functions":
        logging.info("Train/Test model has begun")
        src.model_functions.train_test_model()

    if args.mlstep == "all" or args.mlstep == "slice_performance":
        logging.info("Checking model's performance on slices has begun")
        src.check_slice_performance.check_slice_performance()


if __name__ == "__main__":
    """
    Main entrypoint
    """
    parser = argparse.ArgumentParser(description="ML pipeline")

    parser.add_argument(
        "--mlstep",
        type=str,
        choices=["basic_clean",
                 "model_functions",
                 "slice_performance",
                 "all"],
        default="all",
        help="ML pipeline steps"
    )

    main_args = parser.parse_args()

    execute(main_args)
