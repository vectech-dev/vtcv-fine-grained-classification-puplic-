import abc
import json
import sys

# from vtcv_fine_grained_classification.finetune import finetune
from vtcv_fine_grained_classification.src.configs.config import ExperimentationConfig
from vtcv_fine_grained_classification.test.test import test
# from vtcv_fine_grained_classification.test.test_staging import test_staging
from vtcv_fine_grained_classification.train.train import train


class Executor:
    def __init__(self, args):
        print(args.config)
        if args.config is not None:
            with open(args.config) as json_file:
                self.config = json.load(json_file)

            if "mode" in args and (
                "mode" not in self.config or self.config["mode"] != args.mode
            ):
                self.config["mode"] = args.mode

            if args.batch_size:
                self.config["batch_size"] = args.batch_size
        else:
            self.config = None
        self.args = args

    @abc.abstractmethod
    def execute(self):
        return NotImplemented


class TrainingExecutor(Executor):
    def __init__(self, args):
        super(TrainingExecutor, self).__init__(args)
        assert self.config is not None, "need configuration file for training provided"
        self.config[
            "mode"
        ] = args.mode  # Add this line to update the mode value in the configuration

    def execute(self):
        train_settings = ExperimentationConfig.parse_obj(self.config)
        train(train_settings)


''' class FinetuneExecutor(Executor):
    def __init__(self, args):
        super(FinetuneExecutor, self).__init__(args)
        assert self.config is not None, "need configuration file for training provided"
        self.config[
            "mode"
        ] = args.mode  # Add this line to update the mode value in the configuration

    def execute(self):
        finetune_settings = ExperimentationConfig.parse_obj(self.config)
        finetune(finetune_settings) '''


class TestingExecutor(Executor):
    def __init__(self, args):
        super(TestingExecutor, self).__init__(args)
        assert self.config is None, "No configuration needed for testing"

    def execute(self): 
        test(self.args.exp_dir)


EXECUTORS = {
    "train": TrainingExecutor,
    "test": TestingExecutor,
    # "finetune": FinetuneExecutor,
}


def get_executor(mode: str) -> Executor:
    return EXECUTORS[mode]


def run(args=None):
    from argparse import ArgumentParser

    parser = ArgumentParser("vtcv_fine_grained_classification")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=sorted(EXECUTORS.keys()),
        help="Overwrite mode from the configuration file",
    )

    parser.add_argument(
        "--config", type=str, help="The configuration file for training"
    )

    parser.add_argument(
        "--exp-dir", type=str, help="The experiment directory for tests"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Overwrite the batch size from configuration",
        default=None,
    )

    parser.add_argument(
        "--alt-test",
        type=str,
        help="Provide additional test data for testing",
        default=None,
    )

    parser.add_argument(
        "--skip-default-test",
        action="store_true",
        help="Provide additional test data (staging dataset) for testing",
        default=False,
    )

    parser.add_argument(
        "--staging",
        help="True if the test is for staging",
        default=False,
    )
    args = parser.parse_args(args)
    print(args)  # Add this line to print the parsed arguments

    ExecutorClass = get_executor(args.mode)
    print("Executor class:", ExecutorClass)  # Add this line to print the Executor class
    executor = ExecutorClass(args)
    print("executor object:", executor)  # Add this line to print the executor object
    executor.execute()


if __name__ == "__main__":
    run()
