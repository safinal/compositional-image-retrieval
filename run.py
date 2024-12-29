import argparse
from model_factory import ModelFactory


def parse_args():
    parser = argparse.ArgumentParser(description="Rayan International AI Contest")
    parser.add_argument("--modelType", type=str, required=True, choices=["retrieval", "token_cls"], help="Specify the type of model to train")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    return parser.parse_args()

def main():
    args = parse_args()

    initializer = ModelFactory(args.config, args.model_type)
    initializer.build()

if __name__ == "__main__":
    main()
