import argparse
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

from src.config import ConfigManager
from src.token_classification.dataset import create_dataset
from src.token_classification.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Rayan International AI Contest: Compositional Retrieval")
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="Path to the config file")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config_path = args.config
    config = ConfigManager(config_path)  # Initialize the singleton with the config file


    tokenizer = DistilBertTokenizerFast.from_pretrained(config.get("model")["pretrained"])

    dataset, id2label, label2id, data_collator = create_dataset(tokenizer)

    model = DistilBertForTokenClassification.from_pretrained(
        config.get("model")["pretrained"],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )


    train(model, dataset, data_collator, tokenizer, id2label)

if __name__ == "__main__":
    main()