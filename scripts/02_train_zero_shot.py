import argparse
from configs.dataset_config import DATASETS, CLASS_WEIGHT_RULES
from src.models_zero_shot import ConvNeXtClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="Kermany", help="Source domain name")
    args = parser.parse_args()
    print(f"Training in-domain ConvNeXt model on {args.source}...")
    print(f"Class weighting applied: {CLASS_WEIGHT_RULES.get(args.source)}")

if __name__ == "__main__":
    main()
