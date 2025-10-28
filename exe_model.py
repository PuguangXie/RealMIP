import argparse
import torch
import yaml
import os
from main_model import TSB_eICU
from dataset import get_dataloader
from utils import train, test
from lstm import LSTMModel
from pytorch_lightning import seed_everything


def parse_arguments():
    parser = argparse.ArgumentParser(description="RealMIP")
    parser.add_argument("--config", type=str, default="base.yaml", help="Path to the configuration file")
    parser.add_argument('--device', default='cuda:0', help='Device for computation (e.g., cpu or cuda)')
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility")
    parser.add_argument("--testmissingratio", type=float, default=0.1, help="Missing ratio for testing")
    parser.add_argument("--train_model", action='store_true', help="Flag to indicate whether to train the model")
    parser.add_argument("--nsample", type=int, default=100, help="Number of samples")

    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def find_model(model_name):
    """
    Finds a pre-trained model.
    """
    assert os.path.isfile(model_name), f'Could not find model checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    return checkpoint.get("ema", checkpoint)  # supports checkpoints from train.py


def main():
    args = parse_arguments()

    seed_everything(args.seed)

    config_path = os.path.join("config", args.config)
    config = load_config(config_path)

    config["model"]["test_missing_ratio"] = args.testmissingratio

    train_loader, valid_loader = get_dataloader(
        stage='train',
        seed=args.seed,
        batch_size=config["train"]["batch_size"],
        missing_ratio=config["model"]["test_missing_ratio"]
    )

    _, valid_loader2 = get_dataloader(
        stage='test',
        seed=args.seed,
        batch_size=128,
        missing_ratio=config["model"]["test_missing_ratio"]
    )

    device = torch.device(args.device)
    model_Gen = TSB_eICU(config, args.device).to(device)

    model_Pre = LSTMModel(
        input_dim=37,
        hidden_dim=256,
        num_layers=2,
        output_dim=2,
        dropout=0.3
    ).to(device)

    ckpt_path_model = 'Gen.pth'
    state_dict_model = find_model(ckpt_path_model)
    model_Gen.load_state_dict(state_dict_model)

    ckpt_path_model1 = 'Pre.pth'
    state_dict_model1 = find_model(ckpt_path_model1)
    model_Pre.load_state_dict(state_dict_model1)

    if args.train_model:
        train(
            model_Gen, model_Pre, config["train"], train_loader, valid_loader=valid_loader
        )

    test(model_Gen, model_Pre, valid_loader2)


if __name__ == "__main__":
    main()