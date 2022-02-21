"""Tune Hyperparameters for VAE.
Z dimensions: [32, 64, 128, 256, 512]
1 layer: [32, 64, 128, 256, 512] (should be greater than z_dim)
2 layers: [(64, 32), (128, 64), (128, 32)...]
"""

import argparse
import json
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to the dataset")
    parser.add_argument("vocab_dir", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--num_epoch", type=int, default=30, help="Number of epochs")
    parser.add_argument("--model_name", type=str, default="VAE", help="Model name")
    args = parser.parse_args()
    config = {"device": "/device:GPU:0", "model": "MultiTextVAETransfer", "lr": 0.001}
    config_filename = "config_vae_hp.json"
    commandline = ["python",
                   os.path.join(
                       os.path.dirname(os.path.abspath(__file__)),
                       "train_vae.py"
                   ),
                   args.data_dir,
                   os.path.join(
                       os.path.dirname(os.path.abspath(__file__)),
                       config_filename
                   ),
                   "--vocab_dir={}".format(args.vocab_dir),
                   "--log_dir={}".format(args.log_dir),
                   "--lower",
                   "--print_every=50",
                   "--num_epoch={}".format(args.num_epoch),
                   "--model_name={}".format(args.model_name)
                   ]

    # First: tune learning rate
    for lr in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        config["lr"] = lr
        config["p_dims_1"] = [32, 64]
        with open(config_filename, "w") as f:
            f.write(json.dumps(config, indent=2))
        print(f"--- Learning rate: {lr}")
        subprocess.run(commandline)

    # Second: tune number and size of hidden layers
    dims_list = [16, 32, 64, 128, 256, 512]
    for lr in [0.01, 0.001]:
        config['lr'] = lr
        for (z_i, z_dims) in enumerate(dims_list):
            # 0 layers
            config["p_dims_1"] = [z_dims]
            with open(config_filename, "w") as f:
                f.write(json.dumps(config, indent=2))
            print("--- 0 LAYERS", config["p_dims_1"])
            subprocess.run(commandline)
            for (h_i, h_dims_1) in enumerate(dims_list[z_i+1:]):
                config["p_dims_1"] = [z_dims, h_dims_1]
                with open(config_filename, "w") as f:
                    f.write(json.dumps(config, indent=2))
                print("--- 1 LAYER ---", config["p_dims_1"])
                print(commandline)
                subprocess.run(commandline)
                for h_dims_2 in dims_list[z_i+h_i+2:]:
                    config["p_dims_1"] = [z_dims, h_dims_1, h_dims_2]
                    with open(config_filename, "w") as f:
                        f.write(json.dumps(config, indent=2))
                    print("--- 2 LAYERS ---", config["p_dims_1"])
                    print(commandline)
                    subprocess.run(commandline)

if __name__ == "__main__":
    main()
