import os
import tempfile
import subprocess
import argparse

def retrain(num_runs, seed, config_path, base_out_dir):
    with open(config_path, 'r') as file:
        config = file.readlines()

    for i in range(num_runs):
        modified_config = []
        for line in config:
            if line.startswith('out_dir'):
                modified_config.append(f"out_dir = '{base_out_dir}/model_{i+1}'\n")
            elif line.startswith('seed'):
                modified_config.append(f"seed = {seed}\n")
            else:
                modified_config.append(line)

        temp_config_path = tempfile.mktemp()
        with open(temp_config_path, 'w') as temp_file:
            temp_file.writelines(modified_config)

        command = f"python ./models/nanoGPT/train.py ./models/nanoGPT/{temp_config_path}"
        subprocess.run(command, shell=True)

        os.remove(temp_config_path)
        seed += 1  

def main():
    parser = argparse.ArgumentParser(description='Retrain models with different seeds and output directories.')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of training runs to perform.')
    parser.add_argument('--seed', type=int, default=42, help='Initial random seed for training.')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the training configuration file.')
    parser.add_argument('--base_out_dir', type=str, default='out-shakespeare-char', help='Base directory for output models.')
    
    args = parser.parse_args()
    retrain(args.num_runs, args.seed, args.config_path, args.base_out_dir)

if __name__ == "__main__":
    main()
