#!/usr/bin/env python
import argparse
import os
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_config(path: str) -> dict:
    with open(path, 'r') as fin:
        return yaml.safe_load(fin)


def visualize_from_config(cfg: dict):
    vis_cfg = cfg.get('visualization', {})
    pred_dir = vis_cfg.get('pred_dir')
    output = vis_cfg.get('output', 'benchmark_plot.png')
    frames = []
    for fn in os.listdir(pred_dir):
        if fn.endswith('_benchmark_predictions.csv'):
            df = pd.read_csv(os.path.join(pred_dir, fn), index_col=0)
            melted = df.mean().to_frame(name='prediction').reset_index().rename(columns={'index': 'sig'})
            melted['embedding'] = fn.replace('_benchmark_predictions.csv', '')
            frames.append(melted)
    if not frames:
        raise ValueError(f'No prediction files found in {pred_dir}')
    data = pd.concat(frames)
    plt.figure(figsize=(8,4))
    sns.barplot(data=data, x='sig', y='prediction', hue='embedding')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output)


def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results from YAML config')
    parser.add_argument('config', type=str, help='Path to YAML config file')
    args = parser.parse_args()
    cfg = load_config(args.config)
    visualize_from_config(cfg)


if __name__ == '__main__':
    main()
