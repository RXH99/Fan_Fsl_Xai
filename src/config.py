"""
配置管理：从 yaml 加载实验参数
"""

import yaml
import os

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

    def __getattr__(self, name):
        if name in self.cfg:
            return self.cfg[name]
        raise AttributeError(f"Config missing: {name}")

    @property
    def device(self):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def num_classes(self):
        return len(self.cfg.get("data", {}).get("class_names", []))


def get_default_config():
    """返回默认配置字典，便于快速构建"""
    return {
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "window_size": 1024,
            "window_step": 512,       # 1024//2
            "train_ratio": 0.8,
            "class_names": [],         # 由数据集决定
        },
        "training": {
            "cnn": {
                "batch_size": 64,
                "epochs": 30,
                "lr": 0.001,
            },
            "fewshot": {
                "episodes": 2000,
                "val_episodes": 200,
                "ways": 0,            # 由类别数决定
                "shot": 5,
                "query": 5,
                "lr": 0.0001,
            }
        },
        "model": {
            "cnn": {
                "channels": [16, 32, 64],
                "kernel_sizes": [15, 7, 3],
            },
            "encoder_dim": 64,
        },
        "paths": {
            "output_dir": "outputs",
        }
    }
