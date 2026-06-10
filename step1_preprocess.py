"""
Step 1: 数据预处理

读取风机 .mat 数据集，按 189 类构建，保存为 preprocessed.npz

运行: python step1_preprocess.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"#设置环境变量，解决某些库重复加载的问题

from src.data.preprocess import FanDataPreprocessor
from src.config import get_default_config


def run():
    config = get_default_config()
    config["data"]["raw_dir"] = "data"

    processor = FanDataPreprocessor(config)
    X_train, y_train, X_test, y_test, X_val, y_val = processor.run(
        data_root="data",
        output_dir="data/processed",
        info_path="data/dataset_info.json",
    )

    from collections import Counter
    print(f"\n{'='*50}")
    print("📊 数据预处理完成")
    print(f"{'='*50}")
    for name, X, y in [("train", X_train, y_train),
                        ("test", X_test, y_test),
                        ("val", X_val, y_val)]:
        print(f"  {name}: {len(X)} 样本, {len(set(y.tolist()))} 类")
    print(f"  信号长度: 1024")
    print(f"  X range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"\n✅ 预处理完成，可继续执行 step2_train_cnn.py")


if __name__ == "__main__":
    run()
