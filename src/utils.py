"""
通用工具函数模块
提供权重加载、配置读取等常用功能
"""
import os
import torch
import yaml
from typing import Dict, Any, Optional


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
        
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML解析错误
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, encoding='utf-8') as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML解析错误 ({config_path}): {e}")
    
    return cfg


def load_model_weights(
    model: torch.nn.Module,
    weights_path: str,
    device: torch.device,
    strict: bool = False,
    required: bool = True
) -> tuple:
    """
    加载模型权重，包含完善的错误处理
    
    Args:
        model: PyTorch模型实例
        weights_path: 权重文件路径
        device: 加载设备
        strict: 是否严格匹配（默认False，允许部分加载）
        required: 权重文件是否必须存在（默认True）
        
    Returns:
        (missing_keys, unexpected_keys) 元组
        
    Raises:
        FileNotFoundError: 权重文件不存在且required=True
        RuntimeError: 权重加载失败
    """
    # 检查文件是否存在
    if not os.path.exists(weights_path):
        if required:
            raise FileNotFoundError(
                f"❌ 权重文件不存在: {weights_path}\n"
                f"   请确认:\n"
                f"   1. 训练脚本已成功运行\n"
                f"   2. 输出目录正确配置\n"
                f"   3. 文件路径拼写无误"
            )
        else:
            print(f"⚠️  权重文件不存在（可选）: {weights_path}")
            return ([], [])
    
    # 检查文件大小
    file_size = os.path.getsize(weights_path)
    if file_size == 0:
        raise RuntimeError(f"❌ 权重文件为空: {weights_path}")
    
    # 加载权重
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"❌ 权重加载失败 ({weights_path}):\n   {str(e)}")
    
    # 应用到模型
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    except Exception as e:
        raise RuntimeError(f"❌ 权重应用失败:\n   {str(e)}")
    
    # 打印加载信息
    if missing_keys:
        print(f"⚠️  缺失的键 ({len(missing_keys)}): {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"⚠️  意外的键 ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    
    print(f"✅ 权重加载成功: {weights_path}")
    print(f"   文件大小: {file_size / 1e6:.2f} MB")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    return (missing_keys, unexpected_keys)


def get_uwt_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    从配置中提取UWT参数
    
    Args:
        config: 完整配置字典
        
    Returns:
        UWT参数字典
    """
    uwt_cfg = config.get('inference', {}).get('uwt', {})
    
    return {
        'enabled': uwt_cfg.get('enabled', True),
        'steps': uwt_cfg.get('steps', 3),
        'tau': uwt_cfg.get('tau', 0.3),
        'mix_ratio': uwt_cfg.get('mix_ratio', 0.8),
        'beta_default': uwt_cfg.get('beta_default', 1.0),
        'beta_search': uwt_cfg.get('beta_search', {
            '5w1s': 3.0,
            '5w5s': 1.0,
            '10w1s': 2.0,
            '10w5s': 1.0
        })
    }


def check_gpu_availability() -> torch.device:
    """
    检查GPU可用性并返回合适的设备
    
    Returns:
        torch.device对象
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  GPU不可用，使用CPU（速度较慢）")
    
    return device
