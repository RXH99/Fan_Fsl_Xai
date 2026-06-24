"""
工具模块单元测试
验证 src/utils.py 中各函数的正确性

运行: python tests/test_utils.py
"""
import os
import sys
import unittest
import torch
import tempfile
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    load_yaml_config,
    load_model_weights,
    get_uwt_params,
    check_gpu_availability
)


class TestLoadYAMLConfig(unittest.TestCase):
    """测试 YAML 配置加载功能"""
    
    def test_load_valid_config(self):
        """测试加载有效的配置文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'test': 'value', 'number': 42}, f)
            temp_path = f.name
        
        try:
            config = load_yaml_config(temp_path)
            self.assertEqual(config['test'], 'value')
            self.assertEqual(config['number'], 42)
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with self.assertRaises(FileNotFoundError):
            load_yaml_config("nonexistent_file.yaml")
    
    def test_load_invalid_yaml(self):
        """测试加载无效的 YAML 文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with self.assertRaises(yaml.YAMLError):
                load_yaml_config(temp_path)
        finally:
            os.unlink(temp_path)


class TestLoadModelWeights(unittest.TestCase):
    """测试模型权重加载功能"""
    
    def setUp(self):
        """创建测试模型和临时权重文件"""
        self.model = torch.nn.Linear(10, 5)
        self.device = torch.device('cpu')
        
        # 创建临时权重文件
        self.temp_dir = tempfile.mkdtemp()
        self.valid_weight_path = os.path.join(self.temp_dir, "valid_weights.pth")
        torch.save(self.model.state_dict(), self.valid_weight_path)
        
        # 创建空权重文件
        self.empty_weight_path = os.path.join(self.temp_dir, "empty_weights.pth")
        open(self.empty_weight_path, 'w').close()
    
    def tearDown(self):
        """清理临时文件"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_valid_weights(self):
        """测试加载有效权重"""
        missing, unexpected = load_model_weights(
            self.model, self.valid_weight_path, self.device, strict=True
        )
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(unexpected), 0)
    
    def test_load_nonexistent_weights(self):
        """测试加载不存在的权重文件"""
        with self.assertRaises(FileNotFoundError):
            load_model_weights(
                self.model, "nonexistent.pth", self.device, required=True
            )
    
    def test_load_empty_weights(self):
        """测试加载空权重文件"""
        with self.assertRaises(RuntimeError):
            load_model_weights(
                self.model, self.empty_weight_path, self.device
            )
    
    def test_load_optional_weights(self):
        """测试加载可选权重（不存在时不报错）"""
        missing, unexpected = load_model_weights(
            self.model, "nonexistent.pth", self.device, required=False
        )
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])


class TestGetUWTParams(unittest.TestCase):
    """测试 UWT 参数提取功能"""
    
    def test_get_default_params(self):
        """测试获取默认 UWT 参数"""
        config = {}
        params = get_uwt_params(config)
        
        self.assertTrue(params['enabled'])
        self.assertEqual(params['steps'], 3)
        self.assertEqual(params['tau'], 0.3)
        self.assertEqual(params['mix_ratio'], 0.8)
        self.assertEqual(params['beta_default'], 1.0)
    
    def test_get_custom_params(self):
        """测试获取自定义 UWT 参数"""
        config = {
            'inference': {
                'uwt': {
                    'enabled': False,
                    'steps': 5,
                    'tau': 0.5,
                    'mix_ratio': 0.9,
                    'beta_default': 2.0
                }
            }
        }
        params = get_uwt_params(config)
        
        self.assertFalse(params['enabled'])
        self.assertEqual(params['steps'], 5)
        self.assertEqual(params['tau'], 0.5)
        self.assertEqual(params['mix_ratio'], 0.9)
        self.assertEqual(params['beta_default'], 2.0)
    
    def test_get_beta_search(self):
        """测试获取 beta 搜索范围"""
        config = {
            'inference': {
                'uwt': {
                    'beta_search': {
                        '5w1s': 4.0,
                        '5w5s': 2.0
                    }
                }
            }
        }
        params = get_uwt_params(config)
        
        self.assertEqual(params['beta_search']['5w1s'], 4.0)
        self.assertEqual(params['beta_search']['5w5s'], 2.0)


class TestCheckGPUAvailability(unittest.TestCase):
    """测试 GPU 可用性检查功能"""
    
    def test_returns_device(self):
        """测试返回正确的设备对象"""
        device = check_gpu_availability()
        self.assertIsInstance(device, torch.device)
        self.assertIn(str(device.type), ['cuda', 'cpu'])


if __name__ == '__main__':
    print("="*60)
    print("🧪 运行 src/utils.py 单元测试")
    print("="*60)
    unittest.main(verbosity=2)
