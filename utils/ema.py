import copy
import os
import jittor as jt
from jittor import nn

class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        
        # 初始化时设置EMA模型参数不需要梯度
        for param in self.ema_model.parameters():
            param.stop_grad()
    
    def update(self, model):
        """Update EMA parameters"""
        # 更新EMA参数
        for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
            # 计算新的EMA值，确保结果是Jittor张量
            new_value = self.decay * ema_param + (1.0 - self.decay) * model_param
            # 使用assign方法更新参数，确保类型匹配
            ema_param.assign(new_value)
    
    def get_model(self):
        """Get the EMA model"""
        return self.ema_model
