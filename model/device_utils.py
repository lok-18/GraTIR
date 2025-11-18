import torch

def get_device(device_str=None):
    """
    获取统一的设备对象，用于在整个项目中保持一致的设备使用
    
    Args:
        device_str: 可选的设备字符串，如"cuda:0", "cpu"等
                   如果为None，则自动使用可用的最佳设备
    
    Returns:
        torch.device: 设备对象
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    return torch.device(device_str)

def to_device(tensor_or_module, device=None):
    """
    将张量或模块移动到指定设备上
    
    Args:
        tensor_or_module: 要移动的张量或模块
        device: 目标设备，如为None则使用get_device()获取
        
    Returns:
        移动到设备上的张量或模块
    """
    if device is None:
        device = get_device()
    
    return tensor_or_module.to(device)

def ensure_same_device(*tensors):
    """
    确保所有张量都在同一设备上，如果不是，则移动到第一个张量的设备上
    
    Args:
        *tensors: 要检查的张量列表
        
    Returns:
        处理后的张量列表，全部在同一设备上
    """
    if not tensors:
        return []
    
    target_device = tensors[0].device
    result = [tensors[0]]
    
    for tensor in tensors[1:]:
        if tensor.device != target_device:
            result.append(tensor.to(target_device))
        else:
            result.append(tensor)
    
    return result if len(tensors) > 1 else result[0] 