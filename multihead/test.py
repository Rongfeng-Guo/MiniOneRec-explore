import torch
import torch.nn.functional as F

# 模拟数据 - 你需要根据实际数据来替换这里的内容
batch_size = 4
prompt_length = 24  # 这里的 24 是你遇到的 prompt_mask 大小
completion_length = 32  # 这里的 32 是你遇到的 completion_mask 大小

# 创建模拟的 mask 数据
prompt_mask = torch.ones(batch_size, prompt_length)  # 模拟 prompt_mask
completion_mask = torch.ones(batch_size, completion_length)  # 模拟 completion_mask

# 输出它们的形状
print(f"prompt_mask shape: {prompt_mask.shape}")
print(f"completion_mask shape: {completion_mask.shape}")

# 确保它们的形状匹配 - 如果不匹配，可以使用 padding 来补齐
if prompt_mask.shape[1] != completion_mask.shape[1]:
    max_length = max(prompt_mask.shape[1], completion_mask.shape[1])
    prompt_mask = F.pad(prompt_mask, (0, max_length - prompt_mask.shape[1]))  # 填充 prompt_mask
    completion_mask = F.pad(completion_mask, (0, max_length - completion_mask.shape[1]))  # 填充 completion_mask

    # 输出填充后的形状
    print(f"After padding:")
    print(f"prompt_mask shape: {prompt_mask.shape}")
    print(f"completion_mask shape: {completion_mask.shape}")

# 尝试连接
attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

# 输出最终的 attention_mask 形状
print(f"attention_mask shape: {attention_mask.shape}")