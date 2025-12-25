import numpy as np
import torch
import matplotlib.pyplot as plt


# 创建原始权重矩阵, 入参为输入维度和输出维度
def create_original_matrix(d, k):
    return torch.randn(d, k)

# 实现 LoRA 分解
class LoRALayer:
    def __init__(self, d, k, r):
        self.d = d  # 输入维度
        self.k = k  # 输出维度
        self.r = r  # LoRA 低秩维度
        self.lora_A = torch.randn(d, r) / np.sqrt(r) # LoRA A 矩阵, 除以 r 的平方根进行缩放, 避免梯度爆炸
        self.lora_B = torch.zeros(r, k) # LoRA B 矩阵, 初始化为零

        # 需要梯度
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True

    # 向前传播, 入参 x  是输入张量
    def forward(self, x):
        return (self.lora_A @ self.lora_B) @ x # @ 表示矩阵乘法, 即 torch.matmul

    def get_weight_update(self):
        return self.lora_A @ self.lora_B # 返回权重更新矩阵


# 训练过程, 入参原始权重, LoRA 层, 训练次数, 学习率
def train_lora(original_weight, lora_layer, num_iterations = 10000, learning_rate = 0.01):
    # 选用 Adam 优化器
    optimizer = torch.optim.Adam([lora_layer.lora_A, lora_layer.lora_B], lr=learning_rate)
    loss_history = [] # 记录损失值

    for it in range(num_iterations):
        # 梯度清零
        optimizer.zero_grad()
        # 当前 LoRA 权重
        current_weight = lora_layer.get_weight_update()
        # 计算损失, 这里使用均方误差
        loss = torch.nn.functional.mse_loss(current_weight, original_weight)

        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        if it % 100 == 0:
            print(f"Iteration {it}, Loss: {loss.item():.6f}")

    return loss_history


# 修改可视化函数
def visualize_results(original_weight, lora_approximation, losses):
    plt.figure(figsize=(15, 5))

    # 绘制损失曲线
    plt.subplot(131)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Iterations (x100)')
    plt.ylabel('MSE Loss')
    plt.xlim(0, 10)
    plt.ylim(0, 1.1)

    # 绘制原始权重矩阵
    plt.subplot(132)
    plt.imshow(original_weight.detach().numpy(), cmap='viridis')
    plt.title('Original Weight Matrix')
    plt.colorbar()

    # 绘制LoRA近似后的矩阵
    plt.subplot(133)
    plt.imshow(lora_approximation.detach().numpy(), cmap='viridis')
    plt.title('LoRA Approximation')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # 打印矩阵数值
    print("\n原始权重矩阵的一部分(5x5):")
    print(original_weight.detach().numpy()[:5, :5])

    print("\nLoRA近似后的矩阵的一部分(5x5):")
    print(lora_approximation.detach().numpy()[:5, :5])

    # 计算误差矩阵
    error_matrix = original_weight.detach().numpy() - lora_approximation.detach().numpy()
    print("\n误差矩阵的一部分(5x5):")
    print(error_matrix[:5, :5])

    # 计算一些统计指标
    print("\n统计指标:")
    print(f"最大误差: {np.abs(error_matrix).max():.6f}")
    print(f"平均误差: {np.abs(error_matrix).mean():.6f}")
    print(f"误差标准差: {np.abs(error_matrix).std():.6f}")

    # 计算相似度
    from scipy.stats import pearsonr
    orig_flat = original_weight.detach().numpy().flatten()
    lora_flat = lora_approximation.detach().numpy().flatten()
    correlation, _ = pearsonr(orig_flat, lora_flat)
    print(f"矩阵相似度(相关系数): {correlation:.6f}")

def main():
    d, k, r = 10, 10, 3  # 输入维度, 输出维度, LoRA 低秩维度
    original_weight = create_original_matrix(d, k)  # 创建原始权重矩阵
    lora_layer = LoRALayer(d, k, r)  # 初始化 LoRA 层
    losses = train_lora(original_weight, lora_layer, 1000, 0.01)  # 训练 LoRA
    lora_approximation = lora_layer.get_weight_update()  # 获取 LoRA 近似后的权重矩阵

    # 比较参数量
    original_params = d * k
    lora_params = d * r + r * k
    reduction = (1 - lora_params / original_params) * 100
    print(f"原始参数量: {original_params}, LoRA 参数量: {lora_params}, 参数量减少: {reduction:.2f}%")

    visualize_results(original_weight, lora_approximation, losses)  # 可视化结果


if __name__ == "__main__":
    main()
