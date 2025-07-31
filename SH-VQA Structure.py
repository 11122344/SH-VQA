import numpy as np
import mindquantum as mq
from mindquantum import (
    Simulator,
    Circuit,
    Hamiltonian,
    RY,
    RZ,
    CNOT,
    QubitOperator
)
from mindquantum.framework import MQAnsatzOnlyLayer
from mindspore import nn, context, Tensor
from mindspore.common import dtype as mstype  # 导入MindSpore的数据类型
import matplotlib.pyplot as plt

# 设置MindSpore运行模式
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

# ==============================
# 1. 定义原始哈密顿量（2量子比特Heisenberg模型）
# ==============================
# 原始哈密顿量：H = Z0Z1 + X0X1 + Y0Y1（基态能量为-3.0）
delta = 1  #各向异性参数

hamiltonian = Hamiltonian(
    QubitOperator('Z0 Z1') + 
    QubitOperator('X0 X1') + 
    delta * QubitOperator('Y0 Y1')
)
num_qubits = 2  # 量子比特数

# ==============================
# 2. 定义试验态线路（薛定谔图景：制备试验态）
# ==============================
def ansatz_circuit(theta):
    """
    试验态线路U(θ)：浅层变分线路，用于制备试验态|ψ(θ)⟩=U(θ)|0⟩
    参数：theta（数组，长度为量子比特数）
    """
    circ = Circuit()
    # 每个量子比特应用RY旋转门（参数化）
    for i in range(num_qubits):
        circ += RY(f'theta_{i}').on(i)  # 使用参数名称
    # 添加CNOT纠缠门（增强线路表达能力）
    circ += CNOT(0, 1)
    return circ

# 试验态线路参数数量（2量子比特→2个参数）
ansatz_param_num = num_qubits

# ==============================
# 3. 定义海森堡线路（海森堡图景：变换哈密顿量）
# ==============================
def heisenberg_circuit(alpha, phi):
    """
    海森堡线路T(α, φ)：用于将原始哈密顿量变换为易测量形式
    参数：alpha（数组，长度为量子比特数）、phi（数组，长度为量子比特数）
    """
    circ = Circuit()
    # 每个量子比特应用RY(α)和RZ(φ)旋转门
    for i in range(num_qubits):
        circ += RY(f'alpha_{i}').on(i)
        circ += RZ(f'phi_{i}').on(i)
    # 添加CNOT纠缠门
    circ += CNOT(0, 1)
    return circ

# 海森堡线路参数数量（2量子比特→α:2个，φ:2个，共4个参数）
heisenberg_alpha_param_num = num_qubits
heisenberg_phi_param_num = num_qubits

# ==============================
# 4. 定义总变分线路（试验态+海森堡）
# ==============================
def total_circuit():
    """
    构建总变分线路V = U(θ) + T(α, φ)
    返回：组合后的量子线路
    """
    u_circ = Circuit()
    # 试验态部分
    for i in range(num_qubits):
        u_circ += RY(f'theta_{i}').on(i)
    u_circ += CNOT(0, 1)
    
    t_circ = Circuit()
    # 海森堡部分
    for i in range(num_qubits):
        t_circ += RY(f'alpha_{i}').on(i)
        t_circ += RZ(f'phi_{i}').on(i)
    t_circ += CNOT(0, 1)
    
    return u_circ + t_circ  # 顺序：先U后T（V|0⟩=T(U|0⟩))

# ==============================
# 5. 构建带可训练参数的量子神经网络
# ==============================
# 总参数数量
total_param_num = ansatz_param_num + heisenberg_alpha_param_num + heisenberg_phi_param_num  # 2+2+2=6

# 初始化参数（正态分布）
np.random.seed(42)
initial_params = np.random.normal(0, 0.1, total_param_num)

# 构建模拟器
sim = Simulator('mqvector', num_qubits)

# 构建总线路
total_circ = total_circuit()

# 生成期望值和梯度计算算子
grad_ops = sim.get_expectation_with_grad(hamiltonian, total_circ)

# 使用MQAnsatzOnlyLayer包装
net = MQAnsatzOnlyLayer(grad_ops)
net.weight = Tensor(initial_params, dtype=mstype.float32)

# ==============================
# 6. 设置优化器和训练策略
# ==============================
# 使用MindSpore的Adam优化器
optimizer = nn.Adam(net.trainable_params(), learning_rate=0.01)
train_net = nn.TrainOneStepCell(net, optimizer)

# 记录能量历史
energy_history = []
max_epochs = 300
tol = 1e-6

# 训练循环
print("开始训练SH-VQA...")
for epoch in range(max_epochs):
    # 前向传播并更新参数
    energy = train_net()
    energy_val = energy.asnumpy().item()
    energy_history.append(energy_val)
    
    # 打印训练进度
    if (epoch + 1) % 10 == 0:
        print(f"Step {epoch+1:3d}: Energy = {energy_val:.6f}")
    
    # 检查收敛
    if epoch >= 1 and abs(energy_history[-1] - energy_history[-2]) < tol:
        print(f"\n收敛于第 {epoch+1} 步")
        break

# ==============================
# 7. 结果打印
# ==============================
final_energy = energy_history[-1]
final_params = net.weight.asnumpy()

# 拆分参数
final_theta = final_params[:ansatz_param_num]
final_alpha = final_params[ansatz_param_num: ansatz_param_num + heisenberg_alpha_param_num]
final_phi = final_params[ansatz_param_num + heisenberg_alpha_param_num:]

print("\n" + "="*50)
print("最终结果:")
print(f"最终能量值: {final_energy:.6f}")
print(f"理论基态能量: -3.000000")
print(f"最终参数 θ (试验态参数): {final_theta.round(4)}")
print(f"最终参数 α (海森堡α参数): {final_alpha.round(4)}")
print(f"最终参数 φ (海森堡φ参数): {final_phi.round(4)}")
print("="*50 + "\n")

# ==============================
# 8. 绘制能量收敛曲线
# ==============================
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(energy_history)+1), energy_history, label="Experimental value")
plt.axhline(y=-3.0, color='r', linestyle='--', label="Theoretical value")
plt.xlabel("Steps")
plt.ylabel("Energy")
plt.title("SH-VQA simulation")
plt.legend()
plt.grid(True)
plt.savefig('sh_vqa_convergence.png')
plt.show()

print("训练完成！结果已保存至 sh_vqa_convergence.png")