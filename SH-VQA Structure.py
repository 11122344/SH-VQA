import numpy as np
import mindquantum as mq
from mindquantum import Simulator, Circuit, Hamiltonian, RY, RZ, CNOT, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindspore import nn, context, Tensor
from mindspore.common import dtype as mstype
import matplotlib.pyplot as plt

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

delta = 1  #XXZ各向异性系数

hamiltonian = Hamiltonian(
    QubitOperator('Z0 Z1') + 
    QubitOperator('X0 X1') + 
    delta * QubitOperator('Y0 Y1')
)

num_qubits = 2

# 定义试验态线路
def ansatz_circuit(theta):
    circ = Circuit()
    for i in range(num_qubits):
        circ += RY(f'theta_{i}').on(i) 
    circ += CNOT(0, 1)
    return circ

ansatz_param_num = num_qubits

# 定义海森堡线路
def heisenberg_circuit(alpha, phi):
    circ = Circuit()
    for i in range(num_qubits):
        circ += RY(f'alpha_{i}').on(i)
        circ += RZ(f'phi_{i}').on(i)
    circ += CNOT(0, 1)
    return circ

heisenberg_alpha_param_num = num_qubits
heisenberg_phi_param_num = num_qubits

# 定义总变分线路
def total_circuit():
    u_circ = Circuit()
    for i in range(num_qubits):
        u_circ += RY(f'theta_{i}').on(i)
    u_circ += CNOT(0, 1)
    t_circ = Circuit()
    for i in range(num_qubits):
        t_circ += RY(f'alpha_{i}').on(i)
        t_circ += RZ(f'phi_{i}').on(i)
    t_circ += CNOT(0, 1)
    
    return u_circ + t_circ 


total_param_num = ansatz_param_num + heisenberg_alpha_param_num + heisenberg_phi_param_num  # 2+2+2=6


np.random.seed(42)
initial_params = np.random.normal(0, 0.1, total_param_num)
sim = Simulator('mqvector', num_qubits)
total_circ = total_circuit()

# 生成期望值和梯度计算算子
grad_ops = sim.get_expectation_with_grad(hamiltonian, total_circ)

net = MQAnsatzOnlyLayer(grad_ops)
net.weight = Tensor(initial_params, dtype=mstype.float32)

# 使用MindSpore的Adam优化器
optimizer = nn.Adam(net.trainable_params(), learning_rate=0.01)
train_net = nn.TrainOneStepCell(net, optimizer)

energy_history = []
max_epochs = 300
tol = 1e-6

print("开始训练SH-VQA...")
for epoch in range(max_epochs):
    # 前向传播并更新参数
    energy = train_net()
    energy_val = energy.asnumpy().item()
    energy_history.append(energy_val)

    if (epoch + 1) % 10 == 0:
        print(f"Step {epoch+1:3d}: Energy = {energy_val:.6f}")
    
    if epoch >= 1 and abs(energy_history[-1] - energy_history[-2]) < tol:
        print(f"\n收敛于第 {epoch+1} 步")
        break

final_energy = energy_history[-1]
final_params = net.weight.asnumpy()

final_theta = final_params[:ansatz_param_num]
final_alpha = final_params[ansatz_param_num: ansatz_param_num + heisenberg_alpha_param_num]
final_phi = final_params[ansatz_param_num + heisenberg_alpha_param_num:]

print("\n" + "="*50)
print("最终结果:")
print(f"最终能量值: {final_energy:.6f}")
print(f"理论基态能量: -3.000000")
print(f"最终参数 θ : {final_theta.round(4)}")
print(f"最终参数 α : {final_alpha.round(4)}")
print(f"最终参数 φ : {final_phi.round(4)}")
print("="*50 + "\n")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(energy_history)+1), energy_history, label="Experimental value")
plt.axhline(y=-3.0, color='r', linestyle='--', label="Theoretical value")
plt.xlabel("Steps")
plt.ylabel("Energy")
plt.title("SH-VQA simulation")
plt.legend()
plt.grid(True)
plt.show()
