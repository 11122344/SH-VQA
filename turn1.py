from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RX, RY
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindspore import context, nn, Tensor
import numpy as np

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

n_qubits = 2
init_params = np.random.uniform(0, 2 * np.pi, size=2 * n_qubits)

# 定义哈密顿量
H = QubitOperator('Z0 Z1') + QubitOperator('X0 X1') + QubitOperator('Y0 Y1')
hamiltonian = Hamiltonian(H)

# 构建参数化线路
def build_ansatz():
    circuit = Circuit()
    param_names = []
    for i in range(n_qubits):
        rx = f'rx{i}'
        ry = f'ry{i}'
        circuit += RX(rx).on(i)
        circuit += RY(ry).on(i)
        param_names.extend([rx, ry])
    return circuit, param_names

ansatz_circuit, param_names = build_ansatz()

# 构建模拟器
sim = Simulator('mqvector', n_qubits)
grad_ops = sim.get_expectation_with_grad(hamiltonian, ansatz_circuit)

# 替代 VQE，用 MQAnsatzOnlyLayer 包装
net = MQAnsatzOnlyLayer(grad_ops)
optimizer = nn.Adam(net.trainable_params(), learning_rate=0.1)
train_net = nn.TrainOneStepCell(net, optimizer)
train_net.set_train()

# 训练
convergence_tol = 1e-5
max_epochs = 100
last_energy = None

for epoch in range(max_epochs):
    energy = train_net().asnumpy()
    print(f"Step {epoch}: Energy = {energy.item():.6f}")
    if last_energy is not None and abs(energy - last_energy) < convergence_tol:
        break
    last_energy = energy

print("\n最终能量 E(θ):", energy)
print("对应参数 θ:", net.weight.asnumpy())
