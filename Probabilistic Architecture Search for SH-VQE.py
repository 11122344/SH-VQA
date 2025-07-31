import numpy as np
import mindquantum as mq
from mindquantum import (
    Simulator,
    Circuit,
    Hamiltonian,
    QubitOperator,
    RY,
    RZ,
    CNOT,
    H,
    S,
    X,
    Y,
    Z
)
from mindspore import context
import matplotlib.pyplot as plt
from typing import List

# 设置MindSpore运行模式
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

# ==============================
# 1. 全局参数配置
# ==============================
n_qubits = 2
delta = 2.0

num_architectures = 3
max_epochs = 500
tol = 1e-6
learning_rate = 0.01

base_hamiltonian = QubitOperator('Z0 Z1') + QubitOperator('X0 X1') + delta * QubitOperator('Y0 Y1')
hamiltonian = Hamiltonian(base_hamiltonian)

theoretical_energy = -abs(1 + 1 + delta)
print(f"理论基态能量: {theoretical_energy:.6f}")

# ==============================
# 2. 预定义Clifford架构集合
# ==============================
def get_clifford_architectures(n_qubits: int) -> List[Circuit]:
    architectures = []
    circ1 = Circuit()
    circ1 += CNOT.on(0, 1)
    circ1 += H.on(0)
    circ1 += S.on(1)
    architectures.append(circ1)
    circ2 = Circuit()
    circ2 += CNOT.on(1, 0)
    circ2 += S.on(0)
    circ2 += H.on(1)
    architectures.append(circ2)
    circ3 = Circuit()
    circ3 += H.on(0)
    circ3 += H.on(1)
    circ3 += CNOT.on(0, 1)
    circ3 += S.on(0)
    circ3 += S.on(1)
    architectures.append(circ3)
    return architectures

clifford_architectures = get_clifford_architectures(n_qubits)

# ==============================
# 3. 核心函数：计算变换后的哈密顿量
# ==============================
def get_transformed_hamiltonian(arch_index: int, phi_params: np.ndarray) -> Hamiltonian:
    circ = Circuit()
    for i in range(n_qubits):
        ry_angle = np.pi/2 * np.round(phi_params[2*i] / (np.pi/2))
        rz_angle = np.pi/2 * np.round(phi_params[2*i+1] / (np.pi/2))
        circ += RY(ry_angle).on(i)
        circ += RZ(rz_angle).on(i)
    circ += clifford_architectures[arch_index]
    sim = Simulator('mqvector', n_qubits)
    T_matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for i in range(2**n_qubits):
        initial_state = np.zeros(2**n_qubits, dtype=complex)
        initial_state[i] = 1.0
        sim.set_qs(initial_state)
        sim.apply_circuit(circ)
        T_matrix[:, i] = sim.get_qs()
    sim = Simulator('mqvector', n_qubits)
    ham_matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for i in range(2**n_qubits):
        initial_state_i = np.zeros(2**n_qubits, dtype=complex)
        initial_state_i[i] = 1.0
        sim.set_qs(initial_state_i)
        sim.apply_hamiltonian(hamiltonian)
        result_i = sim.get_qs()
        ham_matrix[:, i] = result_i
    T_dag_matrix = T_matrix.conj().T
    H_T_matrix = T_dag_matrix @ ham_matrix @ T_matrix
    H_T_matrix = 0.5 * (H_T_matrix + H_T_matrix.conj().T)

    h_t_qo = QubitOperator()
    dim = 2**n_qubits
    for i in range(dim):
        for j in range(dim):
            coeff = H_T_matrix[i, j]
            if abs(coeff) > 1e-6:
                term_str = ''
                for k in range(n_qubits):
                    i_bit = (i >> k) & 1
                    j_bit = (j >> k) & 1
                    if i_bit == 0 and j_bit == 0:
                        term_str += f'I{k} '
                    elif i_bit == 0 and j_bit == 1:
                        term_str += f'X{k} '
                    elif i_bit == 1 and j_bit == 0:
                        term_str += f'Y{k} '
                    elif i_bit == 1 and j_bit == 1:
                        term_str += f'Z{k} '
                h_t_qo += QubitOperator(term_str.strip(), coeff)
        return Hamiltonian(h_t_qo)

# ==============================
# 4. 更强的ansatz线路与能量计算
# ==============================
def ansatz_circuit():
    circ = Circuit()
    for i in range(2):
        circ += RY(f'theta_{i}').on(i)
        circ += RZ(f'theta_{2+i}').on(i)
    circ += CNOT.on(0, 1)
    for i in range(2):
        circ += RY(f'theta_{4+i}').on(i)
        circ += RZ(f'theta_{6+i}').on(i)
    return circ

def compute_energy(theta_params, arch_index, phi_params):
    h_t = get_transformed_hamiltonian(arch_index, phi_params)
    u_circ = ansatz_circuit()
    sim = Simulator('mqvector', n_qubits)
    grad_ops = sim.get_expectation_with_grad(h_t, u_circ)
    energy, _ = grad_ops(theta_params)
    return float(energy.real)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ==============================
# 5. 概率化架构搜索算法（增强版）
# ==============================
def probabilistic_architecture_search():
    theta = np.random.uniform(0, 2*np.pi, 8)
    phi = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=2*n_qubits)
    alpha = np.ones(num_architectures)

    energy_history = []
    best_energy = float('inf')
    best_params = None

    print("开始训练概率化架构搜索算法...")

    for epoch in range(max_epochs):
        probs = softmax(alpha)
        arch_idx = np.random.choice(num_architectures, p=probs)

        current_energy = compute_energy(theta, arch_idx, phi)

        eps = 0.01
        grad_theta = np.zeros_like(theta)
        grad_success = True
        for i in range(len(theta)):
            try:
                theta_plus = theta.copy()
                theta_plus[i] += eps
                energy_plus = compute_energy(theta_plus, arch_idx, phi)

                theta_minus = theta.copy()
                theta_minus[i] -= eps
                energy_minus = compute_energy(theta_minus, arch_idx, phi)

                grad_theta[i] = (energy_plus - energy_minus) / (2 * eps)
            except Exception as e:
                print(f"梯度计算出错: {e}")
                grad_success = False
                break

        if grad_success:
            theta -= learning_rate * grad_theta

        if current_energy < best_energy:
            alpha[arch_idx] += 0.5
            best_energy = current_energy
            best_params = (theta.copy(), phi.copy(), alpha.copy())
            print(f"Epoch {epoch+1}: 发现新最佳能量 {float(best_energy):.6f}")
        else:
            alpha[arch_idx] = max(0.1, alpha[arch_idx] - 0.1)

        alpha = np.maximum(alpha, 0.1)
        energy_history.append(current_energy)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: 当前能量 = {float(current_energy):.6f}, 最佳能量 = {float(best_energy):.6f}")
            print(f"架构softmax概率: {probs}")

        if epoch > 20 and float(best_energy) < theoretical_energy + 0.05:
            print(f"收敛于Epoch {epoch+1} (接近理论值)")
            break
        if epoch > 50 and abs(np.mean(energy_history[-10:]) - best_energy) < tol:
            print(f"收敛于Epoch {epoch+1}")
            break

    if best_params is not None:
        theta, phi, alpha = best_params
    else:
        print("警告: 未找到最佳参数，使用最终参数")

    return theta, phi, alpha, energy_history

# ==============================
# 6. 多次trial选最优
# ==============================
n_trials = 5
best_energy = float('inf')
for trial in range(n_trials):
    print(f"\n===== 第{trial+1}次随机初始化 =====")
    theta, phi, alpha, energy_history = probabilistic_architecture_search()
    min_energy = min(energy_history)
    if min_energy < best_energy:
        best_energy = min_energy
        best_theta, best_phi, best_alpha, best_history = theta, phi, alpha, energy_history

# ==============================
# 7. 输出结果与绘图
# ==============================
print("\n" + "="*50)
print("概率化架构搜索最终结果:")
print(f"最优能量: {float(min(best_history)):.6f}")
print(f"理论基态能量: {float(theoretical_energy):.6f}")
print(f"薛定谔线路参数θ: {[f'{float(x):.4f}' for x in best_theta]}")
print(f"海森堡线路参数φ: {[f'{float(x):.4f}' for x in best_phi]}")
print(f"架构概率分布: {best_alpha/np.sum(best_alpha)}")
print("="*50 + "\n")


# 画图
def moving_average(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode='valid')

plt.figure(figsize=(10, 6))
plt.plot(best_history, label="Current energy")
plt.plot(range(2, len(best_history)-2), moving_average(best_history, 5), label="Smoothed energy", color='orange')
plt.axhline(y=float(min(best_history)), color='r', linestyle='-', label="Optimal energy")
plt.axhline(y=float(theoretical_energy), color='g', linestyle='--', label="Theoretical energy")
plt.xlabel("Steps")
plt.ylabel("Energy")
plt.title("Probabilistic Architecture Search Convergence")
plt.legend()
plt.grid(True)
plt.savefig('sh_vqae_pas_convergence_optimized.png')
plt.show()



print("验证各架构的最终能量:")
for i in range(num_architectures):
    energy = compute_energy(best_theta, i, best_phi)
    print(f"架构 {i+1}: 能量 = {float(energy):.6f} (概率权重: {(best_alpha[i]/np.sum(best_alpha)):.4f})")

# 验证理论基态能量（可选）
if __name__ == "__main__":
    import scipy
    Z = np.array([[1,0],[0,-1]])
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    H = np.kron(Z,Z) + np.kron(X,X) + np.kron(Y,Y)
    eigvals = np.linalg.eigvalsh(H)
    print("理论哈密顿量本征值:", eigvals)