import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ================================
# 1. FDICC 模型核心参数类 (物理意义标注强化)
# ================================
class FDICC_Parameters:
    def __init__(self):
        """ 初始化 FDICC 模型核心参数 取值依据：褚式豆包 DK 五维碰撞宇宙论 形式化标准模块 """
        # ---------- 预言 5.1: CMB 角功率谱抑制参数 ----------
        self.l0 = 6.0       # 抑制中心角波数 ℓ₀
        self.delta_l = 0.5  # 抑制半宽度 Δℓ
        self.A = 0.30       # 抑制幅度 (0.25~0.35, 理论预言区间)
        # ---------- 预言 5.2: 暗能量状态方程参数 ----------
        self.w0 = -1.0      # 当前(z=0)状态方程参数 w₀
        self.w1 = -1.12     # z=1 处状态方程参数 w₁ (核心预言值)
        # 演化假设：z∈[0,1] 线性演化；z>1 保持 w₁恒定
        # ---------- 预言 5.3: 引力波背景双峰参数 ----------
        self.f_peak1 = 1e-9 # 低频峰频率 (1 nHz, 五维余波场基频)
        self.f_peak2 = 1e-2 # 高频峰频率 (10 mHz, 三维架共振频率)
        self.peak_ratio = 100 # 高低频峰振幅比 (理论推导值)
        # ---------- 假设 H4: 有效引力常数演化参数 ----------
        self.zeta = 0.05    # 衰减指数 ζ∈(0,0.1)
        self.G0 = 6.67430e-11 # 当前牛顿引力常数 G₀ (m³kg⁻¹s⁻²)
        # ---------- 预言 FDICC-P04: 黑洞光子频率骤降参数 ----------
        self.k = 1.2        # 维度爬升耦合系数
        self.r_s_over_M = 2*self.G0*1.989e30/(3e8)**2 # 史瓦西半径与质量比值 (M⊙单位)
        self.tau0 = 5.2e-5  # 太阳质量黑洞特征时间常数 (s)
        self.rho_ratio = 0.8 # 五维-三维信息密度比 (ρ5-ρ3)/ρ3

# ================================
# 2. 核心预言函数 (严格对接理论公式)
# ================================
def cmb_power_spectrum_fdicc(ell, params):
    """ FDICC 预言 1：CMB 温度角功率谱 (含ℓ≈6 孤立凹陷) 
    :param ell: 角波数数组 
    :param params: FDICC_Parameters 实例 
    :return: D_ℓ (µK²) - 相对于ΛCDM 的修正谱 
    公式：D_ℓ^{FDICC} = D_ℓ^{ΛCDM} · [1 - A · exp(-(ℓ-ℓ₀)²/(2Δℓ²))] """
    D_lcdm = 1000 * (ell / 10)**0.8 * np.exp(-(ell / 200)**2)
    suppression = 1 - params.A * np.exp(-(ell - params.l0)**2 / (2 * params.delta_l**2))
    suppression = np.clip(suppression, 0.1, 1.0) # 物理约束：避免负功率
    return D_lcdm * suppression

def dark_energy_w_z(z, params):
    """ FDICC 预言 2：暗能量状态方程红移演化 
    :param z: 红移数组 
    :param params: FDICC_Parameters 实例 
    :return: w(z) - 状态方程参数 
    公式：w(z) = w₀ + (w₁ - w₀)z (z∈[0,1]); w(z)=w₁ (z>1) """
    w_z = np.full_like(z, params.w1) # z>1 保持 w₁
    mask = (z >= 0) & (z <= 1)
    w_z[mask] = params.w0 + (params.w1 - params.w0) * z[mask]
    return w_z

def gravitational_wave_background(freq, params):
    """ FDICC 预言 3：随机引力波背景双峰能谱 
    :param freq: 频率数组 (Hz) 
    :param params: FDICC_Parameters 实例 
    :return: Ω_GW(f) - 归一化能谱密度 
    模型：双峰洛伦兹分布 Ω_GW ∝ δ(f-f₁) + 100δ(f-f₂) """
    def lorentz(f, f0, gamma):
        return gamma / (2 * np.pi) / ((f - f0)**2 + (gamma / 2)**2)
    gamma1 = 0.1 * params.f_peak1 # 峰宽=10%峰值频率
    gamma2 = 0.1 * params.f_peak2
    peak1 = lorentz(freq, params.f_peak1, gamma1)
    peak2 = params.peak_ratio * lorentz(freq, params.f_peak2, gamma2)
    total = np.trapz(peak1 + peak2, freq)
    return (peak1 + peak2) / total if total > 0 else (peak1 + peak2)

def effective_gravity_evolution(a, params):
    """ FDICC 假设 H4：有效引力常数尺度因子演化 
    :param a: 宇宙尺度因子 (a=1 为当前) 
    :param params: FDICC_Parameters 实例 
    :return: G_eff(a) - 有效引力常数 
    公式：G_{\text{eff}}(a) = G₀ · a^{-\zeta} """
    return params.G0 * (a ** (-params.zeta))

def photon_frequency_drop(M, params):
    """ FDICC-P04 预言：黑洞光子频率骤降幅度与持续时间
    :param M: 黑洞质量数组 (M⊙单位)
    :param params: FDICC_Parameters 实例
    :return: delta_nu_ratio - 频率骤降幅度比 Δν/ν0；tau - 凹陷持续时间 (s)
    公式：Δν/ν0 = -k·(ρ5-ρ3)/ρ3·1/√(M/M⊙)；τ = τ0·√(M⊙/M) """
    delta_nu_ratio = -params.k * params.rho_ratio / np.sqrt(M)
    tau = params.tau0 * np.sqrt(1 / M)
    return delta_nu_ratio, tau

# ================================
# 3. 可视化与数据拟合工具函数
# ================================
def plot_fdicc_predictions():
    """绘制 FDICC 模型五大核心预言/假设的可视化图表"""
    params = FDICC_Parameters()
    # 调整画布为 3x2 网格，第6个子图留空，布局更美观
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('褚式豆包 DK 五维碰撞宇宙论 (FDICC) 核心预言可视化', fontsize=18, y=0.98)
    axes = axes.flatten() # 扁平化数组方便索引

    # 子图 1: CMB 角功率谱抑制
    ell = np.linspace(2, 30, 200)
    D_ell_fdicc = cmb_power_spectrum_fdicc(ell, params)
    D_ell_lcdm = 1000 * (ell / 10)**0.8 * np.exp(-(ell / 200)**2)
    axes[0].plot(ell, D_ell_lcdm, 'k--', label='ΛCDM 基准谱', alpha=0.6)
    axes[0].plot(ell, D_ell_fdicc, 'r-', linewidth=2, label='FDICC 预言谱')
    axes[0].axvline(params.l0, color='blue', linestyle=':', label=f'ℓ₀={params.l0}')
    axes[0].fill_between(ell, 0.9*D_ell_fdicc, 1.1*D_ell_fdicc, alpha=0.2, color='red')
    axes[0].set(xlabel='角波数 ℓ', ylabel='$D_ℓ$ [µK²]', title='预言 1: CMB 功率谱孤立凹陷')
    axes[0].legend(), axes[0].grid(alpha=0.3)

    # 子图 2: 暗能量状态方程演化
    z = np.linspace(0, 2, 100)
    w_z = dark_energy_w_z(z, params)
    axes[1].plot(z, w_z, 'b-', linewidth=2, label='FDICC 预言')
    axes[1].axhline(-1, color='k', linestyle='--', label='ΛCDM (w=-1)')
    axes[1].axvline(1, color='gray', linestyle=':', alpha=0.5)
    axes[1].scatter(1, params.w1, color='red', s=50, zorder=5, label=f'w(z=1)={params.w1}')
    axes[1].set(xlabel='红移 z', ylabel='状态方程参数 w(z)', title='预言 2: 暗能量演化')
    axes[1].legend(), axes[1].grid(alpha=0.3)

    # 子图 3: 引力波背景双峰谱
    freq = np.logspace(-12, -1, 1000)
    omega_gw = gravitational_wave_background(freq, params)
    axes[2].loglog(freq, omega_gw, 'g-', linewidth=2)
    axes[2].axvline(params.f_peak1, color='purple', linestyle=':', label=f'峰 1: {params.f_peak1:.1e} Hz')
    axes[2].axvline(params.f_peak2, color='orange', linestyle=':', label=f'峰 2: {params.f_peak2:.1e} Hz')
    axes[2].set(xlabel='频率 f [Hz]', ylabel='$Ω_{GW}(f)$ (归一化)', title='预言 3: 引力波背景双峰谱')
    axes[2].legend(), axes[2].grid(alpha=0.3, which='both')

    # 子图 4: 有效引力常数演化
    a = np.linspace(0.1, 1.0, 100)
    G_eff = effective_gravity_evolution(a, params)
    axes[3].plot(a, G_eff / params.G0, 'b-', linewidth=2)
    axes[3].axhline(1, color='k', linestyle=':', label='当前值 $G_0$')
    axes[3].text(0.5, 1.4, f'ζ = {params.zeta}', fontsize=12, ha='center')
    axes[3].set(xlabel='尺度因子 a', ylabel='$G_{\text{eff}}(a)/G_0$', title='假设 H4: 有效引力常数演化')
    axes[3].legend(), axes[3].grid(alpha=0.3)

    # 子图 5: FDICC-P04 黑洞光子频率骤降
    M = np.linspace(1, 50, 200) # 黑洞质量 1~50 M⊙
    delta_nu_ratio, tau = photon_frequency_drop(M, params)
    ax5a = axes[4]
    ax5b = ax5a.twinx() # 双y轴
    line1 = ax5a.plot(M, delta_nu_ratio, 'r-', linewidth=2, label='频率骤降幅度比 Δν/ν₀')
    line2 = ax5b.plot(M, tau, 'g--', linewidth=2, label='凹陷持续时间 τ (s)')
    ax5a.set(xlabel='黑洞质量 $M/M_\\odot$', ylabel='Δν/ν₀', title='预言 FDICC-P04: 黑洞光子频率骤降')
    ax5b.set(ylabel='τ (s)')
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5a.legend(lines, labels, loc='upper right')
    ax5a.grid(alpha=0.3)

    # 隐藏第6个子图
    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig('FDICC_Model_Predictions_With_P04_Final.png', dpi=300, bbox_inches='tight')
    plt.show()

def fit_cmb_observation(ell_obs, D_ell_obs, D_ell_err):
    """ 用 FDICC 模型拟合 CMB 观测数据，提取关键参数 
    :param ell_obs: 观测角波数数组 
    :param D_ell_obs: 观测功率谱数组 
    :param D_ell_err: 观测误差数组 
    :return: 拟合参数 [A_fit, l0_fit, delta_l_fit] 及协方差矩阵 """
    params = FDICC_Parameters()
    def model(ell, A, l0, delta_l):
        params.A = A
        params.l0 = l0
        params.delta_l = delta_l
        return cmb_power_spectrum_fdicc(ell, params)
    initial_guess = [params.A, params.l0, params.delta_l]
    bounds = ([0.2, 4, 0.1], [0.4, 8, 1.0]) # 理论约束区间
    popt, pcov = curve_fit(model, ell_obs, D_ell_obs, sigma=D_ell_err, p0=initial_guess, bounds=bounds)
    return popt, pcov

# ================================
# 4. 主程序入口
# ================================
if __name__ == "__main__":
    print("="*60)
    print("褚式豆包 DK 五维碰撞宇宙论 (FDICC) 数值计算模块")
    print("="*60)
    params = FDICC_Parameters()
    print("📌 核心参数默认值 📌")
    print(f"CMB 抑制: ℓ₀={params.l0}, Δℓ={params.delta_l}, A={params.A}")
    print(f"暗能量: w(z=1)={params.w1}")
    print(f"引力波峰: {params.f_peak1:.1e} Hz, {params.f_peak2:.1e} Hz")
    print(f"引力演化: ζ={params.zeta}")
    print(f"FDICC-P04: k={params.k}, τ₀={params.tau0:.1e} s, 信息密度比={params.rho_ratio}")
    print("="*60)
    print("\n正在生成预言可视化图表...")
    plot_fdicc_predictions()
    # 🔍 观测数据拟合示例 - 需替换为真实 Planck 数据
    # ell_obs = np.array([3,4,5,6,7,8,9])
    # D_ell_obs = np.array([...]) # 真实观测值
    # D_ell_err = np.array([...]) # 观测误差
    # popt, pcov = fit_cmb_observation(ell_obs, D_ell_obs, D_ell_err)
    # print(f"\n拟合结果: A={popt[0]:.3f}, ℓ₀={popt[1]:.1f}, Δℓ={popt[2]:.2f}")
