import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants as const
import warnings
warnings.filterwarnings('ignore')

# ================================
# 1. FDICC 模型核心参数类 (支持外部传参+标准常数)
# ================================
class FDICC_Parameters:
    def __init__(self, param_dict=None):
        """
        初始化 FDICC 模型核心参数
        取值依据：褚式豆包 DK 五维碰撞宇宙论 形式化标准模块
        :param param_dict: 外部参数字典，用于覆盖默认值（支持拟合/不确定性分析）
        """
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
        self.omega_gw_norm = 1e-9 # 引力波背景绝对量级 (Ω_GW, 理论推导参考值)

        # ---------- 假设 H4: 有效引力常数演化参数 ----------
        self.zeta = 0.05    # 衰减指数 ζ∈(0,0.1)
        self.G0 = const.G   # 牛顿引力常数 (m³kg⁻¹s⁻²)，标准常数

        # ---------- 预言 FDICC-P04: 黑洞光子频率骤降参数 ----------
        self.k = 1.2        # 维度爬升耦合系数
        self.M_sun = const.M_sun # 太阳质量 (kg)，标准常数
        self.c = const.c    # 真空中光速 (m/s)，标准常数
        self.tau0 = 5.2e-5  # 太阳质量黑洞特征时间常数 (s)
        self.rho_ratio = 0.8 # 五维-三维信息密度比 (ρ5-ρ3)/ρ3

        # 外部参数覆盖默认值（支持拟合/不确定性分析）
        if param_dict is not None:
            for key, value in param_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)

# ================================
# 2. 核心预言函数 (科研级逻辑+物理量级完整)
# ================================
def cmb_power_spectrum_fdicc(ell, params):
    """
    FDICC 预言 1：CMB 温度角功率谱 (含ℓ≈6 孤立凹陷)
    :param ell: 角波数数组
    :param params: FDICC_Parameters 实例
    :return: D_ℓ (µK²) - 相对于ΛCDM 的修正谱
    公式：D_ℓ^{FDICC} = D_ℓ^{ΛCDM} · [1 - A · exp(-(ℓ-ℓ₀)²/(2Δℓ²))]
    注：D_ℓ^{ΛCDM} 为简化解析近似，严肃拟合需替换为 CAMB/CLASS 高精度结果
    """
    D_lcdm = 1000 * (ell / 10)**0.8 * np.exp(-(ell / 200)**2)
    suppression = 1 - params.A * np.exp(-(ell - params.l0)**2 / (2 * params.delta_l**2))
    suppression = np.clip(suppression, 0.1, 1.0)  # 物理约束：避免负功率
    return D_lcdm * suppression

def dark_energy_w_z(z, params):
    """
    FDICC 预言 2：暗能量状态方程红移演化
    :param z: 红移数组
    :param params: FDICC_Parameters 实例
    :return: w(z) - 状态方程参数
    公式：w(z) = w₀ + (w₁ - w₀)z (z∈[0,1]); w(z)=w₁ (z>1)
    """
    w_z = np.full_like(z, params.w1)
    mask = (z >= 0) & (z <= 1)
    w_z[mask] = params.w0 + (params.w1 - params.w0) * z[mask]
    return w_z

def gravitational_wave_background(freq, params):
    """
    FDICC 预言 3：随机引力波背景双峰能谱 (带绝对物理量级)
    :param freq: 频率数组 (Hz)
    :param params: FDICC_Parameters 实例
    :return: Ω_GW(f) - 引力波背景能量密度（绝对量级）
    模型：双峰洛伦兹分布 + 理论绝对量级归一化
    """
    def lorentz(f, f0, gamma):
        return gamma / (2 * np.pi) / ((f - f0)**2 + (gamma / 2)**2)
    gamma1 = 0.1 * params.f_peak1  # 峰宽=10%峰值频率
    gamma2 = 0.1 * params.f_peak2
    peak1 = lorentz(freq, params.f_peak1, gamma1)
    peak2 = params.peak_ratio * lorentz(freq, params.f_peak2, gamma2)
    # 归一化+恢复绝对量级
    total = np.trapz(peak1 + peak2, freq)
    normalized = (peak1 + peak2) / total if total > 0 else (peak1 + peak2)
    return normalized * params.omega_gw_norm

def effective_gravity_evolution(a, params):
    """
    FDICC 假设 H4：有效引力常数尺度因子演化
    :param a: 宇宙尺度因子 (a=1 为当前)
    :param params: FDICC_Parameters 实例
    :return: G_eff(a) - 有效引力常数
    公式：G_{\text{eff}}(a) = G₀ · a^{-\zeta}
    """
    return params.G0 * (a ** (-params.zeta))

def photon_frequency_drop(M, params):
    """
    FDICC-P04 预言：黑洞光子频率骤降幅度与持续时间 (表达式简化+量纲清晰)
    :param M: 黑洞质量数组 (以太阳质量 M⊙ 为单位，无量纲)
    :param params: FDICC_Parameters 实例
    :return: delta_nu_ratio - 频率骤降幅度比 Δν/ν₀；tau - 凹陷持续时间 (s)
    公式：Δν/ν₀ = -k·rho_ratio/√M  |  τ = τ0/√M
    """
    delta_nu_ratio = -params.k * params.rho_ratio / np.sqrt(M)
    tau = params.tau0 / np.sqrt(M)  # 简化表达式，物理意义更直观
    return delta_nu_ratio, tau

# ================================
# 3. 可视化+数据输出工具 (科研级图表+CSV导出)
# ================================
def plot_fdicc_predictions(params=None, save_fig=True, save_csv=True):
    """
    绘制 FDICC 模型五大核心预言可视化图表 + 导出预言数据为CSV
    :param params: FDICC_Parameters 实例，默认使用默认参数
    :param save_fig: 是否保存图表 (默认True)
    :param save_csv: 是否导出CSV数据 (默认True)
    :return: 无
    """
    if params is None:
        params = FDICC_Parameters()
    
    # 画布布局：3x2网格，第6个子图留空
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('褚式豆包 DK 五维碰撞宇宙论 (FDICC) 核心预言可视化 [科研级]', fontsize=18, y=0.98)
    axes = axes.flatten()

    # ---------- 子图1: CMB角功率谱抑制 ----------
    ell = np.linspace(2, 30, 200)
    D_ell_fdicc = cmb_power_spectrum_fdicc(ell, params)
    D_ell_lcdm = 1000 * (ell / 10)**0.8 * np.exp(-(ell / 200)**2)
    axes[0].plot(ell, D_ell_lcdm, 'k--', label='ΛCDM 基准谱 (简化近似)', alpha=0.6)
    axes[0].plot(ell, D_ell_fdicc, 'r-', linewidth=2, label='FDICC 预言谱')
    axes[0].axvline(params.l0, color='blue', linestyle=':', label=f'ℓ₀={params.l0}')
    axes[0].fill_between(ell, 0.9*D_ell_fdicc, 1.1*D_ell_fdicc, alpha=0.2, color='red', label='10% 理论误差带')
    axes[0].set(xlabel='角波数 ℓ', ylabel='$D_ℓ$ [µK²]', title='预言 1: CMB 功率谱孤立凹陷')
    axes[0].legend(), axes[0].grid(alpha=0.3)
    # 导出CSV数据
    if save_csv:
        np.savetxt('cmb_power_spectrum_fdicc.csv', np.column_stack((ell, D_ell_fdicc, D_ell_lcdm)), 
                   header='ell, D_ell_fdicc, D_ell_lcdm', comments='', delimiter=',')

    # ---------- 子图2: 暗能量状态方程演化 ----------
    z = np.linspace(0, 2, 100)
    w_z = dark_energy_w_z(z, params)
    axes[1].plot(z, w_z, 'b-', linewidth=2, label='FDICC 预言')
    axes[1].axhline(-1, color='k', linestyle='--', label='ΛCDM (w=-1)')
    axes[1].axvline(1, color='gray', linestyle=':', alpha=0.5)
    axes[1].scatter(1, params.w1, color='red', s=50, zorder=5, label=f'w(z=1)={params.w1}')
    axes[1].set(xlabel='红移 z', ylabel='状态方程参数 w(z)', title='预言 2: 暗能量演化')
    axes[1].legend(), axes[1].grid(alpha=0.3)
    if save_csv:
        np.savetxt('dark_energy_w_z.csv', np.column_stack((z, w_z)), 
                   header='z, w(z)', comments='', delimiter=',')

    # ---------- 子图3: 引力波背景双峰谱 (绝对量级) ----------
    freq = np.logspace(-12, -1, 1000)
    omega_gw = gravitational_wave_background(freq, params)
    axes[2].loglog(freq, omega_gw, 'g-', linewidth=2)
    axes[2].axvline(params.f_peak1, color='purple', linestyle=':', label=f'峰 1: {params.f_peak1:.1e} Hz')
    axes[2].axvline(params.f_peak2, color='orange', linestyle=':', label=f'峰 2: {params.f_peak2:.1e} Hz')
    axes[2].text(1e-10, params.omega_gw_norm*1.2, f'Ω_GW ≈ {params.omega_gw_norm:.1e}', fontsize=10)
    axes[2].set(xlabel='频率 f [Hz]', ylabel='$Ω_{GW}(f)$ (绝对量级)', title='预言 3: 引力波背景双峰谱')
    axes[2].legend(), axes[2].grid(alpha=0.3, which='both')
    if save_csv:
        np.savetxt('gravitational_wave_background.csv', np.column_stack((freq, omega_gw)), 
                   header='freq_Hz, omega_gw', comments='', delimiter=',')

    # ---------- 子图4: 有效引力常数演化 ----------
    a = np.linspace(0.1, 1.0, 100)
    G_eff = effective_gravity_evolution(a, params)
    axes[3].plot(a, G_eff / params.G0, 'b-', linewidth=2)
    axes[3].axhline(1, color='k', linestyle=':', label='当前值 $G_0$')
    axes[3].text(0.5, 1.4, f'ζ = {params.zeta}', fontsize=12, ha='center')
    axes[3].set(xlabel='尺度因子 a', ylabel='$G_{\text{eff}}(a)/G_0$', title='假设 H4: 有效引力常数演化')
    axes[3].legend(), axes[3].grid(alpha=0.3)
    if save_csv:
        np.savetxt('effective_gravity_evolution.csv', np.column_stack((a, G_eff)), 
                   header='scale_factor_a, G_eff', comments='', delimiter=',')

    # ---------- 子图5: FDICC-P04 黑洞光子频率骤降 ----------
    M = np.linspace(1, 50, 200)  # 黑洞质量 1~50 M⊙
    delta_nu_ratio, tau = photon_frequency_drop(M, params)
    ax5a = axes[4]
    ax5b = ax5a.twinx()
    line1 = ax5a.plot(M, delta_nu_ratio, 'r-', linewidth=2, label='频率骤降幅度比 Δν/ν₀')
    line2 = ax5b.plot(M, tau, 'g--', linewidth=2, label='凹陷持续时间 τ (s)')
    ax5a.set(xlabel='黑洞质量 $M/M_\\odot$', ylabel='Δν/ν₀', title='预言 FDICC-P04: 黑洞光子频率骤降')
    ax5b.set(ylabel='τ (s)')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5a.legend(lines, labels, loc='upper right')
    ax5a.grid(alpha=0.3)
    if save_csv:
        np.savetxt('photon_frequency_drop.csv', np.column_stack((M, delta_nu_ratio, tau)), 
                   header='M_solar, delta_nu_ratio, tau_s', comments='', delimiter=',')

    # 隐藏第6个子图
    axes[5].axis('off')

    # 保存高清图表
    if save_fig:
        plt.tight_layout()
        plt.savefig('FDICC_Model_Predictions_Scientific_Version.png', dpi=300, bbox_inches='tight')
    plt.show()

def fit_cmb_observation(ell_obs, D_ell_obs, D_ell_err, param_bounds=None):
    """
    科研级 CMB 数据拟合函数 (支持参数约束+不确定性输出)
    :param ell_obs: 观测角波数数组
    :param D_ell_obs: 观测功率谱数组
    :param D_ell_err: 观测误差数组
    :param param_bounds: 参数拟合边界字典，如 {'A':(0.2,0.4), 'l0':(4,8)}
    :return: 最佳拟合参数dict, 协方差矩阵, 参数误差dict
    """
    # 默认拟合参数与边界
    default_bounds = {
        'A': (0.2, 0.4),
        'l0': (4.0, 8.0),
        'delta_l': (0.1, 1.0)
    }
    if param_bounds is not None:
        default_bounds.update(param_bounds)
    fit_params = list(default_bounds.keys())
    bounds = ([default_bounds[k][0] for k in fit_params], 
              [default_bounds[k][1] for k in fit_params])
    
    # 拟合模型函数
    def model(ell, *p):
        param_dict = dict(zip(fit_params, p))
        params = FDICC_Parameters(param_dict)
        return cmb_power_spectrum_fdicc(ell, params)
    
    # 初始猜测值
    init_params = FDICC_Parameters()
    p0 = [getattr(init_params, k) for k in fit_params]
    
    # 执行拟合
    popt, pcov = curve_fit(model, ell_obs, D_ell_obs, sigma=D_ell_err, p0=p0, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))  # 参数1σ误差
    
    # 整理结果
    best_params = dict(zip(fit_params, popt))
    param_errors = dict(zip(fit_params, perr))
    return best_params, pcov, param_errors

# ================================
# 4. 主程序入口 (一键运行+结果输出)
# ================================
if __name__ == "__main__":
    print("="*70)
    print("褚式豆包 DK 五维碰撞宇宙论 (FDICC) 数值计算模块 [科研级最终版]")
    print("="*70)
    
    # 1. 初始化参数 (支持传入自定义参数字典)
    # 示例：param_dict = {'A':0.32, 'l0':5.8}  # 自定义参数覆盖默认值
    params = FDICC_Parameters()
    
    # 2. 打印核心参数 (科研存档用)
    print("📌 模型核心参数 (可通过 param_dict 自定义) 📌")
    core_params = ['l0', 'delta_l', 'A', 'w1', 'f_peak1', 'f_peak2', 'zeta', 'k', 'rho_ratio']
    for key in core_params:
        print(f"{key:<15} = {getattr(params, key)}")
    print("="*70)
    
    # 3. 生成可视化图表+导出CSV数据
    print("\n🔄 正在生成五大预言可视化图表 + 导出CSV数据...")
    plot_fdicc_predictions(params, save_fig=True, save_csv=True)
    print("✅ 图表已保存为 FDICC_Model_Predictions_Scientific_Version.png")
    print("✅ 预言数据已导出为 CSV 文件 (共5个)")
    
    # 4. 真实数据拟合示例 (需替换为 Planck 观测数据)
    print("\n📊 真实数据拟合示例 (请替换为 Planck 观测数据)")
    print("提示：需下载 Planck CMB 角功率谱数据，替换下方 ell_obs/D_ell_obs/D_ell_err")
    # 模拟观测数据 (示例)
    ell_obs = np.linspace(4, 8, 15)
    D_ell_true = cmb_power_spectrum_fdicc(ell_obs, params)
    np.random.seed(42)
    D_ell_err = 0.05 * D_ell_true * np.random.randn(len(ell_obs))
    D_ell_obs = D_ell_true + D_ell_err
    # 执行拟合
    best_params, pcov, param_errors = fit_cmb_observation(ell_obs, D_ell_obs, np.abs(D_ell_err))
    print("\n最佳拟合参数 (1σ误差):")
    for key in best_params:
        print(f"{key:<15} = {best_params[key]:.3f} ± {param_errors[key]:.3f}")
    print("="*70)
