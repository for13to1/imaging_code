"""
Debevec 1997 HDR 算法实现

包含论文中的核心部分：
1. 相机响应函数校准 (Section 2.1)
2. 辐射度图恢复 (Section 2.2)
3. 虚拟摄影 (Section 2.7)
4. RGBE 格式存储 (Section 2.2.1)
5. 颜色通道平衡 (Section 2.6)
6. 绝对辐射度校准 (Section 2.5)

参考论文:
Debevec, P.E., & Malik, J. (1997). Recovering high dynamic range radiance maps from photographs.
"""

import struct
import json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2


def triangle_weights(length: int = 256) -> np.ndarray:
    """
    三角形权重函数（帽形函数）

    论文 Section 2 (Equation 4): 对接近饱和的像素赋予低权重，
    中间色调像素赋予高权重。
    """
    assert length >= 2, "length必须大于等于2"

    w = np.zeros(length, dtype=np.float32)
    half = length // 2
    max_val = length - 1
    epsilon = 1e-6

    for i in range(length):
        w[i] = float(min(i, max_val - i))

    # 确保两端为极小值而非完全为0以便数值稳定（可选）
    # 但严格对称的帐篷函数在 0 和 255 处应为 0
    # 我们这里保留一个微小的 epsilon 以避免权重全为 0 的极端情况
    w = np.maximum(w, epsilon)

    return w


def sample_pixel_locations(
    rows: int, cols: int, samples: int, random: bool = False, seed: Optional[int] = None
) -> List[Tuple[int, int]]:
    """
    采样像素位置

    论文 Section 2.1: 均匀采样或随机采样像素位置用于校准。
    像素应该在空间上均匀分布，且亮度分布均匀。
    """
    points = []

    if random:
        if seed is not None:
            np.random.seed(seed)
        for _ in range(samples):
            x = np.random.randint(0, cols)
            y = np.random.randint(0, rows)
            points.append((y, x))
    else:
        x_points = int(np.sqrt(samples * cols / rows))
        y_points = samples // x_points
        x_step = cols / x_points
        y_step = rows / y_points
        for i in range(y_points):
            for j in range(x_points):
                y = int(i * y_step + y_step / 2)
                x = int(j * x_step + x_step / 2)
                y = min(y, rows - 1)
                x = min(x, cols - 1)
                points.append((y, x))

    return points


def calibrate_debevec(
    images: List[np.ndarray],
    times: np.ndarray,
    samples: int = 70,
    lambda_smooth: float = 100.0,
    random_sampling: bool = False,
    seed: Optional[int] = None,
    ldr_size: int = 256,
) -> np.ndarray:
    """
    Debevec 相机响应函数校准

    论文 Section 2.1: 使用最小二乘法求解相机响应函数 g，使得：
        g(Z_ij) = ln(E_i) + ln(t_j)

    其中：
    - Z_ij: 第 i 个像素在第 j 张图像中的像素值
    - E_i: 第 i 个像素的场景辐射度
    - t_j: 第 j 张图像的曝光时间
    - g: 相机响应函数（对数域）

    返回指数域的响应函数 f(z) = exp(g(z))。
    """
    images = [np.asarray(img) for img in images]
    times = np.asarray(times, dtype=np.float32)

    assert len(images) == len(times), "图像数量必须等于曝光时间数量"

    if len(images[0].shape) == 3:
        rows, cols, channels = images[0].shape
    else:
        rows, cols = images[0].shape
        channels = 1
        images = [img[:, :, np.newaxis] for img in images]

    weights = triangle_weights(ldr_size)
    points = sample_pixel_locations(rows, cols, samples, random_sampling, seed)

    response_functions = []
    n_points = len(points)
    n_images = len(images)
    log_times = np.log(times)

    for ch in range(channels):
        n_data_eq = n_points * n_images
        n_smooth_eq = ldr_size - 2
        n_constraint_eq = 1
        n_total_eq = n_data_eq + n_smooth_eq + n_constraint_eq

        A = np.zeros((n_total_eq, ldr_size + n_points), dtype=np.float32)
        B = np.zeros(n_total_eq, dtype=np.float32)

        row_idx = 0
        for i, (y, x) in enumerate(points):
            for j, img in enumerate(images):
                val = int(img[y, x, ch])
                wij = float(weights[val])

                A[row_idx, val] = wij
                A[row_idx, ldr_size + i] = -wij
                B[row_idx] = wij * log_times[j]
                row_idx += 1

        A[row_idx, ldr_size // 2] = 1.0
        B[row_idx] = 0.0
        row_idx += 1

        for i in range(n_smooth_eq):
            wi = lambda_smooth * weights[i + 1]
            A[row_idx, i] = wi
            A[row_idx, i + 1] = -2 * wi
            A[row_idx, i + 2] = wi
            row_idx += 1

        solution = np.linalg.lstsq(A, B, rcond=None)[0]
        g = solution[:ldr_size]
        response_functions.append(g)

    response = np.stack(response_functions, axis=1)
    response = np.exp(response)

    return response


def load_image_series(directory: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    从指定目录及其 image_list.txt 加载曝光序列图像和曝光时间

    参数:
        directory: 包含图像和 image_list.txt 的目录路径

    返回:
        (images, times): 图像列表 (RGB) 和曝光时间 NumPy 数组
    """

    dir_path = Path(directory)
    list_path = dir_path / "image_list.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"在目录 {directory} 中未找到 image_list.txt")

    images = []
    times = []

    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过注释行和空行
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            # 如果这一行只有一个数字，那可能是 Number of Images，跳过它
            if len(parts) < 2:
                continue

            filename, time_val = parts[0], parts[1]
            img_path = str(dir_path / filename)

            # 使用 OpenCV 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue

            # OpenCV 默认读取为 BGR，转换为 RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img_rgb)
            # 根据表头 "1/shutter_speed" 说明，该列数值是曝光时间的倒数 (N)
            # 因此，实际曝光时间 delta_t = 1.0 / N
            times.append(1.0 / float(time_val))

    return images, np.array(times, dtype=np.float32)


def save_response(
    filename: str, response: np.ndarray, metadata: Optional[dict] = None
) -> None:
    """
    保存相机响应函数到 JSON 文件

    参数:
        filename: 输出文件名 (.json)
        response: 响应函数数组 (256, channels)
        metadata: 可选的元数据字典
    """
    data = {
        "response": response.tolist(),
        "channels": response.shape[1] if len(response.shape) > 1 else 1,
        "metadata": metadata or {},
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_response(filename: str) -> Tuple[np.ndarray, dict]:
    """
    从 JSON 文件加载相机响应函数

    返回:
        (response_array, metadata)
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    response = np.array(data["response"], dtype=np.float32)
    metadata = data.get("metadata", {})

    return response, metadata


def plot_response(
    response: np.ndarray,
    title: str = "Camera Response Function",
    log_scale: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    绘制相机响应函数曲线

    参数:
        response: 响应函数数组
        title: 图表标题
        log_scale: 是否使用对数坐标轴显示曝光量 (经典 H&D 曲线形态)
        save_path: 如果提供，将图表保存到此路径
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    ldr_size = response.shape[0]
    z = np.arange(ldr_size)

    colors = ["r", "g", "b"]
    labels = ["Red Channel", "Green Channel", "Blue Channel"]

    channels = response.shape[1] if len(response.shape) > 1 else 1

    for i in range(channels):
        channel_resp = response[:, i] if channels > 1 else response.flatten()
        color = colors[i] if channels == 3 else "k"
        label = labels[i] if channels == 3 else "Luminance"

        if log_scale:
            plt.plot(
                z, np.log(channel_resp + 1e-8), color=color, label=label, linewidth=2
            )
            plt.ylabel("Log Exposure ln(E*Δt)")
        else:
            plt.plot(z, channel_resp, color=color, label=label, linewidth=2)
            plt.ylabel("Relative Radiance / Exposure")

    plt.title(title)
    plt.xlabel("Pixel Value (Z)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()

    plt.tight_layout()  # 使布局更加紧凑

    if save_path:
        # 增加 dpi 选项使保存的图片更清晰
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")

    # 静默保存，关闭画布
    plt.close()


def recover_radiance_map(
    images: List[np.ndarray],
    times: np.ndarray,
    response: np.ndarray,
    weights: Optional[np.ndarray] = None,
    ldr_size: int = 256,
) -> np.ndarray:
    """
    恢复场景辐射度图（HDR图像）

    论文 Section 2.2 (Equation 6): 使用相机响应函数恢复场景的辐射度。

    ln(E_i) = (Σ w(Z_ij) * [g(Z_ij) - ln(t_j)]) / (Σ w(Z_ij))

    其中 g(Z_ij) = ln(response[Z_ij])。
    """
    if weights is None:
        weights = triangle_weights(ldr_size).flatten()

    images = [np.asarray(img) for img in images]
    times = np.asarray(times, dtype=np.float32)

    if len(images[0].shape) == 2:
        rows, cols = images[0].shape
        channels = 1
        images = [img[:, :, np.newaxis] for img in images]
    else:
        rows, cols, channels = images[0].shape

    radiance = np.zeros((rows, cols, channels), dtype=np.float32)
    weight_sum = np.zeros((rows, cols, channels), dtype=np.float32)

    for img, t in zip(images, times):
        for ch in range(channels):
            pixel_vals = img[:, :, ch]
            w = weights[pixel_vals]
            g_z = np.log(response[pixel_vals, ch])
            rad_estimate = g_z - np.log(t)
            radiance[:, :, ch] += w * rad_estimate
            weight_sum[:, :, ch] += w

    radiance = radiance / (weight_sum + 1e-8)
    radiance = np.exp(radiance)

    return radiance.squeeze() if channels == 1 else radiance


def virtual_photograph(
    radiance_map: np.ndarray,
    response: np.ndarray,
    exposure_time: float,
    ldr_size: int = 256,
) -> np.ndarray:
    """
    虚拟摄影 - 将HDR辐射度图映射回LDR像素值

    论文 Section 2.7: 使用响应函数将辐射度值映射回像素值，
    模拟不同曝光时间的照片。

    根据论文 Equation 1: Z = f(E * Δt)
    其中 f 是响应函数，E 是辐射度，Δt 是曝光时间

    参数:
        radiance_map: HDR辐射度图
        response: 相机响应函数 (指数域)
        exposure_time: 曝光时间
        ldr_size: LDR像素值范围 (默认256 for 8-bit)

    返回:
        模拟的LDR图像，像素值范围 [0, ldr_size-1]
    """
    radiance_map = np.asarray(radiance_map, dtype=np.float32)
    response = np.asarray(response, dtype=np.float32)

    is_grayscale = len(radiance_map.shape) == 2
    if is_grayscale:
        radiance_map = radiance_map[:, :, np.newaxis]

    rows, cols, channels = radiance_map.shape

    # 计算曝光量 X = E * Δt
    exposure = radiance_map * exposure_time

    # 构建从响应值到像素值的查找表
    # response[z] 表示像素值 z 对应的响应值
    # 我们需要找到 response 中最接近 exposure 的像素值
    ldr_image = np.zeros((rows, cols, channels), dtype=np.uint8)

    for ch in range(channels):
        # 对每个像素，找到使 |response[z] - exposure| 最小的 z
        resp_ch = response[:, ch]

        # 将 exposure 限制在响应函数的有效范围内
        min_resp = resp_ch[resp_ch > 0].min() if np.any(resp_ch > 0) else 1e-6
        max_resp = resp_ch.max()

        exp_ch = np.clip(exposure[:, :, ch], min_resp, max_resp)

        # 查找最近的响应值对应的像素值
        for i in range(rows):
            for j in range(cols):
                target = exp_ch[i, j]
                # 找到最接近的响应值
                idx = np.argmin(np.abs(resp_ch - target))
                ldr_image[i, j, ch] = idx

    return ldr_image.squeeze() if is_grayscale else ldr_image


def save_rgbe(filename: str, radiance_map: np.ndarray) -> None:
    """
    将辐射度图保存为RGBE格式 (.hdr文件)

    论文 Section 2.2.1: RGBE格式使用共享指数存储HDR图像，
    每个像素包含3个8位尾数(RGB)和1个8位共享指数。
    这种格式比常规RGB图像只多33%存储空间。

    参数:
        filename: 输出文件路径
        radiance_map: HDR辐射度图，形状为 (H, W, 3) 或 (H, W)
    """
    radiance_map = np.asarray(radiance_map, dtype=np.float32)

    # 确保是彩色图像
    if len(radiance_map.shape) == 2:
        # 灰度图转为RGB
        radiance_map = np.stack([radiance_map] * 3, axis=-1)

    rows, cols, channels = radiance_map.shape
    assert channels == 3, "RGBE格式需要3通道图像"

    # RGBE编码
    # 找到每个像素的最大分量
    max_component = np.max(radiance_map, axis=-1, keepdims=True)

    # RGBE 编码逻辑：
    # 标准定义：v = m * 2^(E-128), 其中 m 属于 [0.5, 1)
    # 对应到整数存储：E = floor(log2(max_v)) + 129
    # 这样可以确保最大分量的整数部分落在 [128, 255] 之间，提供最高精度。
    shared_exp = np.where(
        max_component > 1e-32, np.floor(np.log2(max_component) + 129), 0
    ).astype(np.uint8)

    # 计算比例因子并换算尾数
    # mantissa = radiance / 2^(shared_exp - 129) * 128 / max_component
    # 或者更简单的：mantissa = radiance * (256.0 / 2^(shared_exp - 128))
    scale = np.power(2.0, shared_exp.astype(np.float32) - 128)
    mantissa = np.where(
        shared_exp > 0, np.clip(radiance_map * 256.0 / scale, 0, 255), 0
    ).astype(np.uint8)

    # 写入文件
    filepath = Path(filename)
    with open(filepath, "wb") as f:
        # 写入Radiance HDR文件头
        f.write(b"#?RADIANCE\n")
        f.write(b"# Made with debevec.py\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n")
        f.write(b"\n")
        f.write(f"-Y {rows} +X {cols}\n".encode())

        # 写入像素数据
        for i in range(rows):
            for j in range(cols):
                f.write(
                    struct.pack(
                        "BBBB",
                        mantissa[i, j, 0],  # R
                        mantissa[i, j, 1],  # G
                        mantissa[i, j, 2],  # B
                        shared_exp[i, j, 0],  # E (共享指数)
                    )
                )


def load_rgbe(filename: str) -> np.ndarray:
    """
    从RGBE格式文件加载HDR辐射度图

    参数:
        filename: RGBE文件路径 (.hdr)

    返回:
        HDR辐射度图，形状为 (H, W, 3)
    """
    filepath = Path(filename)

    with open(filepath, "rb") as f:
        # 读取文件头
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "":
                break
            header_lines.append(line)

        # 解析分辨率
        res_line = f.readline().decode("ascii").strip()
        parts = res_line.split()
        rows = int(parts[1])
        cols = int(parts[3])

        # 读取像素数据
        data = f.read()
        n_pixels = rows * cols

        # 解析RGBE数据
        radiance = np.zeros((rows, cols, 3), dtype=np.float32)

        for i in range(rows):
            for j in range(cols):
                idx = (i * cols + j) * 4
                r = data[idx]
                g = data[idx + 1]
                b = data[idx + 2]
                e = data[idx + 3]

                if e == 0:
                    radiance[i, j] = [0, 0, 0]
                else:
                    scale = np.power(2.0, e - 128) / 256.0
                    radiance[i, j] = [r * scale, g * scale, b * scale]

        return radiance


def balance_color_channels(
    radiance_map: np.ndarray,
    reference_color: Optional[np.ndarray] = None,
    mid_value: Optional[float] = None,
) -> np.ndarray:
    """
    颜色通道平衡

    论文 Section 2.6: 调整RGB通道的相对尺度以匹配参考光源颜色，
    解决不同通道有不同未知尺度因子导致的色偏问题。

    默认情况下，算法选择尺度因子使得像素值 Z_mid 的像素具有单位曝光，
    这意味着 RGB = (Z_mid, Z_mid, Z_mid) 的像素是消色差的。

    参数:
        radiance_map: HDR辐射度图，形状为 (H, W, 3)
        reference_color: 参考光源颜色 [R, G, B]，默认使用等比例
        mid_value: 中间像素值，默认使用像素范围的中点

    返回:
        颜色平衡后的辐射度图
    """
    radiance_map = np.asarray(radiance_map, dtype=np.float32)

    if len(radiance_map.shape) != 3 or radiance_map.shape[2] != 3:
        raise ValueError("颜色平衡需要3通道彩色图像")

    if mid_value is None:
        mid_value = 127.5  # 8-bit图像的中点

    if reference_color is None:
        # 默认: 消色差（灰度）参考
        reference_color = np.array([1.0, 1.0, 1.0])
    else:
        reference_color = np.asarray(reference_color, dtype=np.float32)

    # 归一化参考颜色
    reference_color = reference_color / np.sum(reference_color)

    # 计算每个通道的尺度因子
    # 目标是让 (mid_value, mid_value, mid_value) 映射到 reference_color 的比例
    scales = np.zeros(3)
    for ch in range(3):
        # 找到该通道中值为 mid_value 附近的像素的平均辐射度
        channel_data = radiance_map[:, :, ch]
        # 使用中位数作为参考点
        median_rad = np.median(channel_data)
        if median_rad > 0:
            scales[ch] = reference_color[ch] / median_rad
        else:
            scales[ch] = 1.0

    # 应用尺度因子
    balanced = radiance_map * scales[np.newaxis, np.newaxis, :]

    return balanced


def calibrate_absolute_radiance(
    radiance_map: np.ndarray, known_radiance: float, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    绝对辐射度校准

    论文 Section 2.5: 通过拍摄已知亮度的标准光源来确定比例因子，
    将相对辐射度转换为绝对辐射度。

    参数:
        radiance_map: 相对辐射度图
        known_radiance: 标准光源的已知绝对辐射度值
        mask: 标准光源区域的掩码，如果为None则使用整个图像的平均值

    返回:
        绝对辐射度图
    """
    radiance_map = np.asarray(radiance_map, dtype=np.float32)

    if mask is not None:
        # 使用掩码区域的平均辐射度
        mask = np.asarray(mask, dtype=bool)
        if len(radiance_map.shape) == 3:
            # 对彩色图像，取各通道的平均
            current_radiance = np.mean(
                [
                    radiance_map[:, :, ch][mask].mean()
                    for ch in range(radiance_map.shape[2])
                ]
            )
        else:
            current_radiance = radiance_map[mask].mean()
    else:
        # 使用整个图像的平均辐射度
        current_radiance = radiance_map.mean()

    if current_radiance <= 0:
        raise ValueError("当前辐射度必须大于0")

    # 计算比例因子
    scale_factor = known_radiance / current_radiance

    # 应用比例因子
    absolute_radiance = radiance_map * scale_factor

    return absolute_radiance


def merge_scans_with_responses(
    scan1: np.ndarray,
    scan2: np.ndarray,
    response1: np.ndarray,
    response2: np.ndarray,
    weights: Optional[np.ndarray] = None,
    ldr_size: int = 256,
) -> np.ndarray:
    """
    使用两个不同密度设置的扫描响应曲线融合单张底片的两次扫描

    论文 Section 2.4 方法2: 对同一张底片进行两次不同密度设置的扫描，
    然后利用各自的响应曲线进行融合，恢复完整的动态范围。

    参数:
        scan1: 第一次扫描的图像（低密度设置，捕获暗部细节）
        scan2: 第二次扫描的图像（高密度设置，捕获亮部细节）
        response1: 第一次扫描的响应曲线
        response2: 第二次扫描的响应曲线
        weights: 权重函数，默认为三角形权重
        ldr_size: LDR像素值范围

    返回:
        融合后的HDR辐射度图
    """
    if weights is None:
        weights = triangle_weights(ldr_size)

    scan1 = np.asarray(scan1, dtype=np.float32)
    scan2 = np.asarray(scan2, dtype=np.float32)

    is_grayscale = len(scan1.shape) == 2
    if is_grayscale:
        scan1 = scan1[:, :, np.newaxis]
        scan2 = scan2[:, :, np.newaxis]

    rows, cols, channels = scan1.shape

    # 使用各自的响应曲线将像素值转换为辐射度
    radiance1 = np.zeros((rows, cols, channels), dtype=np.float32)
    radiance2 = np.zeros((rows, cols, channels), dtype=np.float32)

    for ch in range(channels):
        pixel_vals1 = scan1[:, :, ch].astype(np.int32)
        pixel_vals2 = scan2[:, :, ch].astype(np.int32)

        # 使用响应曲线查找辐射度
        rad1 = response1[pixel_vals1, ch]
        rad2 = response2[pixel_vals2, ch]

        radiance1[:, :, ch] = rad1
        radiance2[:, :, ch] = rad2

    # 使用权重融合两次扫描
    # scan1 (低密度) 对暗部更好，scan2 (高密度) 对亮部更好
    weight1 = np.zeros((rows, cols, channels), dtype=np.float32)
    weight2 = np.zeros((rows, cols, channels), dtype=np.float32)

    for ch in range(channels):
        weight1[:, :, ch] = weights[scan1[:, :, ch].astype(np.int32)]
        weight2[:, :, ch] = weights[scan2[:, :, ch].astype(np.int32)]

    # 加权平均
    total_weight = weight1 + weight2 + 1e-8
    merged_radiance = (radiance1 * weight1 + radiance2 * weight2) / total_weight

    return merged_radiance.squeeze() if is_grayscale else merged_radiance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debevec 1997 HDR implementation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="memorial",
        help="dataset directory name (default: memorial)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=70,
        help="number of pixels to sample for calibration (default: 70)",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_smooth",
        type=float,
        default=100.0,
        help="smoothness weight (default: 100.0)",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    dataset_path = base_dir / "dataset" / args.dataset
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{args.dataset}.hdr"
    plot_path = output_dir / f"{args.dataset}_crf.png"

    print(f"--- 1. Loading dataset: {args.dataset} ---")
    print(f"Path: {dataset_path}")

    try:
        images, times = load_image_series(str(dataset_path))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    print(
        f"Successfully loaded {len(images)} images, exposure time range: {min(times)}s ~ {max(times)}s"
    )

    # 2. Calibration
    print(
        f"\n--- 2. Computing camera response curve (samples: {args.samples}, Lambda: {args.lambda_smooth})... ---"
    )
    response = calibrate_debevec(
        images, times, samples=args.samples, lambda_smooth=args.lambda_smooth
    )

    # 3. Plot and save
    print(f"\n--- 3. Generating response curve... ---")
    plot_response(
        response, title=f"Recovered CRF - {args.dataset}", save_path=str(plot_path)
    )
    print(f"Response curve saved to: {plot_path}")

    # 4. Synthesizing HDR Radiance Map
    print("\n--- 4. Synthesizing HDR Radiance Map... ---")
    hdr_map = recover_radiance_map(images, times, response)

    # 5. Save
    save_rgbe(str(output_path), hdr_map)
    print(f"✅ Success! Result saved to: {output_path}")
