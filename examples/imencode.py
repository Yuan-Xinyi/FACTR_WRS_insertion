import cv2
import numpy as np
import matplotlib.pyplot as plt

def encode_image(image, quality=80):
    """
    将图像编码为JPEG格式的byte数组 OpenCV格式
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]  # 可调压缩质量
    success, encoded = cv2.imencode('.jpg', image, encode_param)
    if not success:
        raise ValueError("图像编码失败")
    return encoded

def decode_image(encoded):
    """
    从 JPEG 字节流解码回图像
    """
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

def visualize_original_and_decoded(original_bgr, decoded_bgr):
    """
    显示原始图像与解码图像 使用 matplotlib
    """
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(decoded_rgb)
    plt.title("Decoded Image (JPEG)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# === 示例用法 ===
if __name__ == '__main__':
    # 创建一张测试图像（也可以用 cv2.imread("xxx.jpg") 替代）
    img = np.full((256, 256, 3), 200, dtype=np.uint8)
    cv2.putText(img, "Test Image", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 编码成 JPEG 格式
    encoded = encode_image(img, quality=100)

    # 存为文件（验证编码是否成功）
    with open('test_output.jpg', 'wb') as f:
        f.write(encoded.tobytes())

    # 解码
    decoded = decode_image(encoded)

    # 显示原始图像和解码图像
    visualize_original_and_decoded(img, decoded)
