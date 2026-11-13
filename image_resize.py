from PIL import Image
import os

# 输入文件夹（原图路径）
input_folder = r""
# 输出文件夹（保存结果）
output_folder = r""

# 如果输出文件夹不存在则创建
os.makedirs(output_folder, exist_ok=True)

# resize 的目标尺寸
resize_size = (800, 800)
# 放大倍数
scale_factor = 1

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")

        # resize
        resized = img.resize(resize_size, Image.LANCZOS)

        # 放大
        enlarged = resized.resize(
            (resize_size[0] * scale_factor, resize_size[1] * scale_factor),
            Image.LANCZOS
        )

        # 保存到输出文件夹
        save_path = os.path.join(output_folder, filename)
        enlarged.save(save_path)

        print(f"✅ 已处理: {filename}")

print("全部图像处理完成！")
