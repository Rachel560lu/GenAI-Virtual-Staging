# -*- coding: utf-8 -*-
"""有mask版

input: empty_room.png (没有家具的房间), crude_image.png (有家具的房间草图， 由stage2生成)
output: furnished_room.png (渲染后的房间), edge_map.png (边缘图), furnished_room_harmonized.png (优化渲染后的房间)
"""

import torch
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from PIL import Image
import cv2
import numpy as np

# -----------------------------
# 1. 加载 ControlNet 模型
# -----------------------------
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)

# -----------------------------
# 2. 加载 Stable Diffusion ControlNet Inpaint Pipeline
# -----------------------------
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe.to("cuda")  # GPU

# -----------------------------
# 3. 读取房间图片 + 自动生成家具 mask
# -----------------------------
from PIL import Image
import numpy as np

def load_and_resize(path, max_side=768):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

# 空房间（没有沙发桌子）
empty_path = "stage3_room rendering/inputs/empty_room.png"          
empty_img_pil = load_and_resize(empty_path, max_side=768)

# 已摆好家具的房间 (crued 图)
room_path = "stage3_room rendering/inputs/crude_image.png"     
room_img_pil = load_and_resize(room_path, max_side=768)

# 1. 先让两张图同尺寸
empty_img_pil = empty_img_pil.resize(room_img_pil.size, Image.LANCZOS)
width, height = room_img_pil.size

# 2. 强制宽高为 8 的倍数（diffusers 要求）
width = (width // 8) * 8
height = (height // 8) * 8

room_img_pil = room_img_pil.resize((width, height), Image.LANCZOS)
empty_img_pil = empty_img_pil.resize((width, height), Image.LANCZOS)

# 3. 自动生成家具 mask
empty_np = np.array(empty_img_pil).astype(np.int16)
cured_np = np.array(room_img_pil).astype(np.int16)

# 1) 颜色空间：Lab 的 ΔE 对光照鲁棒些
empty_lab = cv2.cvtColor(np.array(empty_img_pil), cv2.COLOR_RGB2LAB)
cured_lab = cv2.cvtColor(np.array(room_img_pil),  cv2.COLOR_RGB2LAB)
dE = cv2.absdiff(empty_lab, cured_lab)
dE = cv2.cvtColor(dE, cv2.COLOR_LAB2BGR)  # 把a/b差分也混入
dE_gray = cv2.cvtColor(dE, cv2.COLOR_BGR2GRAY)

# 2) 自适应阈值（Otsu），再加一个最低阈值下限，避免过敏感. Otsu 阈值：ret 是标量阈值，mask_tmp 是二值图
ret, mask_tmp = cv2.threshold(dE_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

min_th = 60  # 建议 50~80 之间试
thr = max(int(ret * 0.9), min_th)        # ret 是 float 标量，这里就不会报错了
_, mask_fg = cv2.threshold(dE_gray, thr, 255, cv2.THRESH_BINARY)


# 3) 只在画面下部/中部考虑家具（去掉墙/天花）
h, w = mask_fg.shape
mask_fg[:h//3, :] = 0                         # 上 1/3 不当作家具
# 可选：左右边缘再各去 3~5%（常见误检区）
trim = max(w//20, 10)
mask_fg[:, :trim] = 0
mask_fg[:, -trim:] = 0

# 4) 移除小连通域噪点（地板碎点）
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_fg, connectivity=8)
areas = stats[:, cv2.CC_STAT_AREA]
min_area = (h*w)//400   # 经验阈值：总像素的 0.25%（按需调）
clean = np.zeros_like(mask_fg)
for i in range(1, num_labels):
    if areas[i] >= min_area:
        clean[labels==i] = 255
mask_fg = clean


# 形态学操作，让区域更干净
kernel = np.ones((5, 5), np.uint8)
mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)

# 先做一次闭运算"合并孔洞"，再做轻微膨胀（更可控）
kernel = np.ones((7, 7), np.uint8)
mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=1)

# inpaint 约定：白=重绘
mask_inpaint = 255 - mask_fg

# 小半径羽化边（15~31 均可），避免吃到墙/地板
mask_inpaint = cv2.GaussianBlur(mask_inpaint, (21, 21), 0)
mask_pil = Image.fromarray(mask_inpaint).resize((width, height), Image.NEAREST)

# ★ NEW：给 mask 做一点高斯模糊，让边缘变软
mask_inpaint = cv2.GaussianBlur(mask_inpaint, (99, 99), 0)

mask_pil = Image.fromarray(mask_inpaint)
mask_pil = mask_pil.resize((width, height), Image.NEAREST)

# 4. 为 Canny 和 ControlNet 准备输入图像
input_image_rgb = np.array(room_img_pil)                          # RGB
input_image = cv2.cvtColor(input_image_rgb, cv2.COLOR_RGB2BGR)    # BGR

# -----------------------------
# 4. 提取边缘 (ControlNet 需要结构图)
# -----------------------------
canny_image = cv2.Canny(input_image, 100, 200)
canny_image = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB)
canny_pil = Image.fromarray(canny_image)
canny_pil = canny_pil.resize((width, height), Image.NEAREST)


# -----------------------------
# 5. 定义装修风格 prompt
# -----------------------------
prompt = "Modern-style living room interior, no furniture added into the room, use natural lighting from the window"
negative_prompt = " add furniture, add sofa, add table, change room structure, change furniture texture, dim lighting,  low quality"


# -----------------------------
# 6. 生成效果图（带自动家具 mask 的 Inpaint）
# -----------------------------
generator = torch.Generator(device="cuda")


output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,   # 没需要可删
    image=room_img_pil,                # ★ 基础图：有沙发桌子的房间
    control_image=canny_pil,           # ★ ControlNet 的 Canny 结构图
    mask_image=mask_pil,               # ★ 自动生成家具 mask（黑=保留）
    num_inference_steps=25,
    guidance_scale=4.5,
    height=height, width=width,
    num_images_per_prompt=1,
    generator=generator
)


result_image = output.images[0]

# -----------------------------
# 7. 保存效果图
# -----------------------------
result_image.save("furnished_room.png")
canny_pil.save("edge_map.png")
mask_pil.save("mask.png")
print("装修效果图生成完成！")

from diffusers import UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline

# Initialize the img2img_pipe
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
img2img_pipe.to("cuda")

# ★ 优化 scheduler（只需设置一次）
img2img_pipe.scheduler = UniPCMultistepScheduler.from_config(
    img2img_pipe.scheduler.config
)

# Load the base image for harmonization
base = Image.open("furnished_room.png").convert("RGB")

result2 = img2img_pipe(
    prompt=prompt + ", high detail, sharp focus, 8k, high clarity",   # ★ 强调清晰细节
    negative_prompt=negative_prompt + ", blurry, low detail, soft, smudged",
    image=base,
    strength=0.2,              # ★ 降到 0.05–0.10，只做轻微 harmonize
    guidance_scale=7.0,         # ★ 稍微提高 CFG，让它更听 prompt
    num_inference_steps=30      # ★ 步数拉到 30 左右，细节会回来一些
).images[0]

result2.save("furnished_room_harmonized.png")
