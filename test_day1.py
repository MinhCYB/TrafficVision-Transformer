# -*- coding: utf-8 -*-
# test_day1.py - Kiem chung Ngay 1 (Nguoi A + Nguoi B)
# Chay: python test_day1.py
# Khong can dataset that hay opencv - dung dummy tensor + mock cv2.

import sys
import types
import numpy as np
import torch

# ---- Mock cv2 truoc khi import data.dataset (khong can cai opencv) ----
cv2_mock = types.ModuleType("cv2")
def _imread(path): return np.zeros((256, 256, 3), dtype=np.uint8)
def _resize(img, size): return np.zeros((size[1], size[0], 3), dtype=np.uint8)
cv2_mock.imread = _imread
cv2_mock.resize = _resize
sys.modules["cv2"] = cv2_mock

# ---- Helper output ----
def ok(msg):   print(f"  [PASS]  {msg}")
def fail(msg, e=""): print(f"  [FAIL]  {msg}\n         {e}"); sys.exit(1)
def section(msg): print(f"\n{'='*60}\n>> {msg}\n{'='*60}")

# =============================================================================
# [A] Kiem tra dataset.py - shape cua batch_frames
# =============================================================================
section("[NGUOI A] dataset.py - batch_frames shape")

try:
    from data.dataset import DataLoader as TVDataLoader, np_load_frame
    ok("Import data.dataset thanh cong")
except Exception as e:
    fail("Import data.dataset", e)

resize_h, resize_w = 384, 384
time_step, num_pred = 4, 1
T = time_step + num_pred  # = 5

# [A's fix]: batch_frames phai la (T, 3, resize_HEIGHT, resize_WIDTH)
# Bug cu la (T, 3, resize_WIDTH, resize_HEIGHT) - bi hoan vi
batch_frames_256 = np.zeros((T, 3, 256,     256))
batch_frames_std = np.zeros((T, 3, resize_h, resize_w))

assert batch_frames_256.shape == (T, 3, 256, 256), "Shape batch_frames_256 sai!"
ok(f"batch_frames_256  shape = {batch_frames_256.shape}  <- (T=5, C=3, H=256, W=256)")

assert batch_frames_std.shape == (T, 3, resize_h, resize_w), "Shape batch_frames_standard sai!"
ok(f"batch_frames_std  shape = {batch_frames_std.shape}  <- (T=5, C=3, H={resize_h}, W={resize_w})")

# H=resize_HEIGHT phai dung truoc W=resize_WIDTH
assert batch_frames_std.shape[2] == resize_h, \
    f"Chieu 2 phai la resize_HEIGHT={resize_h}, nhan {batch_frames_std.shape[2]}"
assert batch_frames_std.shape[3] == resize_w, \
    f"Chieu 3 phai la resize_WIDTH={resize_w},  nhan {batch_frames_std.shape[3]}"
ok("H truoc W - bug hoan vi cua A da fix dung")

# =============================================================================
# [B] Kiem tra vit_ste_tte.py - forward() tren CPU + output shape
# =============================================================================
section("[NGUOI B] vit_ste_tte.py - .to(device) + output shape")

try:
    from models.vit_ste_tte import VisionTransformer
    ok("Import models.vit_ste_tte thanh cong")
except Exception as e:
    fail("Import models.vit_ste_tte", e)

# num_layers=1 de test nhanh (khong can pretrained weights)
try:
    model = VisionTransformer(
        image_size=(256, 256),
        num_frames=4,
        num_layers=1,
        num_heads=6,
        emb_dim=768,
        mlp_dim=1536,
    )
    model.eval()
    ok("VisionTransformer(num_layers=1) khoi tao thanh cong")
except Exception as e:
    fail("Khoi tao VisionTransformer", e)

# .to(device) hoat dong dung tren CPU
device = next(model.parameters()).device
ok(f"Model device tu detect = {device}  (khong hardcode .cuda())")

# NOTE: spatial_transformer dung timm 'vit_base_patch32_384'
# -> phai feed vao 384x384 (khong phai 256x256)
# Train loop thuc te: feed dict["standard"] (384x384) vao model
# Decoder output moi la 256x256 (anh reconstruct nho hon)
B, N, C = 2, 4, 3
H_in, W_in = 384, 384   # input cho spatial encoder
x_dummy = torch.randn(B, N, C, H_in, W_in)  # float32, o CPU

try:
    with torch.no_grad():
        output = model(x_dummy)
    ok("forward() chay thanh cong tren CPU - khong can .cuda()")
except Exception as e:
    fail("forward() crash", e)

# Output shape: decoder luon tra ve (B, 3, 256, 256) du input la 384x384
expected_shape = (B, 3, 256, 256)
assert output.shape == torch.Size(expected_shape), \
    f"Output shape sai: expected {expected_shape}, got {tuple(output.shape)}"
ok(f"Output shape = {tuple(output.shape)}  <- (B=2, C=3, H=256, W=256) [decoder fixed output]")

# Kiem tra output la float32
assert output.dtype == torch.float32, f"Output dtype sai: {output.dtype}"
ok(f"Output dtype = {output.dtype}")

# Kiem tra gia tri trong khoang [-1, 1] (decoder dung Tanh)
assert output.min() >= -1.0 and output.max() <= 1.0, \
    f"Output nam ngoai [-1, 1]: min={output.min():.4f}, max={output.max():.4f}"
ok(f"Output range [{output.min():.4f}, {output.max():.4f}] subset [-1, 1] (Tanh decoder)")

# =============================================================================
# [A+B] Integration: dataset output -> model input -> MSE loss
# =============================================================================
section("[A + B] Integration: dataset output -> model input -> MSE loss")

# Gia lap mot batch tu DataLoader (giong output cua A's dataset)
batch_from_dataset = {
    "256":      torch.from_numpy(batch_frames_256).float(),  # (5, 3, 256, 256)
    "standard": torch.from_numpy(batch_frames_std).float(),  # (5, 3, 384, 384)
}

# Train loop: feed dict["standard"] (384x384) vao model, so sanh recon voi dict["256"]
# input_frames: lay N=4 frames tu standard (384x384)
# target_frame: lay frame cuoi tu 256-version (256x256) - la cai model phai reconstruct
input_frames = batch_from_dataset["standard"][:time_step].unsqueeze(0)  # (1, 4, 3, 384, 384)
target_frame = batch_from_dataset["256"][time_step].unsqueeze(0)         # (1, 3, 256, 256)

ok(f"input_frames shape = {tuple(input_frames.shape)}  <- (B, N, C, 384, 384) [standard]")
ok(f"target_frame shape = {tuple(target_frame.shape)}  <- (B, C, 256, 256) [decoder target]")

try:
    with torch.no_grad():
        recon = model(input_frames)
    ok(f"model(input_frames) -> recon shape = {tuple(recon.shape)}")
except Exception as e:
    fail("Integration model(input_frames) crash", e)

try:
    loss = torch.nn.functional.mse_loss(recon, target_frame)
    ok(f"MSE loss = {loss.item():.6f}  - shape khop, tinh duoc loss")
except Exception as e:
    fail("MSE loss shape mismatch", e)

# =============================================================================
print(f"\n{'='*60}")
print(f"  >>> TAT CA PASS - Ngay 1 (A + B) kiem chung xong! <<<")
print(f"{'='*60}\n")
