import io
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image as PILImage
import numpy as np
import cv2

# Fix torchvision breaking change in basicsr
import torchvision.transforms.functional as TF
import sys
sys.modules['torchvision.transforms.functional_tensor'] = TF
sys.modules['torchvision.transforms.functional_tensor'].rgb_to_grayscale = TF.rgb_to_grayscale

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ─────────────────────────────────────────────
# Model Architecture (exact copy from notebook)
# ─────────────────────────────────────────────

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect")
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.InstanceNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.InstanceNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        # Encoder
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features,    features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU(),
        )
        # Attention gates
        self.attn7 = AttentionGate(features * 8, features * 8, features * 4)
        self.attn6 = AttentionGate(features * 8, features * 8, features * 4)
        self.attn5 = AttentionGate(features * 8, features * 8, features * 4)
        self.attn4 = AttentionGate(features * 8, features * 8, features * 4)
        self.attn3 = AttentionGate(features * 4, features * 4, features * 2)
        self.attn2 = AttentionGate(features * 2, features * 2, features)
        self.attn1 = AttentionGate(features,     features,     features // 2)
        # Decoder
        self.up1 = Block(features * 8,      features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2,  features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2,  features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2,  features * 8, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 8 * 2,  features * 4, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features * 4 * 2,  features * 2, down=False, act="relu", use_dropout=False)
        self.up7 = Block(features * 2 * 2,  features,     down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)

        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, self.attn7(u1, d7)], 1))
        u3 = self.up3(torch.cat([u2, self.attn6(u2, d6)], 1))
        u4 = self.up4(torch.cat([u3, self.attn5(u3, d5)], 1))
        u5 = self.up5(torch.cat([u4, self.attn4(u4, d4)], 1))
        u6 = self.up6(torch.cat([u5, self.attn3(u5, d3)], 1))
        u7 = self.up7(torch.cat([u6, self.attn2(u6, d2)], 1))
        return self.final_up(torch.cat([u7, self.attn1(u7, d1)], 1))


# ─────────────────────────────────────────────
# Load weights on startup
# ─────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).parent

# Priority: latest trained weights first
WEIGHT_CANDIDATES = [
    BASE_DIR / "latestOutput" / "cGAN_Colorization_Checkpoints" / "generator.pth",
    BASE_DIR / "latestOutput" / "cGAN_Colorization_Checkpoints" / "gen.pth.tar",
    BASE_DIR / "generator.pth",
    BASE_DIR / "gen.pth.tar",
]

gen = Generator(in_channels=3).to(DEVICE)
# gen.eval() # Keep dropout active to simulate noise for cGAN and avoid mode collapse

loaded = False
for path in WEIGHT_CANDIDATES:
    if path.exists():
        print(f"Loading weights from: {path}")
        checkpoint = torch.load(path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            gen.load_state_dict(checkpoint["state_dict"])
        else:
            gen.load_state_dict(checkpoint)
        loaded = True
        print(f"✅ Model loaded successfully from {path.name}  |  Device: {DEVICE}")
        break

if not loaded:
    print("[WARNING] No generator weight files found! Check generator.pth.")

# ─────────────────────────────────────────────
# Load RealESRGAN Post-processing
# ─────────────────────────────────────────────
esrgan_path = BASE_DIR / "latestOutput" / "RealESRGAN_x4plus_anime_6B.pth"
print(f"Loading RealESRGAN from: {esrgan_path}")
upsampler = None
if esrgan_path.exists():
    esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(esrgan_path),
        model=esrgan_model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=(DEVICE.type == "cuda")
    )
    print("✅ RealESRGAN loaded successfully!")
else:
    print("[WARNING] RealESRGAN weights not found at expected path. Image upscaling will be skipped.")

# ─────────────────────────────────────────────
# Preprocessing transform (same as training)
# ─────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(title="Sketch to Color GAN")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


from fastapi import Response


def _img_response(img_np, filename="output.png"):
    """Helper to convert a numpy RGB array to a PNG Response."""
    output_img = PILImage.fromarray(img_np)
    buf = io.BytesIO()
    output_img.save(buf, format="PNG")
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


# ─────────────────────────────────────────────
# 1) /colorize — Raw GAN output only (256×256)
# ─────────────────────────────────────────────
@app.post("/colorize")
async def colorize(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        img = PILImage.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = gen(input_tensor)

    # Denormalize: [-1, 1] → [0, 1]
    output = output * 0.5 + 0.5
    output = output.clamp(0, 1)

    output_np = (output.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)

    return _img_response(output_np, "colorized.png")


# ─────────────────────────────────────────────
# 2) /upscale — Real-ESRGAN 4× super-resolution
# ─────────────────────────────────────────────
@app.post("/upscale")
async def upscale(file: UploadFile = File(...)):
    if upsampler is None:
        raise HTTPException(status_code=503, detail="Real-ESRGAN model is not loaded.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        img = PILImage.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    img_np = np.array(img)

    try:
        output_np, _ = upsampler.enhance(img_np, outscale=4)
    except Exception as e:
        print(f"[Error] Real-ESRGAN upscale failed: {e}")
        raise HTTPException(status_code=500, detail="Upscaling failed.")

    return _img_response(output_np, "upscaled.png")


# ─────────────────────────────────────────────
# 3) /color-transfer — Reinhard histogram transfer
# ─────────────────────────────────────────────
@app.post("/color-transfer")
async def color_transfer(
    file: UploadFile = File(...),
    reference_file: UploadFile = File(...),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    if reference_file.content_type and not reference_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Reference file must be an image.")

    try:
        contents = await file.read()
        img = PILImage.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read source image.")

    try:
        ref_contents = await reference_file.read()
        ref_img = PILImage.open(io.BytesIO(ref_contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read reference image.")

    target_np = np.array(img)
    source_np = np.array(ref_img)

    # Reinhard color transfer in CIELAB space
    source = cv2.cvtColor(source_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    target = cv2.cvtColor(target_np, cv2.COLOR_RGB2LAB).astype(np.float32)

    s_mean, s_std = cv2.meanStdDev(source)
    t_mean, t_std = cv2.meanStdDev(target)

    for i in range(3):
        std_ratio = (s_std[i][0] / t_std[i][0]) if t_std[i][0] != 0 else 0
        target[:, :, i] = ((target[:, :, i] - t_mean[i][0]) * std_ratio) + s_mean[i][0]

    target = np.clip(target, 0, 255).astype(np.uint8)
    output_np = cv2.cvtColor(target, cv2.COLOR_LAB2RGB)

    return _img_response(output_np, "color_transferred.png")
