# rp_handler.py
"""
RunPod serverless handler · Stable Diffusion img-to-img
------------------------------------------------------
• Carga el pipeline una sola vez en GPU (float16)
• Soporta seed, strength, guidance_scale, steps, num_images
• Devuelve la primera imagen como data-URL base64
"""

import base64, io, os, torch, runpod
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image

MODEL_ID = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")

def _load_pipeline():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    ).to("cuda")

    # Aceleraciones opcionales (si xformers está disponible)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    pipe.enable_model_cpu_offload()
    return pipe

PIPE = _load_pipeline()         # <-- se crea una vez por worker

def handler(event):
    p = event["input"]

    init_img   = load_image(p["image_url"]).convert("RGB")
    generator  = torch.Generator(device="cuda").manual_seed(p.get("seed", 0))

    result = PIPE(
        prompt                = p["prompt"],
        negative_prompt       = p.get("negative_prompt"),
        image                 = init_img,
        strength              = p.get("strength", 0.8),
        guidance_scale        = p.get("guidance_scale", 7.5),
        num_inference_steps   = p.get("steps", 30),
        generator             = generator,
        num_images_per_prompt = p.get("num_images", 1)
    ).images[0]

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()

    return {"image": f"data:image/png;base64,{encoded}"}

# Entrypoint del worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
