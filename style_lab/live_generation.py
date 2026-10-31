"""Optional live image generation: SD1.5 + depth ControlNet img2img, the same pipeline as the grid script.

Only offered on a GPU device (see ADR 0002); never loaded on CPU.
"""
from PIL import Image

DIFFUSION_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/control_v11f1p_sd15_depth"
SEED = 42
STEPS = 50


def live_generation_available(device: str) -> bool:
    return device.startswith("cuda") or device == "mps"


class DepthImg2Img:
    def __init__(self, device: str):
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler

        self.device = device
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            DIFFUSION_MODEL, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        if device.startswith("cuda"):
            self.pipe.enable_model_cpu_offload()  # fits a 16 GB T4 with room to spare
        else:
            self.pipe.to(device)

    def generate(self, reference: Image.Image, depth_map: Image.Image, prompt: str, strength: float,
                 guidance: float, depth_conditioned: bool = True, seed: int = SEED, steps: int = STEPS) -> Image.Image:
        """Without depth conditioning = the same pipeline with the ControlNet's influence set to 0."""
        import torch

        generator = torch.Generator("cpu").manual_seed(seed)
        return self.pipe(
            prompt, image=reference.convert("RGB"), control_image=depth_map.convert("RGB"),
            strength=strength, guidance_scale=guidance, num_inference_steps=steps,
            controlnet_conditioning_scale=1.0 if depth_conditioned else 0.0, generator=generator,
        ).images[0]
