"""Depth maps with Depth Anything V2 Small. Used live by the app and inside the grid script, so the two match."""
from PIL import Image

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    def __init__(self, device: str = "cpu"):
        from transformers import pipeline

        self.device = device
        self._pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device=device)

    def depth_map(self, image: Image.Image) -> Image.Image:
        """A grayscale depth map the same size as `image`; brighter means nearer."""
        depth = self._pipe(image.convert("RGB"))["depth"]
        return depth.convert("L").resize(image.size)
