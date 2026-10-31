"""Render tab ④'s image grid (52 JPEGs + manifest.json) on a Colab T4 GPU.

How to run (Google Colab, Runtime -> Change runtime type -> T4 GPU), one cell per line:

    !git clone https://github.com/george1577/style-transfer-explorer.git
    %cd style-transfer-explorer
    !pip install -q "diffusers>=0.30" "accelerate>=1.0" "transformers>=4.46" pillow
    !python scripts/render_grid.py
    !zip -qr grid.zip data/grid && echo done      # then download grid.zip from the Files panel

Unzip it into the repo so the images land in `data/grid/`, and commit `data/grid/`. The app finds each image through
`data/grid/manifest.json`, so don't rename files. Takes about 15 minutes on a T4 (models download ~5 GB first).

What it renders: for each of the 2 reference images x 2 prompts, every strength (0.3/0.5/0.7/0.9) x guidance (3/7.5/12)
with depth conditioning, plus one image without depth conditioning at strength 0.7 / guidance 7.5. Same seed and step
count for every image. Images are not square: each keeps its reference image's aspect ratio, with a 512 px long
side (the size the reference images are stored at). Grid contents and the manifest format live in
`style_lab/image_grid.py`, shared with the app.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from style_lab.depth import DEPTH_MODEL, DepthEstimator  # noqa: E402
from style_lab.image_grid import MANIFEST_PATH, REFERENCE_IMAGES, grid_plan, write_manifest  # noqa: E402
from style_lab.live_generation import CONTROLNET_MODEL, DIFFUSION_MODEL, DIFFUSION_SEED, STEPS, DepthImg2Img  # noqa: E402


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found. Run this on Colab with a T4 runtime (see the docstring).")
    out_dir = MANIFEST_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    references = {key: Image.open(ref.path).convert("RGB") for key, ref in REFERENCE_IMAGES.items()}
    estimator = DepthEstimator(device="cuda")
    depth_maps = {key: estimator.depth_map(image) for key, image in references.items()}
    del estimator

    generator = DepthImg2Img(device="cuda")
    jobs = grid_plan()
    start = time.time()
    for n, job in enumerate(jobs, 1):
        image = generator.generate(references[job.reference], depth_maps[job.reference], job.prompt,
                                   strength=job.strength, guidance=job.guidance,
                                   depth_conditioned=job.depth_conditioned, seed=DIFFUSION_SEED, steps=STEPS)
        image.resize(references[job.reference].size).save(out_dir / job.filename, quality=88)
        print(f"[{n}/{len(jobs)}] {job.filename} ({time.time() - start:.0f}s)", flush=True)

    write_manifest(MANIFEST_PATH, jobs, seed=DIFFUSION_SEED, steps=STEPS,
                   models={"diffusion": DIFFUSION_MODEL, "controlnet": CONTROLNET_MODEL, "depth": DEPTH_MODEL})
    print(f"Wrote {len(jobs)} images and {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
