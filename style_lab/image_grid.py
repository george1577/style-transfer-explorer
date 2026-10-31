"""The image grid: a manifest of precomputed images, looked up by their generation parameters.

The Colab grid script (`scripts/render_grid.py`) writes the manifest with `write_manifest`; the app reads it with
`GridIndex.load`. Images are found by parameters, never by filename conventions.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "grid" / "manifest.json"

# The precomputed axes; the app's sliders offer exactly these values.
STRENGTHS = (0.3, 0.5, 0.7, 0.9)
GUIDANCES = (3.0, 7.5, 12.0)
# The with/without depth conditioning comparison is rendered at this one setting.
COMPARISON_STRENGTH, COMPARISON_GUIDANCE = 0.7, 7.5


@dataclass(frozen=True)
class ReferenceImage:
    key: str
    title: str
    artist: str
    path: Path
    source_url: str
    prompts: tuple[str, str]


REFERENCE_IMAGES = {
    r.key: r
    for r in (
        ReferenceImage(
            key="wanderer",
            title="Wanderer above the Sea of Fog (c. 1817)",
            artist="Caspar David Friedrich",
            path=ROOT / "data" / "images" / "wanderer.jpg",
            source_url="https://commons.wikimedia.org/wiki/File:Caspar_David_Friedrich_-_Wanderer_above_the_sea_of_fog.jpg",
            prompts=(
                "Star Wars' Darth Vader with a red light saber",  # notebook 3's prompt
                "an astronaut on a rocky peak above a sea of clouds, Japanese ukiyo-e woodblock print",
            ),
        ),
        ReferenceImage(
            key="paris-street",
            title="Paris Street; Rainy Day (1877)",
            artist="Gustave Caillebotte",
            path=ROOT / "data" / "images" / "paris-street-rainy-day.jpg",
            source_url="https://commons.wikimedia.org/wiki/File:Gustave_Caillebotte_-_Paris_Street;_Rainy_Day_-_Google_Art_Project.jpg",
            prompts=(
                "a neon-lit cyberpunk city street at night in the rain, people with umbrellas",
                "a snowy village street in winter, people in warm coats, watercolor painting",
            ),
        ),
    )
}



@dataclass(frozen=True)
class GridImage:
    path: Path
    reference: str
    prompt: str
    strength: float
    guidance: float
    depth_conditioned: bool
    seed: int
    steps: int
    models: dict


@dataclass(frozen=True)
class Missing:
    """No image for these parameters yet; the UI shows a placeholder with `reason`."""
    reason: str


class GridIndex:
    def __init__(self, images: list[GridImage]):
        self._images = {self._key(i.reference, i.prompt, i.strength, i.guidance, i.depth_conditioned): i
                        for i in images}

    @staticmethod
    def _key(reference, prompt, strength, guidance, depth_conditioned):
        return (reference, prompt, round(float(strength), 3), round(float(guidance), 3), bool(depth_conditioned))

    @classmethod
    def load(cls, manifest_path: Path) -> "GridIndex":
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            return cls([])
        data = json.loads(manifest_path.read_text())
        images = [
            GridImage(path=manifest_path.parent / e["file"], reference=e["reference"], prompt=e["prompt"],
                      strength=e["strength"], guidance=e["guidance"], depth_conditioned=e["depth_conditioned"],
                      seed=data["seed"], steps=data["steps"], models=data["models"])
            for e in data["images"]
        ]
        return cls(images)

    def lookup(self, reference: str, prompt: str, strength: float, guidance: float,
               depth_conditioned: bool = True) -> GridImage | Missing:
        image = self._images.get(self._key(reference, prompt, strength, guidance, depth_conditioned))
        if image is None:
            return Missing("not rendered yet" if self.is_empty else "not in the grid manifest")
        if not image.path.exists():
            return Missing(f"listed in the manifest but {image.path.name} is not on disk")
        return image

    @property
    def is_empty(self) -> bool:
        return not self._images


@dataclass(frozen=True)
class GridJob:
    """One image the grid script renders."""
    reference: str
    prompt: str
    strength: float
    guidance: float
    depth_conditioned: bool = True

    @property
    def filename(self) -> str:
        # Only for humans browsing the folder; the app finds images through the manifest.
        slug = re.sub(r"[^a-z0-9]+", "-", self.prompt.lower()).strip("-")[:40]
        depth = "depth" if self.depth_conditioned else "nodepth"
        return f"{self.reference}_{slug}_s{self.strength}_g{self.guidance}_{depth}.jpg"


def write_manifest(path: Path, jobs: list[GridJob], seed: int, steps: int, models: dict) -> None:
    images = [{"file": j.filename, "reference": j.reference, "prompt": j.prompt, "strength": j.strength,
               "guidance": j.guidance, "depth_conditioned": j.depth_conditioned} for j in jobs]
    Path(path).write_text(json.dumps({"seed": seed, "steps": steps, "models": models, "images": images}, indent=2))


def grid_plan() -> list[GridJob]:
    """Every image in the grid: each prompt at every strength x guidance, plus one without depth conditioning."""
    jobs = []
    for ref in REFERENCE_IMAGES.values():
        for prompt in ref.prompts:
            jobs += [GridJob(ref.key, prompt, s, g) for s in STRENGTHS for g in GUIDANCES]
            jobs.append(GridJob(ref.key, prompt, COMPARISON_STRENGTH, COMPARISON_GUIDANCE, depth_conditioned=False))
    return jobs
