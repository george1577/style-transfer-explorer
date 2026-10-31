"""Seam A, image side: the grid index over a fake manifest (no models, no real grid)."""
import json

from style_lab.image_grid import GridImage, GridIndex, Missing

SETTINGS = {"seed": 7, "steps": 30, "models": {"diffusion": "fake-sd", "controlnet": "fake-cn", "depth": "fake-depth"}}


def write_fake_grid(tmp_path, entries, files=None):
    """A manifest listing `entries`; image files are created for `files` (default: all entries)."""
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    (grid_dir / "manifest.json").write_text(json.dumps({**SETTINGS, "images": entries}))
    for name in files if files is not None else [e["file"] for e in entries]:
        (grid_dir / name).write_bytes(b"jpeg")
    return grid_dir / "manifest.json"


def entry(file, reference="wanderer", prompt="a knight", strength=0.5, guidance=7.5, depth=True):
    return {"file": file, "reference": reference, "prompt": prompt,
            "strength": strength, "guidance": guidance, "depth_conditioned": depth}


def test_lookup_returns_the_image_for_those_parameters(tmp_path):
    manifest = write_fake_grid(tmp_path, [
        entry("a.jpg", strength=0.5, guidance=7.5),
        entry("b.jpg", strength=0.7, guidance=7.5),
        entry("c.jpg", strength=0.7, guidance=12),
        entry("d.jpg", prompt="a robot", strength=0.7, guidance=12),
        entry("e.jpg", reference="paris", strength=0.7, guidance=12),
    ])
    index = GridIndex.load(manifest)

    found = index.lookup("wanderer", "a knight", strength=0.7, guidance=12)

    assert isinstance(found, GridImage)
    assert found.path == tmp_path / "grid" / "c.jpg"
    assert (found.strength, found.guidance, found.depth_conditioned) == (0.7, 12, True)
    assert (found.seed, found.steps) == (7, 30)
    assert found.models == SETTINGS["models"]


def test_without_depth_conditioning_variant_resolves_separately(tmp_path):
    manifest = write_fake_grid(tmp_path, [
        entry("with.jpg", strength=0.7, guidance=7.5, depth=True),
        entry("without.jpg", strength=0.7, guidance=7.5, depth=False),
    ])
    index = GridIndex.load(manifest)

    with_depth = index.lookup("wanderer", "a knight", 0.7, 7.5, depth_conditioned=True)
    without = index.lookup("wanderer", "a knight", 0.7, 7.5, depth_conditioned=False)

    assert with_depth.path.name == "with.jpg"
    assert without.path.name == "without.jpg"
    assert without.depth_conditioned is False


def test_parameters_not_in_the_manifest_come_back_missing(tmp_path):
    manifest = write_fake_grid(tmp_path, [entry("a.jpg", strength=0.5, guidance=7.5)])
    index = GridIndex.load(manifest)

    assert isinstance(index.lookup("wanderer", "a knight", 0.9, 7.5), Missing)
    assert isinstance(index.lookup("wanderer", "a robot", 0.5, 7.5), Missing)
    assert isinstance(index.lookup("wanderer", "a knight", 0.5, 7.5, depth_conditioned=False), Missing)


def test_listed_image_whose_file_is_absent_comes_back_missing(tmp_path):
    manifest = write_fake_grid(tmp_path, [entry("a.jpg"), entry("gone.jpg", strength=0.9)], files=["a.jpg"])

    assert isinstance(GridIndex.load(manifest).lookup("wanderer", "a knight", 0.9, 7.5), Missing)


def test_before_the_grid_is_rendered_every_lookup_is_missing(tmp_path):
    index = GridIndex.load(tmp_path / "grid" / "manifest.json")

    assert isinstance(index.lookup("wanderer", "a knight", 0.7, 7.5), Missing)
    assert index.is_empty


def test_manifest_written_by_the_grid_script_is_read_by_the_app(tmp_path):
    from style_lab.image_grid import GridJob, write_manifest

    job = GridJob(reference="paris", prompt="a snowy street", strength=0.3, guidance=3.0, depth_conditioned=False)
    (tmp_path / job.filename).write_bytes(b"jpeg")
    write_manifest(tmp_path / "manifest.json", [job], seed=11, steps=40, models=SETTINGS["models"])

    found = GridIndex.load(tmp_path / "manifest.json").lookup("paris", "a snowy street", 0.3, 3.0,
                                                               depth_conditioned=False)

    assert found.path == tmp_path / job.filename
    assert (found.seed, found.steps, found.models) == (11, 40, SETTINGS["models"])


def test_grid_plan_is_52_images_with_one_unconditioned_image_per_prompt():
    from style_lab.image_grid import REFERENCE_IMAGES, grid_plan

    jobs = grid_plan()
    unconditioned = [j for j in jobs if not j.depth_conditioned]

    assert len(REFERENCE_IMAGES) == 2
    assert all(len(r.prompts) == 2 for r in REFERENCE_IMAGES.values())
    assert len(jobs) == 52
    assert len(set(jobs)) == 52
    assert len({j.filename for j in jobs}) == 52
    assert {(j.strength, j.guidance) for j in jobs if j.depth_conditioned} == {
        (s, g) for s in (0.3, 0.5, 0.7, 0.9) for g in (3.0, 7.5, 12.0)}
    assert len(unconditioned) == 4
    assert {(j.strength, j.guidance) for j in unconditioned} == {(0.7, 7.5)}
    assert {(j.reference, j.prompt) for j in unconditioned} == {
        (r.key, p) for r in REFERENCE_IMAGES.values() for p in r.prompts}



def test_live_generation_is_offered_only_on_a_gpu_device():
    from style_lab.live_generation import live_generation_available

    assert live_generation_available("cuda")
    assert live_generation_available("cuda:0")
    assert live_generation_available("mps")
    assert not live_generation_available("cpu")
