"""Device choice: CPU unless STE_DEVICE says otherwise; GPU-only features need CUDA."""
from style_lab.device import DEVICE_ENV_VAR, is_cuda, pick_device
from style_lab.live_generation import live_generation_available


def test_device_is_cpu_unless_the_env_var_is_set(monkeypatch):
    monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)
    assert pick_device() == "cpu"

    monkeypatch.setenv(DEVICE_ENV_VAR, "mps")
    assert pick_device() == "mps"


def test_cuda_devices_are_recognised():
    assert is_cuda("cuda") and is_cuda("cuda:1")
    assert not is_cuda("mps") and not is_cuda("cpu")


def test_live_generation_is_offered_only_on_cuda():
    assert live_generation_available("cuda")
    assert live_generation_available("cuda:0")
    assert not live_generation_available("mps")
    assert not live_generation_available("cpu")
