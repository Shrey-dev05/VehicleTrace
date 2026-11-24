"""
Device and Execution Hardware Selector
"""
import logging
from dataclasses import dataclass
import torch

log = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    device: str = "cpu"
    name: str = "CPU"
    cuda_available: bool = False
    fp16: bool = False
    total_gb: float = 0.0
    reason: str = ""
    verified: bool = False


def get_optimal_device(force_cpu: bool = False, request_fp16: bool = True) -> DeviceInfo:
    """Select CUDA if operational and kernels execute successfully; fallback safely to CPU."""
    cuda_avail = torch.cuda.is_available()
    if force_cpu or not cuda_avail:
        return DeviceInfo(
            device="cpu",
            name="CPU",
            cuda_available=cuda_avail,
            fp16=False,
            reason="CPU mode selected or CUDA unavailable"
        )
    
    try:
        # Run test tensor operation to ensure CUDA driver/kernel works
        t = torch.randn(64, 64, device="cuda")
        _ = float((t @ t).sum().item())
        props = torch.cuda.get_device_properties(0)
        use_fp16 = request_fp16 and (props.major >= 6)
        total_gb = round(props.total_memory / (1024 ** 3), 2)
        return DeviceInfo(
            device="cuda",
            name=props.name,
            cuda_available=True,
            fp16=use_fp16,
            total_gb=total_gb,
            verified=True
        )
    except Exception as e:
        log.warning(f"CUDA initialization failed ({e}), falling back to CPU.")
        return DeviceInfo(
            device="cpu",
            name="CPU",
            cuda_available=True,
            fp16=False,
            reason=f"CUDA test failed: {e}"
        )
