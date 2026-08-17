import torch
from contextlib import nullcontext

def apply_xpu_patches(device: torch.device):
    if device.type != "xpu":
        return
    print("[XPU PATCH] 正在应用 XPU 兼容性补丁...")
    _patch_amp()
    _patch_task_aligned_assigner(device)
    print("[XPU PATCH] 所有补丁已应用")


def _patch_amp():
    pass


def _patch_task_aligned_assigner(device: torch.device):
    try:
        from ultralytics.utils.tal import TaskAlignedAssigner
        _original_forward = TaskAlignedAssigner._forward
        def _cpu_assigner_forward(self, pd_scores, pd_bboxes, anc_points,
                                   gt_labels, gt_bboxes, mask_gt):
            import torch
            dev = pd_scores.device
            self.bs = pd_scores.shape[0]
            self.n_max_boxes = gt_bboxes.shape[1]
            if self.n_max_boxes == 0:
                return (
                    torch.full_like(pd_scores[..., 0], self.num_classes),
                    torch.zeros_like(pd_bboxes),
                    torch.zeros_like(pd_scores),
                    torch.zeros_like(pd_scores[..., 0]),
                    torch.zeros_like(pd_scores[..., 0]),
                )
            cpu_args = (
                pd_scores.detach().cpu(),
                pd_bboxes.detach().cpu(),
                anc_points.detach().cpu(),
                gt_labels.detach().cpu(),
                gt_bboxes.detach().cpu(),
                mask_gt.detach().cpu(),
            )
            result = _original_forward(self, *cpu_args)
            return tuple(t.to(dev) for t in result)
        TaskAlignedAssigner.forward = _cpu_assigner_forward
        print("[XPU PATCH] TaskAlignedAssigner.forward → CPU (避免 hang)")

    except ImportError:
        print("[XPU PATCH] 警告: 无法导入 TaskAlignedAssigner，跳过补丁")
    except Exception as e:
        print(f"[XPU PATCH] 警告: TaskAlignedAssigner 补丁失败: {e}")


def get_autocast_context(device: torch.device):
    if device.type == "xpu":
        return nullcontext()
    elif device.type == "cuda":
        from torch.cuda.amp import autocast
        return autocast()
    else:
        return nullcontext()


def get_pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"


def get_non_blocking(device: torch.device) -> bool:
    return device.type != "xpu"


def get_gpu_memory_info(device: torch.device) -> str:
    if device.type == "xpu" and hasattr(torch, 'xpu'):
        reserved = torch.xpu.memory_reserved(device.index or 0) / 1e9
        allocated = torch.xpu.memory_allocated(device.index or 0) / 1e9
        return f"XPU 显存: {allocated:.2f}GB / {reserved:.2f}GB (reserved)"
    elif device.type == "cuda":
        reserved = torch.cuda.memory_reserved(device.index or 0) / 1e9
        allocated = torch.cuda.memory_allocated(device.index or 0) / 1e9
        return f"CUDA 显存: {allocated:.2f}GB / {reserved:.2f}GB (reserved)"
    else:
        return "CPU 模式 (无显存信息)"


def print_device_info(device: torch.device):
    print(f"设备类型: {device.type}")
    print(f"设备索引: {device.index}")
    if device.type == "xpu" and hasattr(torch, 'xpu'):
        print(f"设备名称: {torch.xpu.get_device_name(device.index or 0)}")
        print(f"XPU 数量: {torch.xpu.device_count()}")
    elif device.type == "cuda":
        print(f"设备名称: {torch.cuda.get_device_name(device.index or 0)}")
        print(f"CUDA 数量: {torch.cuda.device_count()}")
    print(get_gpu_memory_info(device))