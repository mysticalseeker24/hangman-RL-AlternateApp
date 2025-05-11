import torch
import gc
import psutil

def setup_cuda():
    """
    Sets up CUDA device if available, otherwise uses CPU.
    Returns the selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU instead")
    return device

def cleanup_cuda_memory():
    """
    Basic CUDA memory cleanup.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"CUDA memory after cleanup:")
        print(f"- Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"- Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

def get_system_memory_info():
    vm = psutil.virtual_memory()
    return f"{vm.used/1024**3:.1f}GB used / {vm.total/1024**3:.1f}GB total ({vm.percent}%)"

def memory_efficient_training_step(model, inputs, targets, optimizer, loss_fn, scaler=None):
    """
    Execute a single training step with memory efficiency optimizations
    
    Args:
        model: The model to train
        inputs: Input tensors
        targets: Target tensors
        optimizer: The optimizer to use
        loss_fn: Loss function
        scaler: GradScaler for mixed precision training (optional)
    
    Returns:
        loss value
    """
    # Clear gradients
    optimizer.zero_grad(set_to_none=True)  # More memory efficient than just zero_grad()
    
    if scaler is not None:  # Use mixed precision if scaler is provided
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
        # Scale the loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        # Standard training step
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Return loss value as Python float to avoid GPU memory retention
    return loss.detach().cpu().item()

def monitor_memory_usage(prefix=""):
    """Monitor and log GPU and system memory usage"""
    memory_info = []
    
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        memory_info.append(f"GPU: {gpu_allocated:.2f}GB alloc, {gpu_reserved:.2f}GB reserved")
    
    # System memory
    vm = psutil.virtual_memory()
    memory_info.append(f"RAM: {vm.used/1024**3:.2f}GB/{vm.total/1024**3:.2f}GB ({vm.percent}%)")
    
    # Log memory usage
    print(f"{prefix} Memory Usage: {' | '.join(memory_info)}")
    
    return gpu_allocated if torch.cuda.is_available() else 0

def auto_garbage_collection(threshold_gb=10.0):
    """Automatically trigger garbage collection when memory usage exceeds threshold"""
    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
        if gpu_mem_allocated > threshold_gb:
            print(f"Auto GC triggered - GPU memory: {gpu_mem_allocated:.2f}GB > {threshold_gb}GB threshold")
            cleanup_cuda_memory(force_gc=True)
            return True
    return False

def enable_gradients_for_inference(model, enable=False):
    """Control whether to track gradients during inference"""
    for param in model.parameters():
        param.requires_grad = enable
