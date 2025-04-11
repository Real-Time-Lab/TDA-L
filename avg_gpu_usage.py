import time
import sys

try:
    import pynvml
except ImportError:
    print("pynvml is not installed. Please install it with: pip install nvidia-ml-py3")
    sys.exit(1)

def main(duration=300, gpu_index=0, sample_interval=1):
    """
    Measure the average GPU usage and memory usage for a given duration.

    Args:
        duration (int): Total time to sample in seconds (default: 120 seconds).
        gpu_index (int): GPU index to monitor (default: 0).
        sample_interval (int): Time interval between samples in seconds (default: 1 second).
    """
    # Initialize NVML
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    except pynvml.NVMLError as e:
        print(f"Error accessing GPU {gpu_index}: {e}")
        pynvml.nvmlShutdown()
        sys.exit(1)

    gpu_usages = []
    memory_usages = []
    
    start_time = time.time()
    print(f"Sampling GPU and Memory usage for {duration} seconds...")

    while time.time() - start_time < duration:
        # Get GPU compute utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpu_usages.append(utilization)

        # Get memory utilization (used memory in MB)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_usage = memory_info.used / (1024 ** 2)  # Convert to MB
        memory_usages.append(memory_usage)

        time.sleep(sample_interval)

    # Compute averages
    avg_gpu_usage = sum(gpu_usages) / len(gpu_usages)
    avg_memory_usage = sum(memory_usages) / len(memory_usages)

    print(f"Average GPU compute usage over {duration} seconds: {avg_gpu_usage:.2f}%")
    print(f"Average GPU memory usage over {duration} seconds: {avg_memory_usage:.2f} MB")

    # Shutdown NVML
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
