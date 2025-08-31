from pathlib import Path
import os
import shutil
import string
from custom_logger import logger_config
import secrets
import hashlib
import random
import time
import re

def path_exists(path):
    return file_exists(path) or dir_exists(path)

def file_exists(file_path):
    try:
        return Path(file_path).is_file()
    except:
        pass
    return False

def dir_exists(file_path):
    try:
        return Path(file_path).is_dir()
    except:
        pass
    return False

def list_files_recursive(directory):
    remove_zone_identifier(directory)
    # Initialize an empty array to store the file paths
    file_list = []
    
    # Walk through the directory recursively
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Get the full path of the file and append to the array
            file_list.append(os.path.join(root, file))
    
    return file_list

def list_directories_recursive(directory):
    remove_zone_identifier(directory)
    # Initialize an empty list to store the directory names
    directory_list = []
    
    # Walk through the directory recursively
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            # Get the full path of the directory and append to the list
            directory_list.append(os.path.join(root, dir_name))
    
    return directory_list

def remove_zone_identifier(directory):
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(":Zone.Identifier"):
                    full_path = os.path.join(root, file)
                    remove_file(full_path)
    except: pass

def list_files(directory):
    remove_zone_identifier(directory)
    # Initialize an empty array to store the file paths
    file_list = []
    
    # Get the list of files in the given directory (non-recursive)
    for file in os.listdir(directory):
        # Construct the full path and check if it's a file
        full_path = os.path.join(directory, file)
        if os.path.isfile(full_path):
            file_list.append(full_path)
    
    return file_list

def remove_path(path):
    remove_file(path, True)
    remove_all_files_and_dirs(path)

def remove_file(file_path, retry=True):
    try:
        # Check if the file exists
        if os.path.exists(file_path):
            Path(file_path).unlink()
            logger_config.success(f"{file_path} has been removed successfully.")
    except Exception as e:
        logger_config.warning(f"Error occurred while trying to remove the file: {e}")
        if retry:
            logger_config.debug("retrying after 10 seconds", seconds=10)
            remove_file(file_path, False)

def remove_all_files_and_dirs(directory):
    try:
        shutil.rmtree(directory)  # Recursively delete a directory
    except Exception as e:
        logger_config.warning(f"Failed to delete {directory}. Reason: {e}")

def remove_directory(directory_path):
    try:
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
            logger_config.debug(f'Directory Deleted at: {directory_path}')
    except Exception as e:
        logger_config.warning(f'An error occurred: {e}')

def create_directory(directory_path):
    try:
        # Create the directory
        os.makedirs(directory_path, exist_ok=True)  # exist_ok=True avoids error if the dir already exists
        logger_config.debug(f'Directory created at: {directory_path}')
    except Exception as e:
        logger_config.error(f'An error occurred: {e}')

def get_files_count(directory_path):
    return len(os.listdir(directory_path))

def generate_random_string(length=10):
    characters = string.ascii_letters
    random_string = ''.join(secrets.choice(characters) for _ in range(length))
    return random_string

def generate_random_string_from_input(input_string, length=16):
    # Hash the input string to get a consistent value
    hash_object = hashlib.sha256(input_string.encode())
    hashed_string = hash_object.hexdigest()

    # Use the hash to seed the random number generator
    random.seed(hashed_string)

    # Generate a random string based on the seed
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))

    return random_string

def rename_file(current_name, new_name):
    try:
        # Rename the file
        os.rename(current_name, new_name)
        logger_config.success(f"File renamed from '{current_name}' to '{new_name}'")
    except Exception as e:
        logger_config.error(f"An error occurred: {e}")

def copy(source, dest):
    try:
        shutil.copy2(source, dest)
    except Exception as e:
        logger_config.error(f"An error occurred: {e}")

def get_media_metadata(file_path):
    try:
        import ffmpeg
        probe = ffmpeg.probe(file_path, v='error', select_streams='v:0', show_entries='format=duration,streams')

        # Duration in float seconds
        duration_in_sec_float = float(probe['format']['duration'])
        duration_in_sec_int = int(duration_in_sec_float)

        # File size in MB
        size = int(os.path.getsize(file_path) // (1024 * 1024))

        fps = None
        for stream in probe['streams']:
            if stream['codec_type'] == 'video':
                fps = eval(stream['r_frame_rate'])  # Frames per second (r_frame_rate is in format num/den)

        return duration_in_sec_int, duration_in_sec_float, size, fps
    except Exception as e:
        logger_config.error(f"Error retrieving media metadata: {e}")
        return None, None, None, None

def is_mostly_black(frame, black_threshold=20, percentage_threshold=0.9, sample_rate=10):
    """
    Fast black frame detection using pixel sampling.

    Args:
        frame: OpenCV BGR frame (NumPy array)
        black_threshold: grayscale value below which a pixel is considered black
        percentage_threshold: fraction of black pixels to consider frame mostly black
        sample_rate: sample every N-th pixel in both dimensions (higher = faster)
    Returns:
        True if mostly black, False otherwise
    """
    import cv2
    import numpy as np
    if frame is None or frame.size == 0:
        return True
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Sample pixels
    sampled = gray[::sample_rate, ::sample_rate]
    black_count = np.sum(sampled < black_threshold)
    total_count = sampled.size
    return (black_count / total_count) >= percentage_threshold

def only_alpha(text: str) -> str:
    # Keep only alphabetic characters (make lowercase to ignore case)
    return re.sub(r'[^a-zA-Z]', '', text).lower()

def is_same_sentence(sentence_1, sentence_2, threshold=0.9):
    # Clean both
    sentence_1 = only_alpha(sentence_1)
    sentence_2 = only_alpha(sentence_2)

    import difflib
    similarity = difflib.SequenceMatcher(None, sentence_1, sentence_2).ratio()
    logger_config.info(f"is_same_sentence :: similarity-{similarity}")
    return similarity > threshold

def manage_gpu(size_gb: float = 0, gpu_index: int = 0, action: str = "check"):
    """
    Manage GPU memory:
      - check       → just prints memory + process table
      - clear_cache → clears PyTorch cache
      - kill        → kills all GPU processes
    """
    try:
        import pynvml,signal, gc
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        free_gb = info.free / 1024**3
        total_gb = info.total / 1024**3

        print(f"\nGPU {gpu_index}: Free {free_gb:.2f} GB / Total {total_gb:.2f} GB")

        # Show processes
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        print("\nActive GPU Processes:")
        print(f"{'PID':<8} {'Process Name':<40} {'Used (GB)':<10}")
        print("-" * 60)
        for p in processes:
            used_gb = p.usedGpuMemory / 1024**3
            proc_name = pynvml.nvmlSystemGetProcessName(p.pid).decode(errors="ignore")
            print(f"{p.pid:<8} {proc_name:<40} {used_gb:.2f}")

        if action == "clear_cache":
            try:
                import torch
                gc.collect()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                time.sleep(1)
                print("\n🧹 Cleared PyTorch CUDA cache")
            except ImportError:
                print("\n⚠️ PyTorch not installed, cannot clear cache.")

        elif action == "kill":
            for p in processes:
                proc_name = pynvml.nvmlSystemGetProcessName(p.pid).decode(errors="ignore")
                try:
                    os.kill(p.pid, signal.SIGKILL)
                    print(f"❌ Killed {p.pid} ({proc_name})")
                except Exception as e:
                    print(f"⚠️ Could not kill {p.pid}: {e}")
            manage_gpu(action="clear_cache")
        gc.collect()
        gc.collect()
        return free_gb > size_gb
    except: return False

def is_gpu_available(verbose=True):
    import torch
    if not torch.cuda.is_available():
        if verbose:
            print("CUDA not available.")
        return False
    
    try:
        # Try a tiny allocation to check if GPU is free & usable
        torch.empty(1, device="cuda")
        if verbose:
            print(f"CUDA available. Using device: {torch.cuda.get_device_name(0)}")
        return True
    except RuntimeError as e:
        if "CUDA-capable device(s) is/are busy or unavailable" in str(e) or \
           "CUDA error" in str(e):
            if verbose:
                print("CUDA detected but busy/unavailable. Please CPU.")
            return False
        raise  # re-raise if it's some other unexpected error

def get_device(is_vision=False):
    if not is_vision and os.getenv("USE_CPU_IF_POSSIBLE", None):
        return "cpu"
    else:
        return "cuda" if is_gpu_available() else "cpu"