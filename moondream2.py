from vision_model import VisionModel
from PIL import Image
import time
import torch
import os
import common

class Moondream2(VisionModel):
    def load_model(self):
        # Set environment variables to prevent meta device issues
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Clear CUDA cache first
        if common.is_gpu_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Try multiple loading strategies
        strategies = [
            self._load_strategy_cpu_first,
            self._load_strategy_explicit_device,
            self._load_strategy_cpu_only
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                print(f"🔧 Attempting loading strategy {i+1}/{len(strategies)}")
                self.model = strategy()
                
                # Verify model is properly loaded and not on meta device
                self._verify_model_device()
                print(f"✅ Strategy {i+1} successful!")
                return
                
            except Exception as e:
                print(f"❌ Strategy {i+1} failed: {e}")
                if hasattr(self, 'model'):
                    try:
                        del self.model
                    except:
                        pass
                if common.is_gpu_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                if i == len(strategies) - 1:  # Last strategy
                    raise Exception(f"All loading strategies failed. Last error: {e}")
    
    def _verify_model_device(self):
        """Verify that model parameters are not on meta device"""
        if not hasattr(self, 'model') or self.model is None:
            raise Exception("Model is None")
        
        # Check for meta device parameters
        meta_params = []
        for name, param in self.model.named_parameters():
            if param.device.type == 'meta':
                meta_params.append(name)
        
        if meta_params:
            raise Exception(f"Found {len(meta_params)} parameters still on meta device: {meta_params[:5]}...")
        
        print(f"✅ Model verification passed - all parameters on proper device")
    
    def _load_strategy_cpu_first(self):
        """Strategy 1: Load on CPU first, then move to GPU"""
        from transformers import AutoModelForCausalLM
        
        print("🔧 Loading on CPU first...")
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-06-21",
            trust_remote_code=True,
            device_map=None,  # Force CPU
            torch_dtype=torch.float32,  # CPU compatible
            low_cpu_mem_usage=True
        )
        
        # Move to GPU if available
        if common.is_gpu_available():
            print("🔧 Moving to GPU...")
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            model = model.to(device, dtype=torch.bfloat16)
            print("✅ Successfully moved to GPU")
        
        return model
    
    def _load_strategy_explicit_device(self):
        """Strategy 2: Load with explicit device mapping"""
        from transformers import AutoModelForCausalLM
        
        if common.is_gpu_available():
            device_map = f"cuda:{torch.cuda.current_device()}"
            dtype = torch.bfloat16
        else:
            device_map = "cpu"
            dtype = torch.float32
        
        print(f"🔧 Loading with explicit device_map: {device_map}")
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-06-21",
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        return model
    
    def _load_strategy_cpu_only(self):
        """Strategy 3: CPU-only fallback"""
        from transformers import AutoModelForCausalLM
        
        print("🔧 Loading CPU-only model...")
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-06-21",
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        return model

    def generate(self, image: Image.Image, text: str = "") -> str:
        try:
            with torch.inference_mode():
                if isinstance(image, str):
                    with Image.open(image) as img:
                        result = self.model.caption(img, length="normal")
                        return result["caption"] if isinstance(result, dict) else str(result)
                
                result = self.model.caption(image, length="normal")
                return result["caption"] if isinstance(result, dict) else str(result)
                
        except RuntimeError as e:
            if "meta" in str(e).lower() or "device" in str(e).lower():
                print(f"🔄 Device error in generate, attempting model reload: {e}")
                self.load_model()  # Reload model
                return self.generate(image, text)  # Retry once

    def query(self, image: Image.Image, text: str = "") -> str:
        with torch.inference_mode():
            result = self.model.query(image, text)
            return result["answer"] if isinstance(result, dict) else str(result)

    def point(self, image: Image.Image, text: str = "person") -> str:
        with torch.inference_mode():
            result = self.model.point(image, text)
            return result["points"] if isinstance(result, dict) else str(result)

    def detect(self, image: Image.Image, text: str = "face") -> str:
        with torch.inference_mode():
            result = self.model.detect(image, text)
            return result["objects"] if isinstance(result, dict) else str(result)

if __name__ == "__main__":
    model = Moondream2()
    image = Image.open("temp_dir/page-022.jpg_resize_frame__max_frame/sentence_12_frame_1.jpg")
    text = "Describe what is happening in this video frame as if you're telling a story. Focus on the main subjects, their actions, the setting, and any important details that would help someone understand the scene's context."
    start_time = time.time()
    response = model.generate(image)
    end_time = time.time()
    print("Moondream2:", response)
    print(f"⏱ Total time taken: {end_time - start_time:.2f} seconds")