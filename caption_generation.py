import os
import json
from moondream2 import Moondream2
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from custom_logger import logger_config
import traceback
import torch
import threading
import time # Import time for skip logic
from dotenv import load_dotenv
import subprocess, sys

if os.path.exists(".env"):
	load_dotenv()

# Define a custom exception for handler failures
class HandlerSkippedException(Exception):
	pass

class MultiTypeCaptionGenerator:
	def __init__(self, cache_path, num_types=12, FYI="", local_only=False, skip_duration_seconds=100):
		self.cache_path = cache_path
		self.num_types = num_types
		self.lock = Lock()  # for safely updating temp JSON
		self.model_lock = Lock()
		self.handler_lock = Lock() # NEW: Lock for handler statuses
		self.model = None
		self.FYI = FYI
		self.local_only = local_only
		self.skip_duration = skip_duration_seconds # Duration to skip a failing handler

		# Thread-local storage for models
		self._thread_local = threading.local()

		# --- NEW: Handler Ranking/Skipping State ---
		self.handler_statuses = {
			i: {
				"is_skipped": False,
				"skip_until": 0,
				"failure_count": 0,
			} for i in range(num_types)
		}
		# -------------------------------------------

	def _get_thread_model(self):
		"""Get or create a model instance for the current thread"""
		if not hasattr(self._thread_local, 'model') or self._thread_local.model is None:
			with self.model_lock:
				thread_id = threading.current_thread().ident
				print(f"🔧 Thread {thread_id} initializing model...")
				try:
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
						torch.cuda.synchronize()
					
					os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
					os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
					
					self._thread_local.model = Moondream2()
					print(f"✅ Thread {threading.current_thread().ident} model initialized")
				except Exception as e:
					print(f"❌ Thread {thread_id} model init failed: {e}")
					self._thread_local.model = None
					raise
		
		return self._thread_local.model

	def _load_moondream2(self):
		"""Thread-safe model loading - now uses thread-local storage"""
		return self._get_thread_model()

	def _load_temp(self, temp_path):
		if os.path.exists(temp_path):
			with open(temp_path, "r") as f:
				return json.load(f)
		return [{"in_progress": False, "processed": False, "caption": None, "dialogue": None} 
				for _ in range(self.num_frames)]

	def _save_temp(self, temp_path, data):
		with self.lock:
			with open(temp_path, "w") as f:
				json.dump(data, f, indent=4)

	def _get_next_index(self, temp_path):
		"""Get next available index and mark it as in_progress atomically"""
		with self.lock:
			temp_data = self._load_temp(temp_path)
			
			for i, entry in enumerate(temp_data):
				if not entry["in_progress"] and not entry["processed"]:
					temp_data[i]["in_progress"] = True
					with open(temp_path, "w") as f:
						json.dump(temp_data, f, indent=4)
					print(f"🎯 Assigned index {i} to worker")
					return i, temp_data
			return None, temp_data

	def _worker(self, prompt, extract_scenes_json, temp_path, type_id):
		processed_count = 0
		thread_id = threading.current_thread().ident
		print(f"🚀 Worker {type_id} (Thread {thread_id}) started")
		
		if torch.cuda.is_available():
			try:
				device_id = type_id % torch.cuda.device_count()
				torch.cuda.set_device(device_id)
				print(f"📍 Worker {type_id} using GPU {device_id}")
			except Exception as e:
				print(f"⚠️ Could not set CUDA device for worker {type_id}: {e}")
		
		while True:
			print(f"🔍 Worker {type_id} looking for next frame...")
			result_tuple = self._get_next_index(temp_path)
			if result_tuple[0] is None:
				print(f"✋ Worker {type_id} - no more frames available")
				break
			
			idx, temp_data = result_tuple
			print(f"📋 Worker {type_id} got index {idx}")
			
			scene = extract_scenes_json[idx]
			frame_path = scene["frame_path"][0]
			dialogue = scene["dialogue"]

			print(f"⚡ Type {type_id} processing frame {idx+1}/{len(extract_scenes_json)}")

			try:
				result = None
				if type_id % self.num_types == 0 or self.local_only:
					model = self._load_moondream2()
					if model:
						result = model.generate(frame_path, prompt)
					else:
						print(f"❌ Worker {type_id} - model not available")
						result = None
				else:
					new_prompt = f"""{prompt} Also identify all the characters name in this frame. Keep your description to exactly 100 words or fewer.
{self.FYI}"""
					result = self.search_in_ui_type(type_id, new_prompt, frame_path)

				print(f"📝 Type {type_id} got result for frame {idx}: {result}")

				with self.lock:
					temp_data = self._load_temp(temp_path)
					
					if result:
						temp_data[idx]["caption"] = result.lower()
						temp_data[idx]["processed"] = True
						temp_data[idx]["dialogue"] = dialogue
						temp_data[idx]["frame_path"] = frame_path
						processed_count += 1
						print(f"✅ Type {type_id} completed frame {idx+1} (Total: {processed_count})")
					else:
						temp_data[idx]["processed"] = False
						print(f"❌ Type {type_id} failed frame {idx+1}")
					
					temp_data[idx]["in_progress"] = False
					with open(temp_path, "w") as f:
						json.dump(temp_data, f, indent=4)
			
			# --- MODIFIED: Handle handler skipping ---
			except HandlerSkippedException:
				print(f"🟡 Worker {type_id}'s handler is currently skipped. Releasing frame {idx} for another worker.")
				with self.lock:
					temp_data = self._load_temp(temp_path)
					temp_data[idx]["in_progress"] = False # Release the frame
					with open(temp_path, "w") as f:
						json.dump(temp_data, f, indent=4)
				time.sleep(5) # Brief pause to allow other workers to pick up
			# -------------------------------------------
			except Exception as e:
				print(f"💥 Worker {type_id} error on frame {idx}: {e}")
				
				if ("meta tensor" in str(e) or "Cannot copy out" in str(e) or 
					"meta is not on the expected device" in str(e) or 
					"CUDA" in str(e) or "device" in str(e).lower()):
					print(f"🔄 Worker {type_id} detected device/model error, clearing thread-local model")
					if hasattr(self._thread_local, 'model'):
						try:
							del self._thread_local.model
						except:
							pass
						self._thread_local.model = None
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
						torch.cuda.synchronize()
				
				with self.lock:
					temp_data = self._load_temp(temp_path)
					temp_data[idx]["in_progress"] = False
					temp_data[idx]["processed"] = False
					with open(temp_path, "w") as f:
						json.dump(temp_data, f, indent=4)
		
		if hasattr(self._thread_local, 'model') and self._thread_local.model is not None:
			print(f"🧹 Worker {type_id} cleaning up thread-local model")
			del self._thread_local.model
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		
		print(f"🏁 Worker {type_id} finished. Processed {processed_count} frames.")

	def caption_generation(self, extract_scenes_json):
		self.num_frames = len(extract_scenes_json)
		cache_dir = os.path.join(self.cache_path, "caption_generation.json")
		partial_dir = os.path.join(self.cache_path, "partial_captions")
		os.makedirs(partial_dir, exist_ok=True)
		temp_path = os.path.join(partial_dir, "temp_progress.json")

		if os.path.exists(cache_dir):
			logger_config.info(f"✅ Using cached captions for {len(extract_scenes_json)} frames")
			with open(cache_dir, "r") as f:
				return json.load(f)

		logger_config.info(f"🚀 Starting multi-type caption generation for {len(extract_scenes_json)} frames")
		
		if not os.path.exists(temp_path):
			initial_data = [
				{
					"in_progress": False, 
					"processed": False, 
					"caption": None, 
					"dialogue": extract_scenes_json[i]["dialogue"],
					"frame_path": extract_scenes_json[i]["frame_path"][0]
				} 
				for i in range(self.num_frames)
			]
			self._save_temp(temp_path, initial_data)
		else:
			temp_data = self._load_temp(temp_path)
			
			if len(temp_data) != self.num_frames:
				logger_config.warning(f"⚠️  Temp file has {len(temp_data)} frames, expected {self.num_frames}. Reinitializing...")
				initial_data = [
					{
						"in_progress": False, 
						"processed": False, 
						"caption": None, 
						"dialogue": extract_scenes_json[i]["dialogue"],
						"frame_path": extract_scenes_json[i]["frame_path"][0]
					} 
					for i in range(self.num_frames)
				]
				self._save_temp(temp_path, initial_data)
			else:
				completed_count = 0
				reset_count = 0
				
				for i, data in enumerate(temp_data):
					if data["in_progress"]:
						data["in_progress"] = False
						data["processed"] = False
						data["caption"] = None
						reset_count += 1
					
					if data["processed"] and data["caption"]:
						completed_count += 1
					
					if i < len(extract_scenes_json):
						data["dialogue"] = extract_scenes_json[i]["dialogue"]
				
				self._save_temp(temp_path, temp_data)
				logger_config.info(f"📋 Resuming: {completed_count} completed, {reset_count} reset from in_progress")

		prompt = "Describe what is happening in this video frame as if you're telling a story. Focus on the main subjects, their actions, the setting, and any important details."

		effective_workers = 1 if self.local_only else self.num_types
		print(f"🔧 Using {effective_workers} workers (local_only: {self.local_only})")
		if self.local_only:
			print(f"💾 Memory optimization: Using 1 thread to avoid loading multiple models")

		with ThreadPoolExecutor(max_workers=effective_workers) as executor:
			futures = []
			for type_id in range(effective_workers):
				future = executor.submit(self._worker, prompt, extract_scenes_json, temp_path, type_id)
				futures.append(future)
			
			for future in futures:
				try:
					future.result()
				except Exception as e:
					logger_config.error(f"Worker failed with error: {e}")

		temp_data = self._load_temp(temp_path)
		captions = []

		for i, entry in enumerate(temp_data):
			captions.append({
				"scene_caption": entry["caption"], 
				"scene_dialogue": entry["dialogue"]
			})

		with open(cache_dir, "w") as f:
			json.dump(captions, f, indent=4)
			
		logger_config.info(f"✅ Saved all captions to {cache_dir}")
		
		if hasattr(self, 'model') and self.model:
			del self.model
		if hasattr(self, '_thread_local') and self._thread_local:
			del self._thread_local
			
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		
		return captions

	def search_in_ui_type(self, type_id, prompt, file_path):
		from chat_bot_ui_handler import GoogleAISearchChat, AIStudioUIChat, QwenUIChat, PerplexityUIChat, GeminiUIChat, GrokUIChat, MetaUIChat, CopilotUIChat, BingUIChat, MistralUIChat, PallyUIChat
		from browser_manager.browser_config import BrowserConfig
		import os
		
		handler_key = type_id % self.num_types
		
		# --- NEW: Check handler status before proceeding ---
		with self.handler_lock:
			status = self.handler_statuses[handler_key]
			if status["is_skipped"] and time.time() < status["skip_until"]:
				logger_config.warning(f"Handler {handler_key} is currently skipped. Skipping this task.")
				raise HandlerSkippedException(f"Handler {handler_key} is skipped.")
			# If skip time has passed, reset the status
			elif status["is_skipped"]:
				logger_config.info(f"Reactivating handler {handler_key} after skip period.")
				status["is_skipped"] = False
				status["failure_count"] = 0
		# ----------------------------------------------------

		sources = [GoogleAISearchChat, AIStudioUIChat, QwenUIChat, PerplexityUIChat, GeminiUIChat, GrokUIChat, MetaUIChat, CopilotUIChat, BingUIChat, MistralUIChat, PallyUIChat]
		source = sources[handler_key-1]

		try:
			config = BrowserConfig()
			config.starting_server_port_to_check = [9081, 10081, 11081, 12081, 13081, 14081, 15081, 16081, 17081, 18081, 19081][handler_key-1]
			config.starting_debug_port_to_check = [10224, 11224, 12224, 13224, 14224, 15224, 16224, 17224, 18224, 19224, 20224][handler_key-1]

			if source.__name__ == "GrokUIChat" or source.__name__ == "PerplexityUIChat":
				config.user_data_dir = os.getenv("PROFILE_PATH_1")
			else:
				config.user_data_dir = os.getenv("PROFILE_PATH")

			if source.__name__ == "MetaUIChat" or source.__name__ == "AIStudioUIChat":
				additional_flags = []
				additional_flags.append(f'-v {os.getcwd()}/{os.getenv("TEMP_OUTPUT", "chat_bot_ui_handler_logs")}:/home/neko/Downloads')
				additional_flags.append(f'-v {os.getenv("PARENT_BASE_PATH")}/browser_manager/policies.json:/etc/opt/chrome/policies/managed/policies.json')
				config.additionl_docker_flag = ' '.join(additional_flags)

			src_obj = source(config=config)
			result = src_obj.chat(user_prompt=prompt, file_path=file_path)

			if result and len(result.split(" ")) > 40:
				if "AI responses may include mistakes" in result:
					result = result[:result.index("AI responses may include mistakes")]
				if "Sources\nhelp" in result:
					result = result[:result.index("Sources\nhelp\n")]

				# --- NEW: Reset failure count on success ---
				with self.handler_lock:
					self.handler_statuses[handler_key]["failure_count"] = 0
				# ---------------------------------------------
				return result

		except Exception as e:
			logger_config.error(f"Type {type_id} (Handler {handler_key}) failed processing: {e}")
			# --- NEW: Penalize the handler on failure ---
			with self.handler_lock:
				status = self.handler_statuses[handler_key]
				status["failure_count"] += 1
				# Skip the handler after 3 consecutive failures
				if status["failure_count"] >= 3:
					status["is_skipped"] = True
					status["skip_until"] = time.time() + self.skip_duration
					logger_config.critical(f"Handler {handler_key} failed {status['failure_count']} times. Skipping for {self.skip_duration} seconds.")
			# --------------------------------------------

		return None

# Usage example and main execution
if __name__ == "__main__":
    print(sys.argv)
    extract_scenes_path = sys.argv[1]
    cache_path = sys.argv[2]
    
    # Set default values for FYI and local_only
    FYI = ""
    if len(sys.argv) > 3:
        FYI = sys.argv[3]

    local_only = False
    if len(sys.argv) > 4:
        # Convert the string argument to a boolean
        local_only = sys.argv[4].lower() in ('true', '1', 't')

    # Example usage
    captionGen = MultiTypeCaptionGenerator(cache_path=cache_path, FYI=FYI, local_only=local_only)

    with open(extract_scenes_path, "r") as f:
        data = json.load(f)

    results = captionGen.caption_generation(data)
    print(results)