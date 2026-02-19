from jebin_lib import load_env
load_env()

import os
import json
from moondream2 import Moondream2
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from custom_logger import logger_config
import common
import torch
import threading
import time # Import time for skip logic
import sys
from chat_bot_ui_handler import GoogleAISearchChat, AIStudioUIChat, QwenUIChat, PerplexityUIChat, GeminiUIChat, GrokUIChat, MetaUIChat, CopilotUIChat, BingUIChat, MistralUIChat, PallyUIChat, MoonDream

# Serialize logger output so multi-line entries (message + separator) from
# different threads never interleave.
_log_lock = Lock()

def _log(level, msg):
	with _log_lock:
		getattr(logger_config, level)(msg)

# Define a custom exception for handler failures
class HandlerSkippedException(Exception):
	pass

class MultiTypeCaptionGenerator:
	def __init__(self, cache_path, num_types=12, FYI="", local_only=False, skip_duration_seconds=100):
		self.cache_path = cache_path
		self.sources = [GoogleAISearchChat, QwenUIChat, PerplexityUIChat, GeminiUIChat, MoonDream, GrokUIChat, MetaUIChat, CopilotUIChat, BingUIChat, MistralUIChat, PallyUIChat, GeminiUIChat, GoogleAISearchChat, BingUIChat, GeminiUIChat, GoogleAISearchChat, BingUIChat]
		self.num_types = len(self.sources) + 1
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
			} for i in range(len(self.sources))
		}
		# -------------------------------------------

	def _get_thread_model(self):
		"""Get or create a model instance for the current thread"""
		if not hasattr(self._thread_local, 'model') or self._thread_local.model is None:
			with self.model_lock:
				thread_id = threading.current_thread().ident
				_log('info', f"[Thread {thread_id}] Initializing model...")
				try:
					if common.is_gpu_available():
						torch.cuda.empty_cache()
						torch.cuda.synchronize()

					os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
					os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

					self._thread_local.model = Moondream2()
					_log('success', f"[Thread {thread_id}] Model initialized")
				except Exception as e:
					_log('error', f"[Thread {thread_id}] Model init failed: {e}")
					self._thread_local.model = None
					raise

		return self._thread_local.model

	def _load_moondream2(self):
		"""Thread-safe model loading - now uses thread-local storage"""
		return self._get_thread_model()

	def _load_temp(self, temp_path):
		if os.path.exists(temp_path):
			with open(temp_path, "r") as f:
				try:
					return json.load(f)
				except: pass
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
			progress_start_time_timeout = 60 * 1 * 10  # e.g., 10 minutes

			released = False
			for i, entry in enumerate(temp_data):
				if entry["in_progress"]:
					if entry.get("progress_start_time") and (time.time() - entry["progress_start_time"]) > progress_start_time_timeout:
						entry["in_progress"] = False
						entry["progress_start_time"] = None
						released = True
						_log('warning', f"[Index {i}] Releasing stale in-progress frame")

			if released:
				with open(temp_path, "w") as f:
					json.dump(temp_data, f, indent=4)

			for i, entry in enumerate(temp_data):
				if not entry["in_progress"] and not entry["processed"]:
					temp_data[i]["in_progress"] = True
					temp_data[i]["progress_start_time"] = time.time()  # Add timestamp
					with open(temp_path, "w") as f:
						json.dump(temp_data, f, indent=4)
					return i, temp_data
			return None, temp_data

	def _worker(self, prompt, extract_scenes_json, temp_path, type_id):
		processed_count = 0
		thread_id = threading.current_thread().ident
		handler_key = type_id if self.local_only else ((type_id - 1) % len(self.sources))
		_log('info', f"[W{type_id}|H{handler_key}] Started on thread {thread_id}")

		if common.is_gpu_available() and not self.local_only:
			try:
				device_id = type_id % torch.cuda.device_count()
				torch.cuda.set_device(device_id)
				_log('info', f"[W{type_id}] Using GPU {device_id}")
			except Exception as e:
				_log('warning', f"[W{type_id}] Could not set CUDA device: {e}")
		
		while True:
			# --- NEW: Proactively check if the handler is skipped and pause the worker ---
			if not self.local_only: # This logic only applies to non-local handlers
				skip_time = None
				with self.handler_lock:
					status = self.handler_statuses[handler_key]
					if status["is_skipped"]:
						remaining_skip_time = status["skip_until"] - time.time()
						if remaining_skip_time > 0:
							skip_time = remaining_skip_time
						else:
							# If time is up, reactivate the handler
							_log('info', f"[H{handler_key}] Reactivating after skip period")
							status["is_skipped"] = False
							status["failure_count"] = 0

				# 🔑 do the waiting *after* releasing the lock
				if skip_time:
					_log('warning', f"[W{type_id}|H{handler_key}] Paused — handler skipped for {skip_time:.0f}s more")
					time.sleep(skip_time + 1)
					continue

			result_tuple = self._get_next_index(temp_path)
			if result_tuple[0] is None:
				with self.lock:
					temp_data = self._load_temp(temp_path)
				is_all_processed = all(entry["processed"] for entry in temp_data)
				if is_all_processed:
					_log('success', f"[W{type_id}] All frames processed. Exiting.")
					break
				else:
					time.sleep(5)
					continue

			idx, temp_data = result_tuple

			scene = extract_scenes_json[idx]
			frame_path = scene["frame_path"][0]
			dialogue = scene["dialogue"]

			_log('info', f"[W{type_id}] Processing frame {idx+1}/{len(extract_scenes_json)}")

			try:
				result = None
				if common.get_device() == "cuda" and (type_id % self.num_types == 0 or self.local_only):
					model = self._load_moondream2()
					if model:
						result = model.generate(frame_path, prompt)
					else:
						_log('error', f"[W{type_id}] Model not available")
						result = None
				else:
					new_prompt = f"""{prompt} Also identify all the characters name in this frame. Keep your description to exactly 100 words or fewer.
	{self.FYI}"""
					result = self.search_in_ui_type(type_id, new_prompt, frame_path, thread_id)


				with self.lock:
					temp_data = self._load_temp(temp_path)
					
					if result:
						temp_data[idx]["caption"] = result.lower()
						temp_data[idx]["processed"] = True
						temp_data[idx]["dialogue"] = dialogue
						temp_data[idx]["frame_path"] = frame_path
						processed_count += 1
						_log('success', f"[W{type_id}] Frame {idx+1}/{len(extract_scenes_json)} done (session total: {processed_count})")
					else:
						temp_data[idx]["processed"] = False
						_log('error', f"[W{type_id}] Frame {idx+1} returned no result")
					
					temp_data[idx]["in_progress"] = False
					with open(temp_path, "w") as f:
						json.dump(temp_data, f, indent=4)
			
			except HandlerSkippedException:
				# This block is now a secondary safeguard. The proactive check above should prevent it.
				_log('warning', f"[W{type_id}] Handler skipped — releasing frame {idx}")
				with self.lock:
					temp_data = self._load_temp(temp_path)
					temp_data[idx]["in_progress"] = False # Release the frame
					with open(temp_path, "w") as f:
						json.dump(temp_data, f, indent=4)
				time.sleep(5) # Brief pause to prevent rapid re-grabbing if the proactive check fails
			
			except Exception as e:
				_log('error', f"[W{type_id}] Error on frame {idx}: {e}")
				
				if ("meta tensor" in str(e) or "Cannot copy out" in str(e) or 
					"meta is not on the expected device" in str(e) or 
					"CUDA" in str(e) or "device" in str(e).lower()):
					_log('warning', f"[W{type_id}] Device/model error detected — clearing thread-local model")
					if hasattr(self._thread_local, 'model'):
						try:
							del self._thread_local.model
						except:
							pass
						self._thread_local.model = None
					if common.is_gpu_available():
						torch.cuda.empty_cache()
						torch.cuda.synchronize()
				
				with self.lock:
					temp_data = self._load_temp(temp_path)
					temp_data[idx]["in_progress"] = False
					temp_data[idx]["processed"] = False
					with open(temp_path, "w") as f:
						json.dump(temp_data, f, indent=4)
		
		if hasattr(self._thread_local, 'model') and self._thread_local.model is not None:
			_log('info', f"[W{type_id}] Cleaning up thread-local model")
			del self._thread_local.model
			if common.is_gpu_available():
				torch.cuda.empty_cache()
		
		if hasattr(self._thread_local, 'handler') and self._thread_local.handler is not None:
			_log('info', f"[W{type_id}] Cleaning up thread-local handler")
			try:
				self._thread_local.handler.cleanup()
			except:
				pass
			self._thread_local.handler = None

		_log('info', f"[W{type_id}] Finished — processed {processed_count} frames this session")

	def caption_generation(self, extract_scenes_json):
		self.num_frames = len(extract_scenes_json)
		cache_dir = os.path.join(self.cache_path, "caption_generation.json")
		partial_dir = os.path.join(self.cache_path, "partial_captions")
		os.makedirs(partial_dir, exist_ok=True)
		temp_path = os.path.join(partial_dir, "temp_progress.json")
		temp_data = self._load_temp(temp_path)
		if len([cap for cap in temp_data if not cap["processed"]]) != 0:
			common.remove_file(cache_dir)

		if os.path.exists(cache_dir):
			_log('info', f"Using cached captions for {len(extract_scenes_json)} frames")
			with open(cache_dir, "r") as f:
				return json.load(f)

		_log('info', f"Starting caption generation for {len(extract_scenes_json)} frames")
		
		if not os.path.exists(temp_path):
			initial_data = [
				{
					"in_progress": False, 
					"processed": False, 
					"caption": None, 
					"dialogue": extract_scenes_json[i]["dialogue"],
					"frame_path": extract_scenes_json[i]["frame_path"][0],
					"progress_start_time": None
				} 
				for i in range(self.num_frames)
			]
			self._save_temp(temp_path, initial_data)
		else:
			# If the temp file exists, we are resuming.
			temp_data = self._load_temp(temp_path)
			
			# Ensure the temp file length matches the input scenes.
			if len(temp_data) != self.num_frames:
				_log('warning', f"Temp file has {len(temp_data)} frames, expected {self.num_frames} — reinitializing")
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
				# *** START OF THE FIX ***
				# Reset any "in_progress" flags from a previous crashed run.
				completed_count = 0
				reset_count = 0
				
				for i, data in enumerate(temp_data):
					if data["in_progress"]:
						data["in_progress"] = False # Reset the flag
						data["processed"] = False # Ensure it's not marked as processed
						data["caption"] = None    # Clear any partial caption
						reset_count += 1
					
					if data["processed"] and data["caption"]:
						completed_count += 1
					
					# Always update dialogue and frame_path in case the source changed
					if i < len(extract_scenes_json):
						data["dialogue"] = extract_scenes_json[i]["dialogue"]
						data["frame_path"] = extract_scenes_json[i]["frame_path"][0]

				if reset_count > 0:
					_log('info', f"Reset {reset_count} stale in-progress frames from previous run")

				self._save_temp(temp_path, temp_data)
				_log('info', f"Resuming — {completed_count}/{self.num_frames} frames already done")
				# *** END OF THE FIX ***

		prompt = "Describe what is happening in this video frame as if you're telling a story. Focus on the main subjects, their actions, the setting, and any important details."

		effective_workers = 1 if self.local_only else self.num_types
		_log('info', f"Launching {effective_workers} worker(s) (local_only={self.local_only})")
		if self.local_only:
			_log('info', "Memory optimization: single thread to avoid loading multiple models")

		with ThreadPoolExecutor(max_workers=effective_workers) as executor:
			futures = []
			for type_id in range(effective_workers):
				if type_id == 0: continue
				future = executor.submit(self._worker, prompt, extract_scenes_json, temp_path, type_id)
				futures.append(future)
			
			for future in futures:
				try:
					future.result()
				except Exception as e:
					_log('error', f"Worker failed: {e}")
				except KeyboardInterrupt:
					_log('warning', "Ctrl+C detected — attempting graceful shutdown")
					# ThreadPoolExecutor does not automatically kill threads
					for f in futures:
						f.cancel()

		temp_data = self._load_temp(temp_path)
		if len([cap for cap in temp_data if not cap["processed"]]) != 0:
			raise ValueError("Exited without completed.")
		captions = []

		for i, entry in enumerate(temp_data):
			captions.append({
				"scene_caption": entry["caption"], 
				"scene_dialogue": entry["dialogue"]
			})

		with open(cache_dir, "w") as f:
			json.dump(captions, f, indent=4)
			
		_log('success', f"All captions saved to {cache_dir}")
		
		if hasattr(self, 'model') and self.model:
			del self.model
		if hasattr(self, '_thread_local') and self._thread_local:
			del self._thread_local
			
		if common.is_gpu_available():
			torch.cuda.empty_cache()
		
		return captions

	def search_in_ui_type(self, type_id, prompt, file_path, thread_id):
		from browser_manager.browser_config import BrowserConfig
		import os
		
		# Fix: Use consistent handler indexing
		handler_key = (type_id - 1) % len(self.sources)
		
		# --- Check handler status before proceeding ---
		with self.handler_lock:
			status = self.handler_statuses[handler_key]
			if status["is_skipped"] and time.time() < status["skip_until"]:
				_log('warning', f"[W{type_id}|H{handler_key}] Handler skipped — skipping task")
				raise HandlerSkippedException(f"Handler {handler_key} is skipped.")
			# If skip time has passed, reset the status
			elif status["is_skipped"]:
				_log('info', f"[H{handler_key}] Reactivating after skip period")
				status["is_skipped"] = False
				status["failure_count"] = 0

		
		source = self.sources[handler_key]

		import asyncio as _asyncio
		import subprocess as _sp

		try:
			if not hasattr(self._thread_local, 'handler'):
				self._thread_local.handler = None

			# Initialize handler if not present or if it was cleaned up after a failure
			if self._thread_local.handler is None or not isinstance(self._thread_local.handler, source):
				if self._thread_local.handler:
					try: self._thread_local.handler.cleanup()
					except: pass

				# Only reset asyncio running-loop state when we're about to start a NEW
				# playwright instance. Resetting it while an existing playwright session
				# is active (dispatcher greenlet running) would corrupt the thread-local
				# and risk confusing asyncio internals mid-session.
				try:
					_asyncio.events._set_running_loop(None)
				except Exception:
					pass

				docker_name = f"thread_id_{thread_id}"
				# Kill any stale container from a previous failed run with the same name
				# to free its ports before we try to allocate new ones.
				_sp.run(["docker", "rm", "-f", docker_name],
						stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)

				config = BrowserConfig()
				config.docker_name = docker_name
				neko_file_path = file_path

				if source.__name__ == "MetaUIChat" or source.__name__ == "AIStudioUIChat":
					neko_base_path = "/".join(file_path.split("/")[:5])
					neko_file_path = "/".join(file_path.split("/")[5:])
					# Set up additional docker flags
					config.additionl_docker_flag = ' '.join(common.get_neko_additional_flags(neko_base_path, config))
				
				self._thread_local.handler = source(config=config)
			
			# Update path for current call if needed (for relative path logic)
			neko_file_path = file_path
			if source.__name__ == "MetaUIChat" or source.__name__ == "AIStudioUIChat":
				neko_file_path = "/".join(file_path.split("/")[5:])

			src_obj = self._thread_local.handler
			result = src_obj.chat_fresh(user_prompt=prompt, file_path=neko_file_path)
	
			if not result or len(result.split(" ")) <= 40:
				raise ValueError(f"Handler {handler_key} returned an invalid or empty result.")

			if "AI responses may include mistakes" in result:
				result = result[:result.index("AI responses may include mistakes")]
			if "Sources\nhelp" in result:
				result = result[:result.index("Sources\nhelp\n")]

			# Reset failure count on success
			with self.handler_lock:
				self.handler_statuses[handler_key]["failure_count"] = 0
			return result

		except Exception as e:
			_log('error', f"[W{type_id}|H{handler_key}] Failed: {e}")
			
			# Force cleanup on error to ensure clean slate for next attempt
			if hasattr(self._thread_local, 'handler') and self._thread_local.handler:
				try: self._thread_local.handler.cleanup()
				except: pass
				self._thread_local.handler = None

			# Penalize the handler on failure
			with self.handler_lock:
				status = self.handler_statuses[handler_key]
				status["failure_count"] += 1
				# Skip the handler after 3 consecutive failures
				if status["failure_count"] >= 3:
					status["is_skipped"] = True
					status["skip_until"] = time.time() + self.skip_duration
					_log('warning', f"[H{handler_key}] Failed {status['failure_count']} times — skipping for {self.skip_duration}s")
			
			# Re-raise the exception so the worker can handle it appropriately
			raise

		return None

# Usage example and main execution
if __name__ == "__main__":
	import signal as _signal

	def _exit_handler(signum, frame):
		# os._exit skips Python cleanup (including ThreadPoolExecutor.shutdown(wait=True))
		# so worker threads — which may be blocked on Docker/browser ops — don't hold
		# the process alive. Exit code 130 = 128 + SIGINT(2), the Unix convention.
		os._exit(130)

	_signal.signal(_signal.SIGINT,  _exit_handler)
	_signal.signal(_signal.SIGTERM, _exit_handler)

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