import os
import json
from custom_logger import logger_config
from typing import List, Tuple
from scenedetect import detect, AdaptiveDetector
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from llm_scene_extract import run_transnetv2


def variance_of_laplacian(image):
	"""Blur detection using Laplacian variance."""
	return cv2.Laplacian(image, cv2.CV_64F).var()


def is_mostly_black_fast(image: Image.Image, black_threshold=20, percentage_threshold=0.9):
	"""Fast black frame detection"""
	width, height = image.size
	if width > 200 or height > 200:
		scale = min(200 / width, 200 / height)
		new_size = (int(width * scale), int(height * scale))
		image = image.resize(new_size, Image.LANCZOS)

	grayscale_image = image.convert('L')
	pixels = np.array(grayscale_image)

	black_pixel_count = np.sum(pixels < black_threshold)
	total_pixels = pixels.size

	black_percentage = black_pixel_count / total_pixels
	return black_percentage >= percentage_threshold


def extract_sharpest_scene_frame(cap, scene_start: float, scene_end: float, fps: float,
								  frames_dir: str, frame_index: int) -> Tuple[str, float]:
	"""
	Extracts the sharpest non-black frame within the scene.
	Returns (frame_path or None, best_timestamp)
	"""
	start_frame = int(scene_start * fps)
	end_frame = int(scene_end * fps)
	step = max(1, (end_frame - start_frame) // 5)  # Sample max 5 frames

	best_var = -1
	best_frame = None
	best_time = scene_start

	for f in range(start_frame, end_frame, step):
		cap.set(cv2.CAP_PROP_POS_FRAMES, f)
		ret, frame = cap.read()
		if not ret:
			continue

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		var = variance_of_laplacian(gray)

		# Save temporarily to check for black frame
		tmp_path = os.path.join(frames_dir, f"_tmp_scene_{frame_index}.jpg")
		cv2.imwrite(tmp_path, frame)
		with Image.open(tmp_path) as img:
			if is_mostly_black_fast(img):
				continue

		if var > best_var:
			best_var = var
			best_frame = frame.copy()
			best_time = f / fps

	# Clean up temp file
	if os.path.exists(tmp_path):
		os.remove(tmp_path)

	if best_frame is not None:
		frame_filename = f"scene_{frame_index:04d}_at_{best_time:.2f}s.jpg"
		frame_path = os.path.join(frames_dir, frame_filename)
		cv2.imwrite(frame_path, best_frame)
		return frame_path, best_time
	else:
		logger_config.warning(f"Failed to extract sharp non-black frame for scene {scene_start:.2f}-{scene_end:.2f}s")
		return None, best_time


def map_dialogues_to_scenes(scene_list: List[Tuple[float, float]], dialogues: List[dict],
							video_path: str, frames_dir: str) -> List[dict]:
	"""
	Map each dialogue to its corresponding scene and save a sharp non-black frame.
	"""
	os.makedirs(frames_dir, exist_ok=True)
	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)

	frames_extracted = 0
	scene_dialogue_map = []

	for i, (scene_start, scene_end) in tqdm(enumerate(scene_list), total=len(scene_list), desc="Processing scenes"):
		frame_path, best_time = extract_sharpest_scene_frame(cap, scene_start, scene_end, fps, frames_dir, frames_extracted)

		if frame_path:
			scene_dialogues = [
				d for d in dialogues
				if d['end'] >= scene_start and d['start'] <= scene_end
			]

			scene_dialogue_map.append({
				"scene_start": scene_start,
				"scene_end": scene_end,
				"best_time": best_time,
				"frame_path": [frame_path],
				"dialogues": scene_dialogues,
				"dialogue": ""
			})

			frames_extracted += 1

	cap.release()
	return scene_dialogue_map


def combine_consecutive_same_dialogues(scene_map: List[dict]) -> List[dict]:
	if not scene_map:
		return []

	def are_dialogues_equal(d1, d2):
		return json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)

	combined = []
	current = scene_map[0].copy()

	for next_scene in scene_map[1:]:
		next_dialogues = next_scene["dialogues"]
		current_dialogues = current["dialogues"]

		if are_dialogues_equal(current_dialogues, next_dialogues):
			current["scene_end"] = next_scene["scene_end"]
			current["frame_path"].extend(next_scene["frame_path"])
		else:
			combined.append(current)
			current = next_scene.copy()

	combined.append(current)
	return combined


def combine_dialogues(scene_map: List[dict]) -> List[dict]:
	if not scene_map:
		return []

	combined = []
	for current in scene_map:
		combined.append(current)
		combined[-1]["dialogue"] = " ".join([dia["text"] for dia in current["dialogues"]])

	return combined


def detect_scenes(video_path: str, frame_timestamps: List[float], threshold: float = 30.0) -> List[Tuple[float, float]]:
	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	cap.release()

	min_scene_len = 15
	scene_list = detect(
		video_path,
		AdaptiveDetector(min_scene_len=min_scene_len),
		show_progress=True
	)

	all_scenes = [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

	if frame_timestamps:
		matched_scenes = []
		for ts in frame_timestamps:
			closest_scene = min(all_scenes, key=lambda scene: min(abs(ts - scene[0]), abs(ts - scene[1])))
			matched_scenes.append(closest_scene)

		matched_scenes = list(set(matched_scenes))
	else:
		matched_scenes = all_scenes

	return matched_scenes


def extract_scenes(video_path: str, frame_timestamp: List[float], dialogues, cache_path, frames_dir, threshold: float = 30.0):
	cache_dir = f"{cache_path}/extract_scenes.json"

	if os.path.exists(cache_dir) and not frame_timestamp:
		logger_config.info(f"Using cached scene detection: {video_path}")
		with open(cache_dir, "r") as f:
			data = json.load(f)
		return data

	print("[INFO] Detecting scenes...")
	scenes = run_transnetv2(video_path, frame_timestamp)
	if len(scenes) == 0:
		raise ValueError("scenes is empty")

	print("[INFO] Mapping dialogues to scenes and extracting frames...")
	scene_dialogue_map = map_dialogues_to_scenes(scenes, dialogues, video_path, frames_dir)

	if len(dialogues) > 0:
		print("[INFO] Combining consecutive scenes with identical dialogues...")
		# scene_dialogue_map = combine_consecutive_same_dialogues(scene_dialogue_map)
		scene_dialogue_map = combine_dialogues(scene_dialogue_map)

	with open(cache_dir, "w", encoding="utf-8") as f:
		json.dump(scene_dialogue_map, f, indent=2)
	print(f"\n[INFO] Mapping saved to {cache_dir}")

	return scene_dialogue_map


if __name__ == "__main__":
	video_path = "reuse/019_-_Defying_Gravity/019_-_Defying_Gravity_compressed.mp4"
	extract_scenes(video_path)
