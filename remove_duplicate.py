import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
import common

class FaceDINO:
	def __init__(self, model_name="facebook/dinov3-vitl16-pretrain-lvd1689m", device="cuda", threshold=0.9):
		self.device = device or ("cuda" if common.is_gpu_available() else "cpu")
		self.processor = AutoImageProcessor.from_pretrained(model_name)
		self.model = AutoModel.from_pretrained(model_name).to(self.device)
		self.model.eval()

		self.embeddings = []   # cached embeddings
		self.paths = []		# optional: store paths or ids
		self.threshold = threshold

	def get_embedding(self, image):
		"""Accepts either a cv2 frame (numpy) or an image path"""
		if isinstance(image, str) or isinstance(image, Path):  # path
			img = Image.open(image).convert("RGB")
		else:  # assume numpy frame from cv2
			img = Image.fromarray(image[..., ::-1])  # BGR → RGB

		inputs = self.processor(images=img, return_tensors="pt").to(self.device)
		with torch.inference_mode():
			outputs = self.model(**inputs)
		if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
			emb = outputs.pooler_output.squeeze()
		else:
			emb = outputs.last_hidden_state[:, 0, :]
		return torch.nn.functional.normalize(emb, p=2, dim=-1).cpu()

	def is_duplicate(self, image, save_duplicate=False):
		"""
		Check if image is duplicate compared to cache.
		Returns: (is_dup: bool, similarity: float)
		"""
		emb = self.get_embedding(image)

		if len(self.embeddings) == 0:
			# first image always added
			self.embeddings.append(emb)
			if isinstance(image, (str, Path)):
				self.paths.append(str(image))
			return False, 0.0

		# Compare with cached embeddings
		all_embs = torch.stack(self.embeddings)  # (N, D)
		sims = torch.mm(emb.unsqueeze(0), all_embs.T).squeeze()  # (N,)

		max_sim, idx = torch.max(sims, dim=0)
		is_dup = max_sim >= self.threshold

		if is_dup:
			if save_duplicate:
				# still add it to cache
				self.embeddings.append(emb)
				if isinstance(image, (str, Path)):
					self.paths.append(str(image))
			return True, float(max_sim)

		# Not duplicate → always add
		self.embeddings.append(emb)
		if isinstance(image, (str, Path)):
			self.paths.append(str(image))

		return False, float(max_sim)

	def unload(self):
		"""Free model, embeddings, and clear GPU memory."""
		try:
			import gc
			# Clear embeddings and paths
			self.embeddings.clear()
			self.paths.clear()

			# Delete model and processor
			if hasattr(self, "model"):
				del self.model
			if hasattr(self, "processor"):
				del self.processor

			# Run garbage collection
			gc.collect()

			# Clear CUDA memory if available
			if common.is_gpu_available():
				torch.cuda.empty_cache()
				torch.cuda.ipc_collect()

			print("[FaceDINO] Resources unloaded successfully.")
		except Exception as e:
			print(f"[FaceDINO] Error during unload: {e}")

	def __del__(self):
		"""Auto cleanup when object is destroyed."""
		self.unload()


if __name__ == "__main__":
	import cv2

	dino = FaceDINO(threshold=0.85)

	path = f'{os.getenv("PARENT_BASE_PATH")}/CaptionCreator/reuse/comic_review_Clementine Chapter 02/split_0007'
	frame_id = 0
	for file in sorted([file for file in os.listdir(path) if "(" in file]):
		dup, sim = dino.is_duplicate(f"{path}/{file}")
		if dup:
			frame_id += 1
			print(f"Frame {frame_id}: duplicate={dup}, sim={sim:.3f} {file}")
		# 	cv2.imwrite(f"temp2/frame_{frame_id}.jpg", frame)
			# os.remove(f"temp2/{file}")
	# cap.release()
	del dino
