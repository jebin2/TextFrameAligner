from abc import ABC, abstractmethod
from PIL import Image
import torch

# 🧠 Base VisionModel
class VisionModel(ABC):
	def __init__(self, device=None):
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.model = None
		self.processor = None
		self.load_model()

	@abstractmethod
	def load_model(self):
		"""Load the model and processor/tokenizer"""
		pass

	@abstractmethod
	def generate(self, image: Image.Image, text: str = "") -> str:
		"""Generate response from image and optional text"""
		pass

	# Context manager support
	def __enter__(self):
		# Could also reload model if needed
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		try:
			# Free GPU memory if model is on GPU
			if self.model and hasattr(self.model, "to"):
				self.model.to("cpu")
			torch.cuda.empty_cache()
			del self.model
			del self.processor
		except: pass