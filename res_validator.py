from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ResponseValidator:
	def __init__(self):
		self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
		
		# Flagged response examples
		flagged_examples = [
			"Cannot fulfill the request. The image was flagged.",
			"I cannot process this image due to content policy violations.",
			"This image violates our safety guidelines.",
			"Unable to analyze inappropriate content.",
			"Image blocked by content filter."
		]
		
		self.flagged_embeddings = self.model.encode(flagged_examples)
	
	def is_valid_response(self, response, threshold=0.65):
		"""Returns True if response is valid, False if flagged/blocked"""
		if not response or len(response.split()) < 10:
			return False
		
		response_embedding = self.model.encode([response])
		similarities = cosine_similarity(response_embedding, self.flagged_embeddings)[0]
		
		return np.max(similarities) < threshold

# Usage example and main execution
if __name__ == "__main__":
	resp_val = ResponseValidator()
	print(resp_val.is_valid_response("Cannot fulfill the request. The image was flagged.\n"))