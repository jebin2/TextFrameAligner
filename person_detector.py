import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


class PersonDetectorYOLO:
    def __init__(
        self,
        model_path="yolov8x6_animeface.pt",
        repo_id="Fuyucchi/yolov8_animeface",
        filename="yolov8x6_animeface.pt",
    ):
        self.model_path = model_path
        self.model = None

        # Download model if it doesn't exist
        if not os.path.exists(self.model_path):
            print("⬇️ Model not found. Downloading from Hugging Face...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )
            os.rename(downloaded_path, self.model_path)
            print("✅ Model downloaded:", self.model_path)

    def has_person(self, frame):
        if self.model is None:
            self.model = YOLO(self.model_path)

        results = self.model(frame, verbose=False)

        for result in results:
            if result.boxes is None:
                continue

            # Check class IDs (0 = person in COCO)
            if any(cls == 0 for cls in result.boxes.cls.cpu().numpy()):
                return True

        return False

# Example usage
if __name__ == "__main__":
    import cv2
    cap = cv2.VideoCapture(0)  # webcam
    detector = PersonDetectorYOLO()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if detector.has_person(frame):
            print("Person detected!")
        else:
            print("No person detected.")

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
