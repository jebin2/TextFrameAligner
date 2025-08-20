from ultralytics import YOLO

class PersonDetectorYOLO:
    def __init__(self, model_path="yolov8x6_animeface.pt"):
        self.model = YOLO(model_path)

    def has_person(self, frame):
        results = self.model(frame, verbose=False)  # Run inference
        for result in results:
            # Check detected class IDs for person (class 0 in COCO)
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
