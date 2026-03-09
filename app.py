
import cv2
import numpy as np
from tensorflow.keras.models import load_model


class Application:
    def __init__(self, model_path: str, video_device: int = 0):
        self.model_path = model_path
        self.model = None

        self.video_device = video_device

    def load_model(self):
        self.model = load_model(self.model_path, compile=False)

    def _predict_bgr(self, frame):
        if not self.model:
            raise Exception("Model is not loaded")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # конвертируем из BGR в RGB для подачи в модель
        image = cv2.resize(frame_rgb, (224, 224))
        image = np.expand_dims(image, axis=0)
        pred = self.model.predict(image)
        return pred

    def _draw_mask(self, frame, pred, grid):
        h, w = frame.shape[:2]

        rows = 7
        cols = 7

        cell_w = w // cols
        cell_h = h // rows

        # вертикальные линии
        for i in range(1, cols):
            x = i * cell_w
            cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 1)

        # горизонтальные линии
        for i in range(1, rows):
            y = i * cell_h
            cv2.line(frame, (0, y), (w, y), (255, 255, 255), 1)

        for y in range(7):
            for x in range(7):
                obj = grid[y, x, 6]

                if obj > 0.3:
                    x1 = x * cell_w
                    y1 = y * cell_h
                    x2 = (x + 1) * cell_w
                    y2 = (y + 1) * cell_h

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - inter + 1e-6
        return inter / union

    def _nms(self, boxes, scores, iou_thr=0.5):
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        scores = np.array(scores)

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            rest = order[1:]
            new_rest = []

            for j in rest:
                if self._iou(boxes[i], boxes[j]) < iou_thr:
                    new_rest.append(j)

            order = np.array(new_rest)

        return boxes[keep]

    def _draw_bboxes(self, frame, grid):
        h_img, w_img = frame.shape[:2]

        boxes = []
        scores = []

        for y in range(7):
            for x in range(7):
                dx, dy, dw, dh, bg, hand, obj = grid[y, x]

                if obj < 0.3:
                    continue

                x_center = (x + 0.5 + dx) / 7
                y_center = (y + 0.5 + dy) / 7

                bw = np.exp(dw) / 7
                bh = np.exp(dh) / 7

                xmin = (x_center - bw / 2) * w_img
                xmax = (x_center + bw / 2) * w_img
                ymin = (y_center - bh / 2) * h_img
                ymax = (y_center + bh / 2) * h_img

                boxes.append([xmin, ymin, xmax, ymax])
                scores.append(obj)

        boxes = self._nms(boxes, scores)
        for xmin, ymin, xmax, ymax in boxes:
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

    def run(self):
        cap = cv2.VideoCapture(self.video_device)

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break

            pred = self._predict_bgr(frame)
            grid = pred.reshape(7, 7, 7)

            self._draw_mask(frame, pred, grid)
            self._draw_bboxes(frame, grid)

            cv2.imshow("Hand Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

