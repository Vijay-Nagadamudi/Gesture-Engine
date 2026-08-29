import cv2


class HandRenderer:

    def draw_landmarks(self, frame, landmarks):
        height, width, _ = frame.shape

        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                5,
                (255, 0, 0),
                -1
            )

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

        for start_index, end_index in connections:
            start = landmarks[start_index]
            end = landmarks[end_index]

            start_point = (
                int(start.x * width),
                int(start.y * height)
            )

            end_point = (
                int(end.x * width),
                int(end.y * height)
            )

            cv2.line(
                frame,
                start_point,
                end_point,
                (255, 0, 0),
                2
            )

        return frame