import numpy as np
class LandMarkFeatureExtractor:
    def __init__(self):
        self.reference_landmark = 0
    
    def normalize_landmarks(self, landmarks):
        points = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in landmarks],
            dtype = np.float32
        )
        wrist = points[self.reference_landmark]
        points = points - wrist
        
        distances = np.linalg.norm(points,axis = 1)
        scale = np.max(distances)
        
        if scale == 0:
            return points
        
        points = points / scale
        
        return points
    
    
    def extract(self,landmarks):
        normalized_landmarks = self.normalize_landmarks(landmarks)
        
        features = normalized_landmarks.flatten()
        return features