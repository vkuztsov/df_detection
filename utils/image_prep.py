import dlib
import torch
import numpy as np
from PIL import Image
import cv2
from scipy.signal import convolve2d

class ImagePrep:
    SRM_FILTERS = [
        np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]]),
        np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]),
    ]

    def __init__(self, patch_size=224):
        self.patch_size = patch_size

    @staticmethod
    def apply_srm(image):
        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))
        filtered_channels = [convolve2d(image, f, mode='same', boundary='symm') for f in ImagePrep.SRM_FILTERS]
        return np.stack(filtered_channels, axis=-1)

    @staticmethod
    def is_valid_image(image: Image):
        return image.mode in ('RGB', 'RGBA')

    @staticmethod
    def extract_faces(image: Image, patch_size=224):
        cv_image = np.array(image)
        detector = dlib.get_frontal_face_detector()
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            return None

        face_patches = []

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_image = cv_image[y:y+h, x:x+w]

            if face_image.size == 0:
                continue

            face_resized = cv2.resize(face_image, (patch_size, patch_size))
            
            face_pil = Image.fromarray(face_resized)
            face_patches.append(face_pil)

        return face_patches

    
    def preprocess(self, image: Image, only_faces=False):
        if not self.is_valid_image(image):
            return None

        image = image.convert('RGB')
        width, height = image.size

        if width < self.patch_size or height < self.patch_size:
            return None
        
        patch = image.crop((0, 0, self.patch_size, self.patch_size))

        if only_faces:
            faces = self.extract_faces(image, patch_size=self.patch_size)
            if faces != None:
                patch = faces[0]
            else:
                return None

        gray_image = np.array(patch.convert('L'))
        srm_features = self.apply_srm(gray_image)
        srm_features = (srm_features - srm_features.mean()) / (srm_features.std() + 1e-5)
        srm_tensor = torch.tensor(srm_features, dtype=torch.float32).permute(2, 0, 1)

        return srm_tensor