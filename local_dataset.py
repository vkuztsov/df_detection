import os
from utils.image_prep import ImagePrep
from torch.utils.data import Dataset
from PIL import Image

class LocalDataset(Dataset):
    def __init__(self, root_dir, image_processor: ImagePrep, only_faces=False):
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.only_faces = only_faces
        self.samples = []

        for label, subfolder in enumerate(['fake', 'real']):
            folder_path = os.path.join(root_dir, subfolder)
            for fname in os.listdir(folder_path):
                file_path = os.path.join(folder_path, fname)
                if os.path.isfile(file_path):
                    self.samples.append((file_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path)

        processed_image = self.image_processor.preprocess(image, only_faces=self.only_faces)
        if processed_image is None:
            return None

        return processed_image, label