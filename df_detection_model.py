import torch
import timm
from PIL import Image
from utils.image_prep import ImagePrep

class DFDetectionModel:
    def __init__(self, weights_path, model_name='vit_base_patch16_224', patch_size=224, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = timm.create_model(model_name, pretrained=False, num_classes=2)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.image_processor = ImagePrep(patch_size=patch_size)

    def predict(self, image: Image):
        try:
            image_tensor = self.image_processor.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()

            return {"class": predicted_class, "confidence": confidence}
        except Exception as e:
            return {"error": str(e)}