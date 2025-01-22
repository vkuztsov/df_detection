import argparse
import torch
from torch.utils.data import DataLoader
from utils.image_prep import ImagePrep
from local_dataset import LocalDataset
import timm

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None
    return torch.stack([item[0] for item in batch]), torch.tensor([item[1] for item in batch])

def test_model(args):
    image_processor = ImagePrep(patch_size=args.patch_size)

    test_dataset = LocalDataset(args.test_dir, image_processor, only_faces=args.only_faces)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    model = timm.create_model(args.model_name, pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(args.model_path))
    print(f"Loaded model from {args.model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for patches, labels in test_loader:
            if patches is None:
                continue

            patches = patches.to(device)
            labels = labels.to(device)

            outputs = model(patches)
            _, predicted = torch.max(outputs, 1)

            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    accuracy = (correct_preds / total_preds) * 100 if total_preds > 0 else 0
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Total images tested: {total_preds}")
    print(f"Total correct preds: {correct_preds}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True, help="Path to the test dataset directory (should contain 'real' and 'fake' subfolders)")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for testing")
    parser.add_argument('--patch_size', type=int, default=224, help="Patch size for preprocessing")
    parser.add_argument('--model_name', type=str, default="vit_base_patch16_224", help="Model name (e.g., 'vit_base_patch16_224', 'resnet50')")
    parser.add_argument('--only_faces', default=False, help="Use only face patches", action='store_true')

    args = parser.parse_args()
    test_model(args)
