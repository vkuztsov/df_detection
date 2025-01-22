import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import timm
from PIL import Image
from utils.image_prep import ImagePrep
from local_dataset import LocalDataset

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None
    return torch.stack([item[0] for item in batch]), torch.tensor([item[1] for item in batch])

def train_model(args):
    image_processor = ImagePrep(patch_size=args.patch_size)

    train_dataset = LocalDataset(args.train_dir, image_processor, only_faces=args.only_faces)
    val_dataset = LocalDataset(args.val_dir, image_processor, only_faces=args.only_faces)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    model = timm.create_model(args.model_name, pretrained=True, num_classes=2)

    if args.pretrained_model_path:
        model.load_state_dict(torch.load(args.pretrained_model_path))
        print(f"Loaded pretrained model from {args.pretrained_model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for patches, labels in train_loader:
            if patches is None:
                continue

            patches = patches.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        model.eval()
        val_loss = 0.0
        val_correct_preds = 0
        val_total_preds = 0
        with torch.no_grad():
            for patches, labels in val_loader:
                if patches is None:
                    continue

                patches = patches.to(device)
                labels = labels.to(device)

                outputs = model(patches)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct_preds += (predicted == labels).sum().item()
                val_total_preds += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct_preds / val_total_preds
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

    torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True, help="Path to the training dataset directory (should contain 'real' and 'fake' subfolders)")
    parser.add_argument('--val_dir', type=str, required=True, help="Path to the validation dataset directory (should contain 'real' and 'fake' subfolders)")
    parser.add_argument('--save_path', type=str, default="model.pth", help="Path to save the trained model")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--patch_size', type=int, default=224, help="Patch size for preprocessing")
    parser.add_argument('--model_name', type=str, default="vit_base_patch16_224", help="Model name (e.g., 'vit_base_patch16_224', 'resnet50')")
    parser.add_argument('--only_faces', default=False, help="Use only face patches", action='store_true')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help="Path to the pretrained model for fine-tuning")

    args = parser.parse_args()
    train_model(args)
