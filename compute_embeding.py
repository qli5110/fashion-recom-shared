import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights
import os

# Custom Dataset class to handle image loading
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, img_name

# Function to preprocess images and compute embeddings
def preprocess_and_embed(image_dir, model, transform, batch_size=32):
    dataset = ImageDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    filenames = []
    with torch.no_grad():
        for batch, names in tqdm(dataloader, desc="Computing embeddings"):
            batch = batch.to(device)
            features = model(batch)
            embeddings.append(features.squeeze().cpu().numpy())
            filenames.extend(names)

    embeddings = np.vstack(embeddings)
    return embeddings, filenames

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the pre-trained ResNet50 model with updated syntax
resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))  # Remove the last classification layer
resnet50 = resnet50.to(device)
resnet50.eval()  # Set the model to evaluation mode

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Example usage
# Assume a path where the images are stored
images_path = './fashion-dataset/images/'

# Compute embeddings for all images in the directory
embeddings, filenames = preprocess_and_embed(images_path, resnet50, transform)

# Save embeddings to a CSV file
embedding_df = pd.DataFrame(embeddings)
embedding_df['filename'] = filenames
embedding_df.to_csv('embeddings.csv', index=False)

print("Embeddings saved to 'embeddings.csv'")
