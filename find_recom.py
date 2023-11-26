import torch
import torchvision.transforms as transforms
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights


# Load the necessary data
styles_df = pd.read_csv('./fashion-dataset/styles.csv', on_bad_lines='skip') 
styles_df['filename'] = styles_df['id'].astype(str) + '.jpg'

# Load the saved embeddings
embeddings_df = pd.read_csv('./fashion-dataset/embeddings.csv')
filenames = embeddings_df['filename']
embeddings = embeddings_df.drop('filename', axis=1).values

#----------------------------------------Image processing preparation------------------------
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the pre-trained ResNet50 model
resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))  # Remove the last classification layer
resnet50 = resnet50.to(device)
resnet50.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#-----------------------------Recomendation functions-------------------------------------
# Function to preprocess and compute embedding for a single image
def preprocess_and_embed_single(input_image, model, transform):
    input_image = transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(input_image).squeeze().cpu().numpy()

    return embedding

# Ouput recomendations consider the category
def find_recommendations(input_image, subCategory, top_n=5, model=resnet50, embeddings_df=embeddings_df, styles_df=styles_df, transform=transform):
    """
    Finds top N recommended images based on the given subCategory.

    :param input_image: PIL Image object of the input image.
    :param model: Pre-trained PyTorch model for feature extraction.
    :param transform: Transformations applied to the input image.
    :param embeddings_df: DataFrame containing embeddings and filenames.
    :param styles_df: DataFrame containing style information including subCategory.
    :param subCategory: The specific category to base recommendations on.
    :param top_n: Number of top recommendations to return.
    :return: Filenames of the top N recommended images.
    """
    # Compute embedding for the given image
    given_image_embedding = preprocess_and_embed_single(input_image, model, transform)

    # Filter both embeddings and styles for the specified subCategory
    filtered_styles = styles_df[styles_df['subCategory'] == subCategory]
    filtered_embeddings_df = embeddings_df[embeddings_df['filename'].isin(filtered_styles['filename'])]

    filenames = filtered_embeddings_df['filename']
    embeddings = filtered_embeddings_df.drop('filename', axis=1).values

    # Compute similarities
    similarities = cosine_similarity([given_image_embedding], embeddings)[0]

    # Find top N similar images
    similar_indices = np.argsort(-similarities)[:top_n]
    recommended_images = filenames.iloc[similar_indices].values

    return recommended_images

# #--------------------------------Recomendation call-------------------------------------
# # Call the recommendation function
# recommended_images = find_recommendations(INPUG_IMG, subCATEGORY, resnet50, embeddings_df, styles_df)
# print("Recommended Images:", recommended_images)


# # Call the modified plot function
# plot_recommendations(INPUG_IMG, recommended_images)
