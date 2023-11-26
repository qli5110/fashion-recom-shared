import requests
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Load the URLs and filenames from images.csv
url_table = pd.read_csv('./fashion-dataset/images.csv')

def plot_recommendations(input_image, recommended_image_filenames, url_table=url_table):
    """
    Plots the original image and its top recommended images using links from a CSV file.

    :param input_image: PIL Image object of the input image.
    :param recommended_image_filenames: List of file names for the recommended images.
    :param url_table: DataFrame containing image filenames and their corresponding URLs.
    """
    
    plt.figure(figsize=(15, 10))

    # Plot the original image
    plt.subplot(1, len(recommended_image_filenames) + 1, 1)
    plt.imshow(input_image)
    plt.title("Original Image")
    plt.axis('off')

    # Plot the recommended images
    for i, img_filename in enumerate(recommended_image_filenames, 2):
        img_url = url_table.loc[url_table['filename'] == img_filename, 'link'].iloc[0]
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))

        plt.subplot(1, len(recommended_image_filenames) + 1, i)
        plt.imshow(img)
        plt.title(f"Recommendation {i-1}")
        plt.axis('off')

    plt.show()

