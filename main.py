from PIL import Image
import os
from find_recom import find_recommendations
from plot_recom import plot_recommendations

# Given image path and subCategory
subCATEGORY = 'Topwear'  # Specify the subCategory for the Recomendation you want

# Image directory
given_img_file = '1607.jpg'
img_dataset_path = './fashion-dataset/images/'
given_img_path = os.path.join(img_dataset_path, given_img_file)
INPUG_IMG = Image.open(given_img_path)

#--------------------------------Recomendation call-------------------------------------
# Call the recommendation function
recommended_images = find_recommendations(INPUG_IMG, subCATEGORY)
print("Recommended Images:", recommended_images)

plot_recommendations(INPUG_IMG, recommended_images)