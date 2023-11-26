Dataset used:
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/data


#------------------------------------------------------------------------------
find_recommendations(input_image, subCategory, top_n=5, model=resnet50)

# input_image -> image.                  Find recomenation based on this image
# subCategory -> str.                    What category of cloth you want to recommand
# top_n -> int.                          Number of recommendation to ouput from start from TOP
# model -> torchvision.models.           Model you want to use for process image
# find_recommendations() 
		->[img_id.jpg].          List of top n recomended image file names. 

Availabel subCategory:
['Topwear' 'Bottomwear' 'Watches' 'Socks' 'Shoes' 'Belts' 'Flip Flops'   
 'Bags' 'Innerwear' 'Sandal' 'Shoe Accessories' 'Fragrance' 'Jewellery'  
 'Lips' 'Saree' 'Eyewear' 'Nails' 'Scarves' 'Dress'
 'Loungewear and Nightwear' 'Wallets' 'Apparel Set' 'Headwear' 'Mufflers'
 'Skin Care' 'Makeup' 'Free Gifts' 'Ties' 'Accessories' 'Skin'
 'Beauty Accessories' 'Water Bottle' 'Eyes' 'Bath and Body' 'Gloves'     
 'Sports Accessories' 'Cufflinks' 'Sports Equipment' 'Stoles' 'Hair'     
 'Perfumes' 'Home Furnishing' 'Umbrellas' 'Wristbands' 'Vouchers'] 



#------------------------------------------------------------------------------
plot_recommendations(input_image, recommended_image_filenames, url_table=url_table)

# input_image -> image.                   Find recomenation based on this image
# subCategory -> str.                     What category of cloth you want to recommand
# recommended_image_filenames 
			->[img_id.jpg].   List of top n recomended image file names. 
# plot_recommendations() ->               Ploting the recomended images based on url.  
