import os
from PIL import Image
from tqdm import tqdm

img_name = "tgt_images" # "src_images"

src_images_path = "/home/snoamr/VideoEdit/CogVideo/image-editing/davidata/" + img_name
new_src_images_path = "/home/snoamr/VideoEdit/CogVideo/image-editing/davidata/" + img_name + "_resized"

# create new dir
if not os.path.exists(new_src_images_path):
    os.makedirs(new_src_images_path)

    
# loop over images in dir
for image_name in tqdm(os.listdir(src_images_path)):
    # open image
    image_path = os.path.join(src_images_path, image_name)
    image = Image.open(image_path)

    img_size = image.size

    super_h = 40
    # crop center  horizontally,  (640,480) --> (480,480)
    if img_size == (640, 480):
        image = image.crop((80 + super_h, 2 * super_h, 560 - super_h, 480))

    img_size = image.size

    # save image
    new_image_path = os.path.join(new_src_images_path, image_name)
    image.save(new_image_path)
    print(f"Saved image to {new_image_path}")
