from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_and_pad_image(image_path: str, target_height: int = 480, target_width: int = 720) -> Image.Image:
    """
    Resize image (up or down) such that its larger dimension fits the required size, then pad the other dimension.

    Args:
        image_path: Path to the input image
        target_height: Desired height in pixels
        target_width: Desired width in pixels

    Returns:
        Processed PIL Image
    """
    img = Image.open(image_path)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    orig_width, orig_height = img.size

    width_ratio = target_width / orig_width
    height_ratio = target_height / orig_height

    if orig_width <= target_width and orig_height <= target_height:
        scale_ratio = max(width_ratio, height_ratio)
    else:
        scale_ratio = min(width_ratio, height_ratio)

    new_width = int(orig_width * scale_ratio)
    new_height = int(orig_height * scale_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    final_img = Image.new('RGB', (target_width, target_height), (0, 0, 0))

    left_padding = (target_width - new_width) // 2
    top_padding = (target_height - new_height) // 2

    final_img.paste(img, (left_padding, top_padding))

    return final_img


img = resize_and_pad_image(image_path="./121000000002.jpg")

img.show()

if img.mode != 'RGB':
    img = img.convert('RGB')