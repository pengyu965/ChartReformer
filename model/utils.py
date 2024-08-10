from PIL import Image

def resize_and_pad(image, target_size=(256, 256), background_color=(255, 255, 255)):
    """
    Resize an image to fit the target size while maintaining aspect ratio.
    Pads the resized image to match the target size.

    :param image: A PIL Image object.
    :param target_size: A tuple (width, height) for the target size.
    :param background_color: A tuple (R, G, B) for the background color.
    :return: A PIL Image object.
    """
    # Calculate the ratio of the target size and resize the image accordingly
    original_width, original_height = image.size
    target_width, target_height = target_size
    ratio = min(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * ratio), int(original_height * ratio))
    image = image.resize(new_size)

    # Create a new image with the specified background color and target size
    new_img = Image.new("RGB", target_size, background_color)
    # Calculate position to paste the resized image on the background
    x = (target_width - new_size[0]) // 2
    y = (target_height - new_size[1]) // 2
    new_img.paste(image, (x, y))

    return new_img