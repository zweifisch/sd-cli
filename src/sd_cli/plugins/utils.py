from PIL import Image

def resize_image(image, width, height):
    width0, height0 = image.size

    fit_width = width0 / height0 > width / height

    new_width = width if fit_width else int(width0 * height / height0)
    new_height = height if not fit_width else int(height0 * width / width0)

    resized = image.resize((new_width // 8 * 8, new_height // 8 * 8))
    return resized

    new_image = Image.new("RGB", (width, height), (0, 0, 0))
    new_image.paste(resized, ((width - new_width) // 2, (height - new_height) // 2))

    return new_image
