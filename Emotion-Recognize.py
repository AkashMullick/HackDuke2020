from PIL import, Image, ImageDraw, Image, ImageFont


#Draw bounding box around face
def drawBBox(pImage, bounding, imageSize, caption='')

    width, height = imageSize
    draw = ImageDraw.Draw(pImage)

    draw.polygon([
        bounding.normalized_ver

    ])