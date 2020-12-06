from google.cloud import vision_v1


def batch_annotate_faces(input image):
    client = vision_v1.ImageAnnotationContext()

    source 