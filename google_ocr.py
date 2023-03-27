import cv2
from google.cloud import storage
from google.cloud import vision
import os


def main():
    # Initializing Google Cloud Storage bucket
    bucket_name = 'atu-ocr'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    if not bucket.exists():
        bucket = storage_client.create_bucket(bucket_name)
    else:
        bucket = storage_client.get_bucket(bucket.name)

    # Iterating over files in "data" folder
    for file_name in os.listdir('input'):
        try:
            file_name = os.path.join('input', file_name)
            print("Processing: " + file_name)
            img = cv2.imread(file_name)
            # Cropping out right label area and left label area as separate pictures.
            left_label_img, right_label_img = get_labels(img)
            cv2.imwrite('left_label.jpg', left_label_img)
            cv2.imwrite('right_label.jpg', right_label_img)

            # Uploading temporary cropped out pictures to Google Cloud Storage for further processing.
            blob = bucket.blob("left_label.jpg")
            blob.upload_from_filename("left_label.jpg")
            blob = bucket.blob("right_label.jpg")
            blob.upload_from_filename("right_label.jpg")
            left_label_gs_uri, right_label_gs_uri = 'gs://' + bucket_name + '/left_label.jpg', 'gs://' + bucket_name + '/right_label.jpg'

            # Running Google Cloud Vision OCR API to extract text from two labels.
            left_label_text, right_label_text = detect_text_uri(left_label_gs_uri), detect_text_uri(right_label_gs_uri)
            right_label_text = validate_right_label(right_label_text)
            print("Left label: " + left_label_text)
            print("Right label: " + right_label_text + '\n')

            # Renaming original files with extracted labels.
            file_name = left_label_text + " && " + right_label_text + "." + file_name.split('.')[1]
            # Saving file to output folder
            cv2.imwrite(os.path.join('output', file_name), img)
        except Exception as e:
            print("Error processing: " + file_name)
            print(e)
            continue


def get_labels(img):
    # Resolutions may vary. Need to adjust according to the given dataset.
    left_label = img[0:300, 0:1000]
    right_label = img[0:300, 4000:img.shape[1]]
    return left_label, right_label


def detect_text_uri(uri):
    # Taken from Google Cloud Vision API documentation
    """Detects text in the file located in Google Cloud Storage or on the Web.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return texts[0].description


def validate_right_label(right_label_text):
    # Right label text should match TTT-NT-NN format
    # Where T is a letter and N is a number.
    # If string is not resembling format, we need to further process it and correct OCR errors
    resembling_characters = {
        'G': '0',
        'I': '1',
    }
    # Needs substantial improvement
    right_label_numbers = list(right_label_text[4:5] + right_label_text[-2:])
    right_label_letters = list(right_label_text[0:3] + right_label_text[5:6])
    # Replacing resembling characters, searching using keys when replacing letters

    for i in range(len(right_label_numbers)):
        if right_label_numbers[i] in resembling_characters.keys():
            right_label_numbers[i] = resembling_characters[right_label_numbers[i]]
    for i in range(len(right_label_letters)):
        if right_label_letters[i] in resembling_characters.values():
            right_label_letters[i] = list(resembling_characters.keys())[
                list(resembling_characters.values()).index(right_label_letters[i])]
    right_label_numbers = ''.join(right_label_numbers)
    right_label_letters = ''.join(right_label_letters)

    # Reassembling correct format
    right_label_text = right_label_letters[0:3] + '-' + right_label_numbers[0] + right_label_letters[
        3] + '-' + right_label_numbers[1:3]
    return right_label_text


if __name__ == "__main__":
    main()
