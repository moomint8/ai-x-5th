import urllib.request

IMAGE_FILENAMES = ['img\burger.jpg', 'img\cat.jpg', 'img\salmon.jpg', 'img\pig_icecream.jpg', 'img\snake_01.jpg', 'img\dieRobot.jpeg' , 'img\RUInsaneHuman.jpg']

# for name in IMAGE_FILENAMES:
#   url = f'https://storage.googleapis.com/mediapipe-tasks/image_classifier/{name}'
#   urllib.request.urlretrieve(url, name)

# import cv2
# # from google.colab.patches import cv2_imshow
# import math

# DESIRED_HEIGHT = 480
# DESIRED_WIDTH = 480

# def resize_and_show(image):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
#   else:
#     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
# #   cv2_imshow(img)
#   cv2.imshow("test", img)
#   cv2.waitKey(0)


# # Preview the images.

# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)






# 라이브러리 가져오기
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# task 실행을 위한 추론기 생성
# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path='models\cls\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=1)
classifier = vision.ImageClassifier.create_from_options(options)

# 추론할 데이터 가져오기
# STEP 3: Load the input image.
image = mp.Image.create_from_file(IMAGE_FILENAMES[6])

# 추론 및 결과 호출, 항상 이 형태로 사용
# STEP 4: Classify the input image.
classification_result = classifier.classify(image)
print(classification_result)

# STEP 5: Process the classification result. In this case, visualize it.
# images.append(image)
top_category = classification_result.classifications[0].categories[0]
print(f"{top_category.category_name} ({top_category.score:.2f})")