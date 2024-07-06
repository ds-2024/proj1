from fastapi import FastAPI, File, UploadFile

# STEP 1: Import the necessary modules. 
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='models\det\efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5) #객체 감지기가 인식한 객체의 점수(score)가 0.5 이상일 때만 해당 객체를 유효하게 인식된 것으로 간주
detector = vision.ObjectDetector.create_from_options(options)

app = FastAPI()

from PIL import Image
import numpy as np
import io

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    content = await file.read()
    # STEP 3: Load the input image. 추론할 데이터 가져오기.
    binary = io.BytesIO(content) #io.BytesIO를 사용하여 바이너리 데이터를 메모리에 있는 바이너리 스트림으로 변환
    pil_img = Image.open(binary) #pil_img 변환이유) python에서 자료받는 형식
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Detect objects in the input image.
    detection_result = detector.detect(image)
    # DetectionResult(detections=[
    # Detection(
    #     bounding_box=BoundingBox(
    #         origin_x=2, 
    #         origin_y=41, 
    #         width=336, 
    #         height=417
    #     ), 
    #     categories=[
    #         Category(
    #             index=None, 
    #             score=0.9556650519371033, 
    #             display_name=None, 
    #             category_name='person'
    #         )
    #     ], 
    #     keypoints=[]
    # )
    # ])


    counts = len(detection_result.detections)
    object_list = []
    for detection in detection_result.detections:
        object_category = detection.categories[0].category_name
        object_list.append(object_category)

    print(detection_result)
    print(object_list)

    # # STEP 5: Process the detection result. In this case, visualize it.
    # image_copy = np.copy(image.numpy_view())
    # annotated_image = visualize(image_copy, detection_result)
    # rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return {"counts": counts,
            "object_list": object_list}