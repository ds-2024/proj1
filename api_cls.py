from fastapi import FastAPI, File, UploadFile

# STEP 1: Import the necessary modules. 패키지 가져오기.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object. 태스크 추론시키기 위해서 추론기 객체만들기.
base_options = python.BaseOptions(model_asset_path='models\cls\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=1)
classifier = vision.ImageClassifier.create_from_options(options)

app = FastAPI()

from PIL import Image
import numpy as np
import io
@app.post("/uploadfile/") #헤더만 보냄?
async def create_upload_file(file: UploadFile):

    content = await file.read()
    
    #content -> jpg 파일인데.. http 통신에서는 파일이 character type 왔다갔다함.
    # 1.text -> binary
    # 2.binary -> PIL Image 

    # STEP 3: Load the input image. 추론할 데이터 가져오기.
    binary = io.BytesIO(content) #io.BytesIO를 사용하여 바이너리 데이터를 메모리에 있는 바이너리 스트림으로 변환
    pil_img = Image.open(binary) #pil_img 변환이유) python에서 자료받는 형식
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Classify the input image. 추론하고 추론결과 가져오기. 건드릴필요가 없음.
    classification_result = classifier.classify(image) #classify(image) 변환이유) mediapipe에서 자료받는 형식
    

    # # STEP 5: Process the classification result. In this case, visualize it. 사용자에게 어떻게 보여줄 방법.
    top_category = classification_result.classifications[0].categories[0]
    result = f"{top_category.category_name} ({top_category.score:.2f})"
    
    return {"result": result}