from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import uvicorn
import shutil
import random
import string
from detect import Detect
from preprocessing import makeDirectory, clearDirectory  # , checkFilesSize
from create import setup


random.seed(2023)
big_folder = ''.join(random.choice(string.ascii_letters) for i in range(10))

app = FastAPI()
# Tạo hàm model ngay trong đây, return model
# Sau đó gắn hàm detection.run với model
detection = Detect()


@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    nested_result = {}
    nested_result, error = detection.run(files, model, big_folder)
    clearDirectory(big_folder)
    if len(error) > 0:
        raise HTTPException(status_code=404, detail=error, headers={
                            "Message": "ERROR IS OCCURRED"})

    return nested_result


if __name__ == '__main__':
    makeDirectory(big_folder)
    model = setup()
    print("In processing...")
    uvicorn.run(app=app, host="0.0.0.0", port=8020)
    print("Shutting down...")
    shutil.rmtree("./" + big_folder)
