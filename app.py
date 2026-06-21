# Install dependencies
!pip -q install fastapi uvicorn python-multipart pillow

from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

# Create FastAPI app
app = FastAPI(
    title="Image Classification API"
)

# Sample classes
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

@app.get("/")
def home():
    return {
        "message": "API Running Successfully"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    width, height = image.size

    # Dummy prediction
    prediction = "cat"

    return {
        "filename": file.filename,
        "width": width,
        "height": height,
        "prediction": prediction
    }

print("FastAPI project created successfully.")
print("Ready for GitHub submission.")
