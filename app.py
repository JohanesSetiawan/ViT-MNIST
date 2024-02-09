import torch
import torchvision.transforms.functional as TF
import gradio as gr

from PIL import Image
from utils.params import Parameters
from utils.model import ViT

params = Parameters()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
loaded_model = ViT(num_patches=params.NUM_PATCHES, img_size=params.IMG_SIZE, num_classes=params.NUM_CLASSES, patch_size=params.PATCH_SIZE,
                   embed_dim=params.EMBED_DIM, num_encoders=params.NUM_ENCODERS, num_heads=params.NUM_HEADS, hidden_dim=params.HIDDEN_DIM,
                   dropout=params.DROPOUT, activation=params.ACTIVATION, in_channels=params.IN_CHANNELS).to(device)
checkpoint = torch.load(params.PATH_MODELS)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.eval()

# Get all parameters of the model
model_parameters = loaded_model.state_dict()


def predict_image(image):
    # Convert numpy array to PIL Image
    image = Image.fromarray(image)

    # Convert image to grayscale, resize, and transform to tensor
    image = image.convert("L").resize((params.IMG_SIZE, params.IMG_SIZE))
    image_tensor = TF.to_tensor(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = loaded_model(image_tensor)

    # Get predicted class
    predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class


# Gradio Interface
input_image = gr.Image(sources='upload', label="Upload Image")
output_text = gr.Textbox(label="Prediction Class")

interface = gr.Interface(predict_image, inputs=input_image,
                         outputs=output_text, title="ViT Image Classifier")
interface.launch()
