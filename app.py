import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# Constants
IMG_SIZE = (224, 224)
CLASS_LABELS = ['NORMAL', 'HEMM']

# Load models
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

VGG16_model = load_model('./VGG16/vgg16-vgg16-89.94.h5')
ResNet50_model = load_model('./ResNet/resnet50-resNet-95.63.h5')
DenseNet121_model = load_model('./DenseNet/densenet121-denseNet-91.62.h5')
AlexNet_model = load_model('./AlexNet/sequential-alexNet-68.19.h5')
EfficientNet_model = load_model('./EfficientNet/efficientnetb3-efficientNet.h5')

def load_and_preprocess_image(uploaded_file, img_size):
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(model, uploaded_file):
    preprocessed_image = load_and_preprocess_image(uploaded_file, IMG_SIZE)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = CLASS_LABELS[predicted_class_index]
    return predicted_class_label

def main():
    st.title('Leukemia Classification')
    
    tabs = st.tabs(["VGG16 Model", "ResNet50 Model", "DenseNet121 Model", "AlexNet Model", "EfficientNet Model"])

    models = {
        "VGG16 Model": VGG16_model,
        "ResNet50 Model": ResNet50_model,
        "DenseNet121 Model": DenseNet121_model,
        "AlexNet Model": AlexNet_model,
        "EfficientNet Model": EfficientNet_model,
    }

    for tab, (model_name, model) in zip(tabs, models.items()):
        with tab:
            input_image = st.file_uploader(f'Upload bmp file for {model_name}', type='bmp', key=model_name)
            if input_image is not None:
                try:
                    predicted_label = predict_image(model, input_image)
                    st.write(f'The predicted label: {predicted_label}')
                except Exception as e:
                    st.error(f"Error processing the image: {e}")

if __name__ == '__main__':
    main()
