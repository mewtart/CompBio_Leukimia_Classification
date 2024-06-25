import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

img_size = (224, 224)
class_labels = ['NORMAL', 'HEMM']

VGG16_model = tf.keras.models.load_model('./VGG16/vgg16-vgg16-89.94.h5')
ResNet50_model = tf.keras.models.load_model('./ResNet/resnet50-resNet-95.63.h5')
DenseNet121_model = tf.keras.models.load_model('./DenseNet/densenet121-denseNet-91.62.h5')
AlexNet_model = tf.keras.models.load_model('./AlexNet/sequential-alexNet-68.19.h5')
EfficientNet_model = tf.keras.models.load_model('./EfficientNet/efficientnetb3-efficientNet.h5')

def load_and_preprocess_image(image_path, img_size):
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(model, image_path):
    preprocessed_image = load_and_preprocess_image(image_path, img_size)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

def main():
    st.title('Leukemia Classification')
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["VGG16 Model", "ResNet50 Model", "DenseNet121 Model", "AlexNet Model", "EfficientNet Model"])

    with tab1:
        input_image_vgg16 = st.file_uploader('Upload bmp file', type='bmp', key='vgg16')
        if input_image_vgg16 is not None:
            predicted_label_vgg16 = predict_image(VGG16_model, input_image_vgg16)
            st.write(f'The predicted label: {predicted_label_vgg16}')

    with tab2:
        input_image_resnet50 = st.file_uploader('Upload bmp file', type='bmp', key='resnet50')
        if input_image_resnet50 is not None:
            predicted_label_resnet50 = predict_image(ResNet50_model, input_image_resnet50)
            st.write(f'The predicted label: {predicted_label_resnet50}')
    
    with tab3:
        input_image_densenet121 = st.file_uploader('Upload bmp file', type='bmp', key='densenet121')
        if input_image_densenet121 is not None:
            predicted_label_densenet121 = predict_image(DenseNet121_model, input_image_densenet121)
            st.write(f'The predicted label: {predicted_label_densenet121}')
            
    with tab4:
        input_image_alexnet = st.file_uploader('Upload bmp file', type='bmp', key='alexnet')
        if input_image_alexnet is not None:
            predicted_label_alexnet = predict_image(AlexNet_model, input_image_alexnet)
            st.write(f'The predicted label: {predicted_label_alexnet}')
    
    with tab5:
        input_image_efficientnet = st.file_uploader('Upload bmp file', type='bmp', key='efficientnet')
        if input_image_efficientnet is not None:
            predicted_label_efficientnet = predict_image(EfficientNet_model, input_image_efficientnet)
            st.write(f'The predicted label: {predicted_label_efficientnet}')

if __name__ == '__main__':
    main()
