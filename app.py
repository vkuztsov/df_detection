import gradio as gr
from df_detection_model import DFDetectionModel
from utils.image_prep import ImagePrep

detection_model = DFDetectionModel(weights_path='weights/main_model.pth')
face_detection_model = DFDetectionModel(weights_path='weights/faces_model.pth')

def analyze_image(image):
    prediction = detection_model.predict(image)
    
    faces = ImagePrep.extract_faces(image)
    face_results = []
    face_images_with_results = []

    if faces:
        for x in faces:
            face_prediction = face_detection_model.predict(x)
            
            face_class = face_prediction['class']
            face_confidence = face_prediction['confidence']
            face_results.append({
                "class": face_class,
                "confidence": face_confidence
            })

            face_result = 'Fake' if face_class == 0 else 'Real'
            result_text = f"{face_result}, Confidence: {face_confidence:.2f}"
            face_images_with_results.append((x, result_text))  # Pair image with text

    predicted_class = prediction['class']
    confidence = prediction['confidence']

    confidence_percentage = confidence

    if predicted_class == 1:
        real_probability = confidence_percentage
        fake_probability = 1 - real_probability
    else:
        fake_probability = confidence_percentage
        real_probability = 1 - fake_probability

    results = {
        "Real": real_probability,
        "Fake": fake_probability
    }

    return results, face_images_with_results if faces else []

with gr.Blocks() as demo:
    gr.Markdown("# Synthetic image detection")
    gr.Markdown("Upload an image to analyze its authenticity.")

    with gr.Row():
        image_input = gr.Image(label="Upload Image", type="pil")
        with gr.Column():
            output = gr.Label(label="Analysis Results")
            faces_gallery = gr.Gallery(label="Detected Faces with Analysis", show_label=False)

    analyze_button = gr.Button("Analyze Image")
    analyze_button.click(analyze_image, inputs=image_input, outputs=[output, faces_gallery])

demo.launch()
