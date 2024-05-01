import streamlit as st
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your model and tokenizer
model_id = "qresearch/llama-3-vision-alpha-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16).to("cuda")

def preprocess(image):
    """Preprocess the image to be model-ready."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def predict(image, question):
    """Process image and question, and predict the answer."""
    image = preprocess(image)
    inputs = tokenizer.encode_plus(question, return_tensors="pt")
    inputs['pixel_values'] = image.unsqueeze(0).to("cuda")
    outputs = model.generate(**inputs, max_length=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Streamlit interface
st.title('AI Vision Query App')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    question = st.text_input("Ask a question about the image:")
    if st.button('Predict'):
        with st.spinner('Generating answer...'):
            answer = predict(image, question)
            st.success('Done!')
            st.write(answer)
