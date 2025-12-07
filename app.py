import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time

st.set_page_config(page_title="Object Detection Demo", page_icon="🤖")

st.title("🤖 AI Object Detection System")
st.markdown("""
**Topic:** Machine Learning for Image Recognition  
**System:** YOLOv8 (You Only Look Once) Architecture  
**Status:** Ready for Inference
""")
st.markdown("---")


@st.cache_resource
def load_model():
    model = YOLO('yolov8n.pt')
    return model

with st.spinner('Loading neural network...'):
    model = load_model()

st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.4, 0.05)

uploaded_file = st.file_uploader("📂 Upload Image (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    if st.sidebar.button('🚀 Start Analysis', type="primary"):
        start_time = time.time()
        
        results = model(image, conf=confidence_threshold)
        results = model(image, conf=confidence_threshold)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        res_plotted = results[0].plot()
        res_image = Image.fromarray(res_plotted[..., ::-1])
        
        with col2:
            st.success(f"Result (Time: {inference_time:.2f} sec.)")
            st.image(res_image, use_container_width=True)

        st.markdown("### 📊 Detected Objects Statistics")
        
        boxes = results[0].boxes
        if len(boxes) > 0:
            detected_classes = [model.names[int(cls)] for cls in boxes.cls]
            
            from collections import Counter
            counts = Counter(detected_classes)
            
            st.json(counts)
            
            with st.expander("View Tensor Coordinates (Raw Data)"):
                for box in boxes:
                    st.code(f"Class: {model.names[int(box.cls)]} | Conf: {float(box.conf):.2f} | Coords: {box.xyxy.tolist()}")
        else:
            st.warning("No objects detected at this confidence level.")
else:
    st.info("Please upload an image from the menu to get started.")