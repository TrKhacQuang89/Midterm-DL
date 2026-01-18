import streamlit as st
import numpy as np
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import time

# Import functions from mlp_network
try:
    from mlp_network import load_model, forward
except ImportError:
    st.error("Could not find 'mlp_network.py'. Please make sure it's in the same directory.")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Electronic Component Classifier",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton>button {
        width: 100%;
    }
    .prediction-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #3e4451;
        text-align: center;
        margin-bottom: 20px;
    }
    .class-name {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
    }
    .confidence {
        font-size: 18px;
        color: #8a8d97;
    }
    .sidebar .sidebar-content {
        background-color: #1e2130;
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- UTILS ---
@st.cache_resource
def get_model(path='best_model.npz'):
    if os.path.exists(path):
        return load_model(path)
    return None

def preprocess_image(image, size=64):
    """
    Preprocess PIL image for the model.
    """
    # Convert to grayscale
    img = image.convert('L')
    # Resize
    img = img.resize((size, size))
    # Normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    # Flatten
    flattened = img_array.flatten().reshape(1, -1)
    return flattened

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Model Settings")
    model_path = st.text_input("Model Path", "best_model.npz")
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    
    model = get_model(model_path)
    
    if model:
        st.success("Model loaded successfully!")
        config = model.get('config', {})
        st.json({
            "Input Size": f"{config.get('input_size')} (64x64)",
            "Layers": [
                config.get('hidden1'),
                config.get('hidden2'),
                config.get('hidden3'),
                config.get('hidden4'),
                config.get('hidden5')
            ],
            "Classes": config.get('num_classes')
        })
    else:
        st.warning("Model file not found. Please train the model first.")

    st.markdown("---")
    st.markdown("### 🛠️ Developer")
    st.info("Built with ❤️ using Streamlit & NumPy")

# --- MAIN CONTENT ---
st.markdown("<h1 style='text-align: center;'>🔌 Electronic Component Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image of an electronic component to identify it.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📸 Image Upload")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        predict_btn = st.button("RUN INFERENCE")
    else:
        st.info("Please upload an image to start classification.")

with col2:
    st.markdown("### 🎯 Classification Result")
    
    if uploaded_file is not None and model:
        if predict_btn:
            with st.spinner("Analyzing image..."):
                # Simulate a bit of processing time for "feel"
                time.sleep(0.5)
                
                # Preprocess
                processed_img = preprocess_image(image)
                
                # Predict
                output, _ = forward(processed_img, model)
                probabilities = output[0]
                prediction_idx = np.argmax(probabilities)
                confidence = probabilities[prediction_idx]
                
                # Class Names
                # Try to load from the dataset directory first
                dataset_path = r"e:\learn_midterm\final_classification_dataset\train"
                if os.path.exists(dataset_path):
                    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
                else:
                    # Fallback list
                    class_names = [
                        "Battery", "Capacitor", "Diode", "Display", 
                        "IC", "LED", "Potentiometer", "Resistor", 
                        "Switch", "Transistor"
                    ]
                
                if len(class_names) != len(probabilities):
                    st.warning(f"Warning: Model has {len(probabilities)} output units but {len(class_names)} classes were found.")
                    # Adjust class names if needed
                    if len(class_names) < len(probabilities):
                        class_names += [f"Class {i}" for i in range(len(class_names), len(probabilities))]
                    else:
                        class_names = class_names[:len(probabilities)]

                predicted_class = class_names[prediction_idx]
                
                # Display Prediction Card
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="class-name">{predicted_class}</div>
                    <div class="confidence">{confidence:.2%} Confidence</div>
                </div>
                """, unsafe_allow_html=True)
                
                # --- NEW: VISUAL PREPROCESSING SECTION ---
                st.markdown("---")
                st.markdown("### 🔍 Technical: Preprocessing & Vectorization")
                preprocess_col1, preprocess_col2 = st.columns(2)
                
                with preprocess_col1:
                    st.write("**1. Preprocessed Image (64x64 Grayscale)**")
                    # View what the model actually sees
                    preview_img = image.convert('L').resize((64, 64))
                    st.image(preview_img, width=150)
                
                with preprocess_col2:
                    st.write("**2. Numerical Vector (First 20 values)**")
                    # Show the raw numbers
                    st.code(str(processed_img[0][:20]) + "...", language="python")
                    st.caption("These 4,096 numbers (0 to 1) are what the AI 'reads'.")
                
                st.markdown("---")
                
                # Display Probability Chart
                st.markdown("#### Confidence Distribution")
                chart_data = pd.DataFrame({
                    'Class': class_names,
                    'Probability': probabilities
                })
                st.bar_chart(chart_data.set_index('Class'))
                
                # Top Predictions table
                top_3_indices = np.argsort(probabilities)[-3:][::-1]
                st.markdown("#### Top 3 Predictions")
                for i in top_3_indices:
                    st.write(f"**{class_names[i]}**: {probabilities[i]:.2%}")
                    st.progress(float(probabilities[i]))
        else:
            st.write("Click 'RUN INFERENCE' to see results.")
    elif uploaded_file is not None and not model:
        st.error("Model is not loaded. Cannot run inference.")

# --- FOOTER ---
st.markdown("---")
if st.checkbox("Show Loss History"):
    history_path = 'loss_history.csv'
    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['epoch'], df['train_loss'], label='Train Loss', color='#4CAF50')
        if 'val_loss' in df.columns:
            ax.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#FF9800')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        [t.set_color('white') for t in ax.xaxis.get_ticklabels()]
        [t.set_color('white') for t in ax.yaxis.get_ticklabels()]
        
        st.pyplot(fig)
    else:
        st.warning("Loss history file not found.")
