import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import io

# --- IMPORT YOUR MODEL ---
# This assumes your P2PNet class is in testapp/p2pnet/model.py
try:
    from p2pnet.model import P2PNet
except ImportError:
    st.error("Fatal Error: Could not find 'p2pnet.model.P2PNet'. "
             "Make sure your model.py file is in the 'p2pnet' folder.")
    st.stop()

# --- CONSTANTS (from your infer.py args) ---
WEIGHTS_PATH = "checkpoints/best_model.pth"
MODEL_BASE_CH = 32
MODEL_IMG_SIZE = 256
# Force 'cpu' for compatibility with GCP Cloud Run's default instances
DEVICE = torch.device('cpu') 

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Underwater Image Enhancement",
    page_icon="🌊",
    layout="centered",  # <-- CHANGED: "wide" to "centered" for medium-sized cards
)

# --- HELPER FUNCTIONS (from your infer.py) ---

@st.cache_resource  # Caches the model so it only loads once
def load_model(weights_path):
    """
    Loads and returns the P2PNet model in evaluation mode.
    """
    try:
        model = P2PNet(base_ch=MODEL_BASE_CH).to(DEVICE)
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        
        # Handle checkpoints saved with 'model_state_dict' or as raw state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Error: Weights file not found at {weights_path}.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image_pil):
    """
    Transforms a PIL Image into the format expected by the model.
    """
    transform = transforms.Compose([
        transforms.Resize((MODEL_IMG_SIZE, MODEL_IMG_SIZE)),
        transforms.ToTensor()
    ])
    # Add batch dimension and send to device
    return transform(image_pil).unsqueeze(0).to(DEVICE)

def postprocess_output(output_tensor):
    """
    Converts the model's output tensor back into a PIL Image.
    """
    output_img_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Clip values to be in valid [0, 255] range after scaling
    output_img_np = np.clip(output_img_np * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(output_img_np)

# --- MAIN APP ---

# Load the model
model = load_model(WEIGHTS_PATH)

# Sidebar for introduction and file upload
with st.sidebar:
    st.title("🌊 Underwater Image Enhancement") # <-- I made this title a bit shorter
    st.markdown("""
        **Welcome!**
        
        This app enhances underwater images.
        Upload an image to see the correction.
        
        ---
        **My Introduction:**
        * **Name:** Abhi Tundiya
        * **Project:** Underwater Image Enhancement
    """)
    
    st.header("Upload Your Image")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, PNG)...", 
        type=["jpg", "jpeg", "png"]
    )

# Main page for displaying results
st.title("Underwater Image Enhancement")

if uploaded_file is None:
    st.info("Please upload an image using the sidebar to begin.")

if uploaded_file is not None and model is not None:
    # Read the uploaded image
    image_full = Image.open(uploaded_file).convert("RGB")
    
    # --- CHANGED: Create a 256x256 version for a fair comparison ---
    image_resized = image_full.resize((MODEL_IMG_SIZE, MODEL_IMG_SIZE))
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original (256x256)")
        # Display the *resized* original image
        st.image(image_resized, caption="Uploaded Image (Resized)", use_container_width=True)

    # Process the image and display the result
    with col2:
        st.header("Enhanced (256x256)")
        # Add a spinner while processing
        with st.spinner('Enhancing image... Please wait.'):
            try:
                # 1. Preprocess
                # We can pass the full image here; the function will resize it
                input_tensor = preprocess_image(image_full) 
                
                # 2. Run Model Inference
                with torch.no_grad():
                    output_tensor = model(input_tensor)
                    
                # 3. Postprocess
                output_image = postprocess_output(output_tensor)
                
                # Display the enhanced image
                st.image(output_image, caption="Model Output", use_container_width=True)

                # Add a download button
                buf = io.BytesIO()
                output_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Enhanced Image",
                    data=byte_im,
                    file_name=f"enhanced_{uploaded_file.name}.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"An error occurred during inference: {e}")

elif uploaded_file is not None and model is None:
    st.error("Model could not be loaded. Please check the logs.")