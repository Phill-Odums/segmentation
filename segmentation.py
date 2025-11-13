import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import gdown
import os

# Configure the page
st.set_page_config(
    page_title="SAM Image Segmenter",
    page_icon="🎯",
    layout="wide"
)

@st.cache_resource
def download_sam_model():
    """Download SAM model from Google Drive if not exists"""
    file_id = "188I-GSROCvkYDfCEHBJCWRHZSCbLbQzC"
    model_path = "sam_vit_h_4b8939.pth"
    
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading SAM model from Google Drive (this may take a few minutes)..."):
            try:
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, model_path, quiet=False)
                st.success("✅ Model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Download failed: {e}")
                return None
    return model_path

@st.cache_resource
def load_sam_model():
    """Load and cache the SAM model"""
    # First download the model
    model_path = download_sam_model()
    if model_path is None:
        return None, None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = "vit_h"
    
    try:
        with st.spinner("🔄 Loading SAM model into memory..."):
            sam = sam_model_registry[model_type](checkpoint=model_path)
            sam.to(device=device)
            predictor = SamPredictor(sam)
        return predictor, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

def main():
    st.title("🎯 Segment Anything Model (SAM) Demo")
    st.markdown("Upload an image and add points to segment objects")
    
    # Add file size warning
    st.info("💡 **Note:** The SAM model (~2.4GB) will be downloaded on first run. This may take a few minutes.")

    # Load model
    predictor, device = load_sam_model()

    if predictor is None:
        st.error("Failed to load SAM model. Please check the console for errors.")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )

    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)
        
        # Resize if image is too large for better performance
        max_size = 1024
        if max(image_array.shape) > max_size:
            scale = max_size / max(image_array.shape)
            new_height = int(image_array.shape[0] * scale)
            new_width = int(image_array.shape[1] * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            image_array = np.array(image)
            st.info(f"📐 Image resized to {new_width}x{new_height} for better performance")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            
            # Display image
            st.image(image_array, use_column_width=True, caption="Original Image")
            
            # Point management
            st.markdown("### Point Management")
            
            # Initialize session state
            if 'points' not in st.session_state:
                st.session_state.points = []
            if 'labels' not in st.session_state:
                st.session_state.labels = []
            
            # Manual point input
            st.markdown("**Add points manually:**")
            col_point = st.columns(2)
            with col_point[0]:
                x_coord = st.slider("X coordinate", 0, image_array.shape[1]-1, image_array.shape[1]//2)
            with col_point[1]:
                y_coord = st.slider("Y coordinate", 0, image_array.shape[0]-1, image_array.shape[0]//2)
            
            col_buttons = st.columns(2)
            with col_buttons[0]:
                if st.button("🎯 Add Positive Point", use_container_width=True):
                    st.session_state.points.append([x_coord, y_coord])
                    st.session_state.labels.append(1)
                    st.rerun()
            with col_buttons[1]:
                if st.button("❌ Add Negative Point", use_container_width=True):
                    st.session_state.points.append([x_coord, y_coord])
                    st.session_state.labels.append(0)
                    st.rerun()
            
            # Clear points button
            if st.button("🗑️ Clear All Points", use_container_width=True):
                st.session_state.points = []
                st.session_state.labels = []
                st.rerun()
            
            # Display current points
            if st.session_state.points:
                positive_count = sum(st.session_state.labels)
                negative_count = len(st.session_state.labels) - positive_count
                st.write(f"**Current Points:** {positive_count} positive, {negative_count} negative")
                
                # Show points list
                for i, (point, label) in enumerate(zip(st.session_state.points, st.session_state.labels)):
                    st.write(f"Point {i+1}: ({point[0]}, {point[1]}) - {'Positive' if label == 1 else 'Negative'}")
            
            # Segment button
            if st.session_state.points:
                if st.button("🎪 Segment Image", type="primary", use_container_width=True):
                    st.rerun()

        with col2:
            st.subheader("Segmentation Result")
            
            if st.session_state.points:
                try:
                    # Set image in predictor
                    predictor.set_image(image_array)
                    
                    # Convert points to numpy arrays
                    input_points = np.array(st.session_state.points)
                    input_labels = np.array(st.session_state.labels)
                    
                    # Perform prediction
                    with st.spinner("🔄 Generating segmentation..."):
                        masks, scores, logits = predictor.predict(
                            point_coords=input_points,
                            point_labels=input_labels,
                            multimask_output=True,
                        )
                    
                    # Select the best mask
                    best_mask_idx = np.argmax(scores)
                    mask = masks[best_mask_idx]
                    
                    # Create visualization
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Original image with points
                    ax1.imshow(image_array)
                    for point, label in zip(st.session_state.points, st.session_state.labels):
                        color = 'green' if label == 1 else 'red'
                        marker = 'o' if label == 1 else 'x'
                        ax1.scatter(point[0], point[1], c=color, marker=marker, s=100,
                                  edgecolors='white', linewidth=2)
                    ax1.set_title("Input Image with Points")
                    ax1.axis('off')
                    
                    # Mask only
                    ax2.imshow(mask, cmap='viridis')
                    ax2.set_title("Segmentation Mask")
                    ax2.axis('off')
                    
                    # Overlay
                    ax3.imshow(image_array)
                    ax3.imshow(mask, alpha=0.5, cmap='viridis')
                    ax3.set_title(f"Overlay (Score: {scores[best_mask_idx]:.3f})")
                    ax3.axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display scores for all masks
                    st.write("**Mask Confidence Scores:**")
                    for i, score in enumerate(scores):
                        st.write(f"Mask {i+1}: {score:.3f}")
                    
                    # Download options
                    col_download = st.columns(2)
                    with col_download[0]:
                        # Download mask
                        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                        st.download_button(
                            label="📥 Download Mask",
                            data=mask_image.tobytes(),
                            file_name="segmentation_mask.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    with col_download[1]:
                        # Download overlay
                        overlay_fig, overlay_ax = plt.subplots(figsize=(8, 8))
                        overlay_ax.imshow(image_array)
                        overlay_ax.imshow(mask, alpha=0.5, cmap='viridis')
                        overlay_ax.axis('off')
                        overlay_fig.savefig("overlay.png", bbox_inches='tight', pad_inches=0)
                        with open("overlay.png", "rb") as file:
                            st.download_button(
                                label="📥 Download Overlay",
                                data=file,
                                file_name="segmentation_overlay.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    
                except Exception as e:
                    st.error(f"Error during segmentation: {e}")
                    st.write("Please try adjusting your points and try again.")
            else:
                st.info("👆 Add points to the image and click 'Segment Image' to see results here")
                # Show placeholder
                placeholder_fig, placeholder_ax = plt.subplots(figsize=(10, 8))
                placeholder_ax.imshow(image_array)
                placeholder_ax.set_title("Segmentation results will appear here")
                placeholder_ax.axis('off')
                st.pyplot(placeholder_fig)

    else:
        # Show instructions when no image is uploaded
        st.info("👆 Please upload an image to get started")
        
        with st.expander("📖 How to use this app"):
            st.markdown("""
            ### Step-by-Step Guide:
            
            1. **Upload an image** using the file uploader above
            2. **Add positive points** (green) by:
               - Moving the X and Y sliders to select coordinates
               - Clicking "Add Positive Point"
            3. **Add negative points** (red) for areas that are NOT part of the object
            4. **Click 'Segment Image'** to generate the segmentation
            5. **Adjust points** and re-segment if needed
            
            ### Tips for Best Results:
            - Start with 1-2 positive points on the object you want to segment
            - Use negative points around the object boundary to refine edges
            - The model shows multiple masks - the highest score is usually the best
            - For complex objects, use more points for better accuracy
            """)

        with st.expander("⚙️ Technical Details"):
            st.markdown("""
            - **Model**: SAM ViT-H (Vision Transformer Huge)
            - **Model Size**: ~2.4GB
            - **Input**: Points (positive/negative)
            - **Output**: Binary segmentation mask
            - **Framework**: PyTorch + Streamlit
            
            The model automatically downloads from Google Drive on first use.
            """)

if __name__ == "__main__":
    main()