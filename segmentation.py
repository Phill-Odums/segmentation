import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import gdown

import gdown




# Configure the page
st.set_page_config(
    page_title="SAM Image Segmenter",
    page_icon="🎯",
    layout="wide"
)

@st.cache_resource
def load_sam_model():
    """Load and cache the SAM model to avoid reloading on every interaction"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_type = "vit_b"  # Changed to match your checkpoint file
    
    # Update this path to your actual checkpoint file location
    checkpoint_path = "C:/Users/phill/.spyder-py3/projects/segmentation/sam_vit_b_01ec64.pth"
    
    try:
        # Load the model with proper registry
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, device

# Usage
predictor, device = load_sam_model()
if predictor is not None:
    print("SAM model loaded successfully!")
else:
    print("Failed to load SAM model")

def main():
    st.title("🎯 Segment Anything Model (SAM)")
    st.markdown("Upload an image and click on points to segment objects")

    # Load model
    with st.spinner("Loading SAM model..."):
        predictor, device = load_sam_model()

    if predictor is None:
        st.error("Failed to load SAM model. Please check the model path.")
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

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            # Create a clickable image
            st.markdown("Click on objects in the image to segment them:")

            # Convert image for display
            display_image = image_array.copy()

            # Use st.image with a key to make it clickable
            click_data = st.image(
                display_image,
                use_column_width=True,
                caption="Click to add positive points (green). Right-click for negative points (red)."
            )

            # Initialize session state for points and labels
            if 'points' not in st.session_state:
                st.session_state.points = []
            if 'labels' not in st.session_state:
                st.session_state.labels = []

            # Point collection interface
            st.markdown("**Point Controls:**")
            col1a, col1b, col1c = st.columns(3)

            with col1a:
                if st.button("Add Positive Point"):
                    # For demonstration, we'll use click coordinates
                    # In a real app, you'd get these from actual clicks
                    st.info("Click on the image above to add points")

            with col1b:
                if st.button("Add Negative Point"):
                    st.info("Right-click on the image above to add negative points")

            with col1c:
                if st.button("Clear Points"):
                    st.session_state.points = []
                    st.session_state.labels = []
                    st.rerun()

            # Display current points
            if st.session_state.points:
                st.write(f"Points: {len(st.session_state.points)} positive, "
                        f"{len([l for l in st.session_state.labels if l == 0])} negative")

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
                    masks, scores, logits = predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        multimask_output=True,
                    )

                    # Select the best mask
                    best_mask_idx = np.argmax(scores)
                    mask = masks[best_mask_idx]

                    # Create overlay visualization
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    ax.imshow(image_array)
                    ax.imshow(mask, alpha=0.5, cmap='viridis')

                    # Plot points
                    for point, label in zip(st.session_state.points, st.session_state.labels):
                        color = 'green' if label == 1 else 'red'
                        marker = 'o' if label == 1 else 'x'
                        ax.scatter(point[0], point[1], c=color, marker=marker, s=100,
                                  edgecolors='white', linewidth=2)

                    ax.set_title(f"Segmentation (Score: {scores[best_mask_idx]:.3f})")
                    ax.axis('off')

                    st.pyplot(fig)

                    # Display mask score
                    st.write(f"Mask confidence score: {scores[best_mask_idx]:.3f}")

                    # Option to download mask
                    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                    st.download_button(
                        label="Download Mask",
                        data=mask_image.tobytes(),
                        file_name="segmentation_mask.png",
                        mime="image/png"
                    )

                except Exception as e:
                    st.error(f"Error during segmentation: {e}")
            else:
                st.info("Add points to the image to see segmentation results")

                # Show placeholder image
                placeholder_fig, placeholder_ax = plt.subplots(1, 1, figsize=(10, 10))
                placeholder_ax.imshow(image_array)
                placeholder_ax.set_title("Segmentation will appear here")
                placeholder_ax.axis('off')
                st.pyplot(placeholder_fig)

        # Instructions
        with st.expander("How to use this app"):
            st.markdown("""
            1. **Upload an image** using the file uploader
            2. **Click on objects** in the left image to add positive points (green)
            3. **Right-click** to add negative points (red) for areas that are NOT part of the object
            4. **View results** in the right panel
            5. **Clear points** if you want to start over

            **Tips:**
            - Start with 1-2 positive points on the object you want to segment
            - Use negative points to refine the segmentation
            - The model will automatically select the best mask from multiple candidates
            """)

        # Alternative: Manual point input for cases where click detection is challenging
        with st.expander("Manual Point Input (Alternative)"):
            st.markdown("If click detection doesn't work well, you can manually enter coordinates:")

            col_manual = st.columns(2)
            with col_manual[0]:
                manual_x = st.slider("X coordinate", 0, image_array.shape[1], image_array.shape[1] // 2)
            with col_manual[1]:
                manual_y = st.slider("Y coordinate", 0, image_array.shape[0], image_array.shape[0] // 2)

            col_manual_btn = st.columns(2)
            with col_manual_btn[0]:
                if st.button("Add as Positive Point"):
                    st.session_state.points.append([manual_x, manual_y])
                    st.session_state.labels.append(1)
                    st.rerun()
            with col_manual_btn[1]:
                if st.button("Add as Negative Point"):
                    st.session_state.points.append([manual_x, manual_y])
                    st.session_state.labels.append(0)
                    st.rerun()

if __name__ == "__main__":
    main()


