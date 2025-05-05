import streamlit as st
import grpc
import io
import os
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime
import base64
import text_to_image_pb2
import text_to_image_pb2_grpc
import pandas as pd
import numpy as np
import cv2
import random
from io import BytesIO

# Set Streamlit page configuration
st.set_page_config(page_title="Da Vinchi", layout="wide")
SAVE_DIR = "saved_images"
os.makedirs(SAVE_DIR, exist_ok=True)

@st.cache_resource
def get_grpc_stub():
    channel = grpc.insecure_channel('localhost:6000')
    return text_to_image_pb2_grpc.TextToImageStub(channel)

stub = get_grpc_stub()

# --- Image processing functions ---

def apply_filter(img, filter_name):
    if filter_name == "None":
        return img
    elif filter_name == "Sketch":
        return img.filter(ImageFilter.CONTOUR)
    elif filter_name == "Warm Tone":
        return ImageEnhance.Color(img).enhance(1.5)
    elif filter_name == "Cool Tone":
        arr = np.array(img)
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.9, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))
    elif filter_name == "Glitch":
        arr = np.array(img)
        arr[:, :, 1] = np.roll(arr[:, :, 1], random.randint(-50, 50))
        return Image.fromarray(arr)

def adjust_image(img, brightness=1.0, contrast=1.0, saturation=1.0):
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    return img

def find_caption_by_tags(df, keywords):
    for _, row in df.iterrows():
        tags = [tag.lower() for tag in row['tags']] if 'tags' in row else []
        if any(k in tags for k in keywords):
            quote = row.get('quote', "").strip('“”"')
            author = row.get('author', "").strip()
            return f"\"{quote}\" — {author}"
    return "No matching quote found."

# --- Session state initialization ---
for key in ["liked_images", "selected_image", "last_generated_image", "current_caption"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "liked_images" else []

# --- Load captions dataset ---
try:
    df = pd.read_json("hf://datasets/Abirate/english_quotes/quotes.jsonl", lines=True)
except:
    df = pd.DataFrame(columns=['tags', 'quote', 'author'])

def refresh_timeline():
    try:
        imgs = [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith(('.png', '.jpg'))]
        imgs.sort(reverse=True)
        st.session_state.liked_images = imgs
    except Exception as e:
        st.error(f"Error loading timeline: {e}")

# --- Layout ---
col_timeline, col_main = st.columns([0.2, 0.8])

with col_timeline:
    st.markdown("### 🕒 Timeline")
    if st.session_state.liked_images:
        cols = st.columns(2)
        for i, path in enumerate(st.session_state.liked_images):
            cols[i % 2].image(path, use_container_width=True, caption=os.path.basename(path))
    else:
        st.info("No images in timeline yet.")

with col_main:
    st.markdown("### 🎨 Generate Image")
    with st.form("image_form"):
        prompt = st.text_input("Prompt", "a sunset over the ocean")
        col1, col2, col3 = st.columns(3)
        with col1: height = st.number_input("Height", 64, step=64)
        with col2: width = st.number_input("Width", 64, step=64)
        with col3: steps = st.slider("Steps", 5, 50, 25)
        guidance = st.slider("Guidance", 1.0, 15.0, 7.5)
        dtype = st.selectbox("Precision", ["float32", "float16"], 1)
        submit = st.form_submit_button("🚀 Generate")

    if submit:
        if width % 8 != 0 or height % 8 != 0:
            st.error("Width and height must be multiples of 8.")
        else:
            with st.spinner("Generating..."):
                try:
                    req = text_to_image_pb2.ImageRequest(
                        prompt=prompt, height=height, width=width,
                        steps=steps, guidance=guidance, dtype=dtype
                    )
                    res = stub.GenerateImage(req)
                    if res.status == "success":
                        img_bytes = base64.b64decode(res.base64_image)
                        img = Image.open(io.BytesIO(img_bytes))
                        st.session_state.last_generated_image = img
                        st.session_state.current_caption = find_caption_by_tags(df, prompt.lower().split())
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- Show image and caption ---
    if st.session_state.last_generated_image:
        st.image(st.session_state.last_generated_image, caption="Generated Image", use_container_width=True)
        st.markdown(f"*Caption:* {st.session_state.current_caption}")

        if st.button("❤ Save to Timeline"):
            try:
                name = f"liked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                path = os.path.join(SAVE_DIR, name)
                st.session_state.last_generated_image.save(path)
                st.success("Image saved!")
                refresh_timeline()
            except Exception as e:
                st.error(f"Error saving: {e}")

        st.markdown("---")
        st.markdown("### ✏ Edit Image")

        with st.expander("🎛 Editing Tools", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                filter_choice = st.selectbox("Filter", ["None", "Sketch", "Oil Painting", "Warm Tone", "Cool Tone", "Glitch"])
            with col2:
                brightness = st.slider("Brightness", 0.5, 1.5, 1.0, 0.1)
                contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
                saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.1)

        # Apply edits on the original generated image
        edited_img = apply_filter(st.session_state.last_generated_image.copy(), filter_choice)
        edited_img = adjust_image(edited_img, brightness, contrast, saturation)
        st.image(edited_img, caption="Edited Image", use_container_width=True)

        # Save/download options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Edited Version"):
                try:
                    name = f"edited_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    path = os.path.join(SAVE_DIR, name)
                    edited_img.save(path)
                    st.success("Edited image saved!")
                    refresh_timeline()
                except Exception as e:
                    st.error(f"Failed to save: {e}")
        with col2:
            buf = BytesIO()
            edited_img.save(buf, format="PNG")
            st.download_button("⬇ Download", buf.getvalue(), "edited_image.png", "image/png")

# Refresh timeline on initial load
if not st.session_state.liked_images:
    refresh_timeline()