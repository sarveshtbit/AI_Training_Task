import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("digit_cnn.keras")

st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit.")

uploaded_file = st.file_uploader(
    "Upload digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=200
    )

    # Convert to grayscale
    image = image.convert("L")

    # Resize
    image = image.resize((28, 28))

    # Convert to numpy
    image_array = np.array(image)

    # Normalize
    image_array = image_array.astype("float32") / 255.0

    # Reshape
    image_array = image_array.reshape(
        1, 28, 28, 1
    )

    # Prediction
    probabilities = model.predict(
        image_array,
        verbose=0
    )

    prediction = np.argmax(
        probabilities
    )

    confidence = np.max(
        probabilities
    )

    st.subheader(
        f"Predicted Digit: {prediction}"
    )

    st.write(
        f"Confidence: {confidence:.2%}"
    )

    # Probability chart
    probabilities = probabilities[0]

    st.subheader("Prediction Probabilities")

    st.bar_chart(
        probabilities
    )