import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision.utils import save_image
from function_vae.coord_to_img import convert_to_img_without_show_mnist, convert_to_img_without_show_frey_face
from function_vae.generation import generate_mnist
from streamlit_drawable_canvas import st_canvas
from function_vae.recognizer import recognition_digit
import torchvision.transforms as transforms
import torch.nn as nn
from models import *


def page2():
    model = torch.load("function_vae/save_weights/weightscnn")
    model.eval()

    st.markdown("<h1 style='text-align: center;'>Variational Autoencoder on MNIST Dataset</h1>",
                unsafe_allow_html=True)

    st.subheader(" ")

    st.markdown("<h3 style='text-align: center;'>Displayed below are some examples of images that were used to train our model.</h3>",
                unsafe_allow_html=True)
    st.subheader(" ")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        img = Image.open("function_vae/images/some_mnist_images/img0.png")
        st.image(img, use_column_width=True)
        img = Image.open("function_vae/images/some_mnist_images/img5.png")
        st.image(img, use_column_width=True)
    with col2:
        img = Image.open("function_vae/images/some_mnist_images/img1.png")
        st.image(img, use_column_width=True)
        img = Image.open("function_vae/images/some_mnist_images/img6.png")
        st.image(img, use_column_width=True)
    with col3:
        img = Image.open("function_vae/images/some_mnist_images/img2.png")
        st.image(img, use_column_width=True)
        img = Image.open("function_vae/images/some_mnist_images/img7.png")
        st.image(img, use_column_width=True)
    with col4:
        img = Image.open("function_vae/images/some_mnist_images/img3.png")
        st.image(img, use_column_width=True)
        img = Image.open("function_vae/images/some_mnist_images/img8.png")
        st.image(img, use_column_width=True)
    with col5:
        img = Image.open("function_vae/images/some_mnist_images/img4.png")
        st.image(img, use_column_width=True)
        img = Image.open("function_vae/images/some_mnist_images/img9.png")
        st.image(img, use_column_width=True)

    st.subheader("")

    st.markdown("<h3 style='text-align: center;'>A VAE can approximate the distribution of our data, and below we can see where each digit resides in our latent space.</h3>",
                unsafe_allow_html=True)

    image = Image.open("function_vae/images/space.png")
    st.image(image, caption="2D dimension cluster", use_column_width=True)
    st.header(" ")

    st.markdown("<h3 style='text-align: center;'>Using this graph, we can generate new digits by selecting coordinates of our choice.</h3>",
                unsafe_allow_html=True)

    st.header(" ")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('')
        st.subheader('')
        st.subheader('')
        coord1 = st.slider("coord1", float(-6), float(6), float(0), step=0.1)
        coord2 = st.slider("coord2", float(-6), float(6), float(0), step=0.1)
    with col2:
        img = convert_to_img_without_show_mnist((coord1, coord2))
        save_image(img, "function_vae/images/some_mnist_images/made.png")
        img = Image.open("function_vae/images/some_mnist_images/made.png")
        st.image(img, caption="generated", use_column_width=True)

    st.header(" ")
    st.header(" ")

    st.markdown("<h3 style='text-align: center;'>BONUS : This section does not use our VAE, but instead utilizes a convolutional neural network. You can draw a digit and see what the model predicts. </h3>",
                unsafe_allow_html=True)

    drawing_mode = "freedraw"

    stroke_width = 10

    stroke_color = "#ffffff"

    bg_color = "#000000"

    realtime_update = True

    st.subheader("")

    col1, col2 = st.columns((0.1, 1))
    with col2:
        canvas_result = st_canvas(
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            drawing_mode=drawing_mode,
            key="canvas",
            width=560,
            height=560,
            update_streamlit=realtime_update,
        )
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data,
                              "RGBA")

        pred = recognition_digit(img)
        st.markdown(f"<h3 style='text-align: center;'>Based on the input provided, the convolutional neural network predicted that the drawn digit is a {pred} </h3>",
                    unsafe_allow_html=True)


page2()
