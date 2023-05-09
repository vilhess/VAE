# VAE AI Model with Streamlit

This is a simple application that showcases the Variational Autoencoder (VAE) model applied on two datasets: MNIST and FREY FACE. The application is built using Streamlit, a Python library for building interactive web applications.

## What is a VAE?

A VAE is a type of deep generative model that is capable of learning the underlying structure of complex data. It consists of an encoder network that maps the input data to a low-dimensional latent space, and a decoder network that generates the output data from the latent representation.

## What can a VAE be used for?

VAEs are used in a wide range of applications, including image and speech recognition, natural language processing, and anomaly detection. They are particularly useful in situations where there is a need for unsupervised learning, i.e., when there is no labeled training data available.

## How does a VAE work?

The VAE model is trained using a technique called "variational inference", which involves minimizing the difference between the true data distribution and the model's distribution over the latent space. This is done using a combination of gradient descent and a special type of loss function called the "KL divergence".

## MNIST Dataset

In this section, we use the VAE model to generate new digits. First, we display some examples of images that were used to train our model. Then, we show where each digit resides in our latent space. Using this graph, we can generate new digits by selecting coordinates of our choice.

## FREY FACE Dataset

In this section, we apply the VAE model to faces of the same man with different facial expressions. The VAE will approximate the distribution of each expression and generate new ones for this man. We use a 6D latent space to generate new faces.

## Drawing Recognition

In this bonus section, we utilize a convolutional neural network to recognize hand-drawn digits. The user can draw a digit on a canvas, and the model predicts what the digit is.

## How to run the application

1. Clone the repository to your local machine
2. Install the required dependencies using `pip install -r requirements.txt`
3. Run the application using `streamlit run VAE.py`
