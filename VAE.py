import streamlit as st

# Set page title and icon
st.set_page_config(page_title='VAE AI Model', page_icon='🤖')

# Define page content


def main():
    st.markdown('<h1 style="text-align:center;">Variational Autoencoder (VAE)</h1>',
                unsafe_allow_html=True)

    st.markdown('## 🤔 What is a VAE?')
    st.write('A VAE is a type of deep generative model that is capable of learning the underlying structure of complex data. It consists of an encoder network that maps the input data to a low-dimensional latent space, and a decoder network that generates the output data from the latent representation.')

    st.markdown('## 🎯 What can a VAE be used for?')
    st.write('VAEs are used in a wide range of applications, including image and speech recognition, natural language processing, and anomaly detection. They are particularly useful in situations where there is a need for unsupervised learning, i.e., when there is no labeled training data available.')

    st.markdown('## 🧐 How does a VAE work?')
    st.write('The VAE model is trained using a technique called "variational inference", which involves minimizing the difference between the true data distribution and the model\'s distribution over the latent space. This is done using a combination of gradient descent and a special type of loss function called the "KL divergence".')

    st.markdown('## 👩‍💻 Ready to try it out?')
    st.write('We\'ve included two VAE models that you can use on two differents datasets: MNIST and FREY FACE.')


main()
