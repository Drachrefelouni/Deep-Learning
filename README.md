## CNN-Architectures: Descriptions of each architecture 

1. **EfficientNet V1**
   - Short Description: A highly efficient convolutional neural network architecture.
   - Reference: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

2. **EfficientNet V2**
   - Short Description: An improved version of the EfficientNet architecture aimed at enhancing performance while maintaining resource efficiency.
   - Reference: [Going Deeper with Image Transformers](https://arxiv.org/abs/2103.17239)

3. **GoogLeNet (InceptionNet)**
   - Short Description: A convolutional neural network architecture known for introducing the Inception module for efficient feature extraction at different scales.
   - Reference: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

4. **LeNet**
   - Short Description: An early convolutional neural network architecture designed by Yann LeCun et al. for handwritten digit recognition.
   - Reference: [Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/lenet/)

5. **MobileNet V1**
   - Short Description: A lightweight convolutional neural network architecture optimized for mobile devices and resource-constrained applications.
   - Reference: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

6. **MobileNet V2**
   - Short Description: An enhanced version of the MobileNet V1 architecture focusing on improving performance and resource efficiency.
   - Reference: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

7. **MobileNet V3**
   - Short Description: A further improved version of the MobileNet series, emphasizing performance optimization using inverted residuals and linear bottlenecks.
   - Reference: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

8. **ResNeXt**
   - Short Description: A convolutional neural network architecture extending the ResNet residual block concept by introducing cardinality for improved feature representation.
   - Reference: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

9. **SeNet**
   - Short Description: A convolutional neural network architecture incorporating squeeze-and-excitation blocks for dynamic feature attention adaptation.
   - Reference: [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

10. **VGG**
    - Short Description: A convolutional neural network architecture characterized by a series of convolutional and pooling layers, known for its simplicity and effectiveness.
    - Reference: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

## AutoEncoder Architectures: Descriptions of each implementation


1. **AVE_Denoising_Autoencoder.py**
   - Short Description: Script implementing an autoencoder for denoising images using the AVE method.
   - Reference:[Denoising Adversarial Autoencoders](https://ieeexplore.ieee.org/abstract/document/8438540)

2. **AVE_for_Face_Images.py**
   - Short Description: Script implementing an autoencoder for face image processing using the AVE method.

3. **AVE_for_Face_Smile.py**
   - Short Description: Script implementing an autoencoder for detecting smiles in face images using the AVE method.

4. **AVE_for_Handwritten_Digits.py**
   - Short Description: Script implementing an autoencoder for handwritten digit recognition using the AVE method.

5. **AutoEncoder_Linear.py**
   - Short Description: Script implementing a linear autoencoder.

6. **AutoEncoder_Linear_SoftMax.py**
   - Short Description: Script implementing a linear autoencoder with softmax activation.

7. **AutoEncoders_ConV.py**
   - Short Description: Script implementing convolutional autoencoders.
   - Reference: Not provided.

8. **VAE_gumbel_softmax.py**
   - Short Description: Script implementing a variational autoencoder with the Gumbel-Softmax reparameterization trick.
   - Reference:[Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/pdf/1611.01144.pdf)

9. **VA_AutoEncoder_MNIST.py**
    - Short Description: Script implementing a variational autoencoder for MNIST digit generation.
    
## Diffusion-model: Descriptions of each implementation    

1. **DDPM.py**
   - Short Description: Script implementing a Diffusion Probabilistic Model (DDPM) for unsupervised learning tasks.
   - Reference:[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)

2. **DDPM_Conditional.py**
   - Short Description: Script implementing a conditional Diffusion Probabilistic Model (DDPM) for conditional image generation tasks.
   - Reference:[ Conditioning Method for Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2108.02938.pdf)
   
   
## Generative-Adversarial-Networks (GAN): Descriptions of each implementation    

Here are short descriptions of each file for your README.md:

1. **Conditional_GAN.py**
   - Short Description: Script implementing a conditional generative adversarial network (GAN) for conditional image generation tasks.
   - Reference:[Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)

2. **DCGAN_CELIBA.py**
   - Short Description: Script implementing a Deep Convolutional GAN (DCGAN) for generating celebrity face images.
   - Reference:[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)

3. **DCGAN_Cifar.py**
   - Short Description: Script implementing a Deep Convolutional GAN (DCGAN) for generating CIFAR-10 images.

4. **DCGAN_MNIST.py**
   - Short Description: Script implementing a Deep Convolutional GAN (DCGAN) for generating MNIST digit images.

5. **Simple_GAN_V1.py**
   - Short Description: Script implementing a simple GAN architecture (version 1) for generating images.
   - Reference:[Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf)

6. **Simple_GAN_V2.py**
   - Short Description: Script implementing a simple GAN architecture (version 2) for generating images.
   - Reference:[Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf)
   
7. **Simple_GAN_tensorboard.py**
   - Short Description: Script implementing a simple GAN architecture with tensorboard visualization support.

8. **WGAN_MNIST.py**
   - Short Description: Script implementing a Wasserstein GAN (WGAN) for generating MNIST digit images.
   - Reference:[Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)
 
## Computer Vision Applications: Descriptions for each script:

1. **CNN_tensorBord_Pytorch.py**
   - Description: This script implements a Convolutional Neural Network (CNN) using the PyTorch framework with integration for TensorBoard, facilitating visualization of model training metrics such as loss and accuracy over time. It provides a useful tool for monitoring the training process and diagnosing potential issues in model performance.


2. **Example Classification.py**
   - Description: This script serves as an illustrative example of image classification using a neural network model. It demonstrates the process of loading data, defining a neural network architecture, training the model, and evaluating its performance on a classification task. It is intended to provide a practical demonstration of how to implement a simple classification pipeline using Python and popular deep learning libraries.


3. **Init_weights.py**
   - Description: This Python script defines a Convolutional Neural Network (CNN) architecture using PyTorch. It includes convolutional and pooling layers, as well as fully connected layers for classification. The script initializes the network's weights and biases using the Kaiming initialization method and provides a method for forward pass computation. Additionally, it allows for inspection of the model's parameters. This CNN architecture serves as a foundational building block for various image classification tasks, offering a flexible and customizable framework for deep learning projects.


4. **Neural Style Transfer (Pytorch).py**
   - Description: The provided Python script utilizes PyTorch to perform neural style transfer. It employs a pre-trained VGG19 model to extract features from content and style images, optimizing a generated image to minimize content and style losses. The script iteratively updates the generated image using the Adam optimizer and periodically saves the result.

5. **Progress_bar.py**
   - Description: This script illustrates the implementation of a progress bar for monitoring training progress during the training of machine learning models. A progress bar is a visual indicator that provides real-time feedback on the progress of a long-running task, such as model training. By displaying the progress of epochs or batches, it helps users track the training process, estimate remaining time, and identify potential issues such as slow convergence or overfitting.


6. **Transfer Learning and Fine Tuning VGG16.py**
   - Description: This Python script utilizes PyTorch to perform neural style transfer with support for parallel processing on multiple GPUs. It employs a pre-trained VGG19 model to extract features from content and style images and optimizes a generated image using the Adam optimizer to minimize content and style losses. The script iteratively updates the generated image and periodically saves it. It leverages CUDA for GPU acceleration and supports parallel processing across multiple GPUs using PyTorch's DataParallel module. Overall, it provides an efficient and scalable implementation of neural style transfer.


7. **save_load.py**
   - Description: This PyTorch script illustrates the process of saving and loading model checkpoints. It initializes a VGG16 model and an Adam optimizer for training, saves a checkpoint, and then loads and applies the checkpoint. This functionality ensures the preservation of training progress and model parameters, facilitating efficient experimentation and deployment.
 
 
Contact: achref.el.ouni@outlook.fr
 
