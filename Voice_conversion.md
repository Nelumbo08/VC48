## Voice Conversion

Voice conversion is a process of converting the voice of one person to the voice of another person, keeping the linguistic content the same. We are working with high fidelity speech data with a lot of variability in terms of phonetic content. the target for the voice conversion model is to have a high quality voice conversion with a low amount of artifacts, enabled by a high quality speaker embedding. The learning rule could be adversarial or diffusion keeping in mind that the training time must not be very steep and the models should converge easily on the probability distribution of the vocal texture of the target speaker. 


### Generator architecture 

The generator should follow an UNET type of architecture that downsamples the mel spectrogram input to a latent space that has two parts, the speaker dependent and the speaker independent part. The speaker dependent part of the latent space would be used later to model the probability distribution of the target domain and the source domain, so as to match the input distribution to that of the target dmain through iterative updates enabled by teacher forcing. The speaker independent (content domain) latent space is not to be altered but to be forced to stay the same as that of the source domain. the separation of the informations could be employed by using convolutive non negative matrix factorization based autoencoders/deep-learning methods. The whole architecture should be convolutive with varied receptive fields to capture the temporal information of the speech. Use of vision transformers or conformers for global and local conrtext is suggested. The architecture can also be a replication of GANILLA (https://github.com/giddyyupp/ganilla) gan for image style transfer. Adapt the architecture in a way it can also replicate the Masked-CycleGAN VC for the inpainting capabilities. There should be a mix of 2d convolution and then followed by 1d convolutio with gated linear units for information flow. The convolutions in the downsampling part should be replicated as it is in the upsampling block with matching features only replaced by transpose convolutions. 

### Discriminator architecture

The discriminator should be a convolutional neural network that takes the mel spectrogram as input and outputs a scalar value that indicates the probability of the input being real or fake. The discriminator should be able to distinguish between the source domain and the target domain. employ capsule based architecture for the final layer with the routing done through attention instead of the routing by agreement, so that the discriminator can learn the spectral features with respect to the occurance and also angle with a global context. 

Use a multiperiod discriminator as in HiFiGAN so as to ensure proper temporal information retention. Keep the discriminators and generators parameters and multiplication-additions as close as possible so as to ensure there is no disparity during the learning process. 

## Loss functions

use feature specific loss functions like perceptual loss, adversarial loss, cycle consistency loss, identity loss, style loss, etc. Enforce a loss function that calculates the logF0 RMSE between the source and target domain. employ a cross entropy based loss function for the adversarial training. The training rule must be mathematically sound and important loss funtions should be differentiable for backward propagation. create a balance between the loss function with the aim of keeping the process stable and also ensuring that the loss function is not too steep. enforce an intermidiate embedding loss based on convergence that would enforce the generator to match the source domain speaker dependent embedding to that of the target domain speaker embedding (like variational autoencoder) 

## Training

Train the model with various modes of training like supervised, unsupervised, semi-supervised, etc. Use of teacher forcing is suggested for the training of the generator. Make the training process robust so as to keep it zero-shot.


