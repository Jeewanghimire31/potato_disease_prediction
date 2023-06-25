## Model
Here, the Architecture of the CNN model works in two phases.
### Feature Extraction
#### I. Input Layer
 It’s the layer in which we give input to our model. In CNN, Generally the input will be an image or sequence of images. This layer holds the raw input of the image. This layer holds the raw input of the image with width 32, height 32 and depth 3 and so on.
#### II. Convolution Layers
This is the layer, which is used to extract the feature from the input dataset. It applies a set of learnable filters known as the kernels to the input images. The filters/kernels are smaller matrices, usually 2×2, 3×3, or 5×5 shapes. it slides over the input image data and computes the dot product between kernel weight and the corresponding input image patch. The output of this layer is referred to as feature maps. Suppose we use a total of 5 filters for this layer we’ll get an output volume of dimension 32 x 32 x 5.
#### III. Activation Layer
By adding an activation function to the output of the preceding layer, activation layers add nonlinearity to the network. For Example, the Relu activation function (max(0, X)) sets the negative values to 0 and keeps positive values unchanged. The output values will still retain the original range of the input data, But some values may be transformed or filtered based on the activation function.
#### IV. Pooling Layer
Its main function is to reduce the size of volume which makes the computation fast, reduces memory and also prevents overfitting. Two common types of pooling layers are max pooling and average pooling. If we use a max pooling with 2 * 2 filters and stride 2, the resultant volume will be of dimension 16 x 16 x 5. Here, strides determine how much the pooling window moves horizontally and vertically after each pooling operation.
### Classification or Regression
#### I. Flattening
After the convolution and pooling layer, all the features are mapped into a one dimensional vector so that they can be passed into a completely linked layer.
#### II. Fully Connected Layer
In this layer,  it will take input as the output of the feature extraction and will compute the final classification or regression task. In this layer, each neuron in the previous layer, and the output of each neuron is computed using a set of weights and biases. This allows the network to learn complex relationships between the features extracted from the previous layers and the classification or regression. The output of fully connected layer is then used for classification e.g through a softmax function for multi-class classification or regression e.g through a linear activation function for continuous output values.



#### All the steps of Feature Extraction and Classification or Regression are present in the CNN model.


## Datasets
The Dataset can be easily available in kaggle. In the process of Data manipulation and Data Preprocessing if the data gets in large amounts we can use the concept of Batch Size.

### Batch size
The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters. Batch size defines the number of samples we use in one epoch to train a neural network.<br>
For example, let’s consider the training size of 1000 samples and the batch size of 100. A neural network will take the first 100 samples in the first epoch and do forward and backward propagation. After that, it’ll take the subsequent 100 samples in the second epoch and repeat the process.Overall, the network will be trained for the predefined number of epochs or until the desired condition is not met.


### Epochs
An epoch is when all the training data is used at once and is defined as the total number of iterations of all the training data in one cycle for training the machine learning model.<br>
For example, Consider a dataset that has 200 samples. These samples take 1000 epochs or 1000 turns for the dataset to pass through the model. It has a batch size of 5. This means that the model weights are updated when each of the 40 batches containing five samples passes through. Hence the model will be updated 40 times.

## API
In this i have used the FastAPI concepts and while running this you will get the predicted value if you give the potato leaf picture. It is working fine give it a try and if there is any issuse you can open the pull request. 
