# **Traffic Sign Recognition** 

## Writeup Report

[//]: # (Image References)

[hist_train]: ./writeup-examples/hist_train.jpg "Training Set Representation per Label"
[hist_valid]: ./writeup-examples/hist_valid.jpg "Validation Set Representation per Label"
[hist_test]: ./writeup-examples/hist_test.jpg "Test Set Representation per Label"
[imgs_test]: ./writeup-examples/internet-examples.jpg "13 Downloaded Images for Testing"
[piecharts]: ./writeup-examples/pie_chars.jpg "Pie Charts Representing the Top 5 Predictions per Image"
[sign_color]: ./writeup-examples/sign_color.png "'Go straight or right' Traffic Sign in Color"
[sign_grey]: ./writeup-examples/sign_grey.png "'Go straight or right' Traffic Sign in GrayScale"

You're reading it! and here is a link to my [project code](https://github.com/mashraf94/Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_Project2.ipynb)

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 image
* The size of the validation set is 4410 image
* The size of test set is 12630 image
* The shape of a traffic sign image is 32x32 for 3 color channels: RGB
* The number of unique classes/labels in the data set is 43 classes


Here is an exploratory visualization of the data set using bar charts to represent the number of images for each label within the data sets. Through these representations, we'd realize that, specially the training data set, has huge variances in the density of data images per label. This had a huge impact throughout the visualization of the network's feature maps, since the activations for the low density labels were a lot more vague than others with high densities.

Training Set Representation per Label:

![alt text][hist_train]

Validation Set Representation per Label:

![alt text][hist_valid]

Test Set Representation per Label:

![alt text][hist_test]

## Data Preprocessing
1. Convert the images to grayscale using the OpenCV library:
  * Decreased the computational complexity of the model 
  * Made the model more focused on the shapes defining each traffic sign, while reducing the color noise in each image. 
    - Nonetheless, I tried running the model using the three color channels, and it was a lot slower and scored lower accuracies.

2. Used the OpenCV library for Histogram Equalization:
  * increase the global contrast of the images, enhancing the intensity distribution throughout each image.

Here is an example of a 'Go straight or right' sign image before and after grayscaling, and histogram equalization.

![alt text][sign_color]                                            ![alt text][sign_grey]

3. Tried a Gaussian Blur from OpenCV to reduce any noise in the data:
  * Reduced the performance of the network, since it reduced the resolution of the images.
  * The gaussian blur preprocessing failed to improve the network's efficiency hence was removed.
  
4. Normalizing the data using the OpenCV Library Min-Max Normalization:
  * Reduced the pixels values between -1. and 1. to centralize the data around the origin
  * Reduced the data's mean and standard deviation for better and more valuable training.
    - Manipulated the alpha=-1. and beta=1. but other variations caused a drastic change in validation and test accuracies

5. Shuffling the entire data set, and labels, using the sklearn library:
  * Randomly shuffling the data for training to attain a random distribution throughout each batch for Stochastic Gradient Descent.
  * The shuffling had a major role in the training of the model and a huge impact on the network's accuracy.


## Model Architecture
This model architecture follows the implementation of the LeNet CNN.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 --  Grayscaled image   							| 
| Convolution 5x5     	| 1x1 stride + VALID padding + outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride + outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride + VALID padding + outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride + outputs 5x5x16 				|
| Fully connected		| 120 unit        									|
| RELU					|												|
| Dropout				| 0.5 keep_prob        									|
| Fully connected		| 83 unit        									|
| RELU																	|
| Dropout				| 0.5 keep_prob        									|
|	Output					|	43 Logits											|
 


## Model Training
### Tuned Parameters
The choice of these hyperparameters required several trials to reach this final combination which represents the maximum performance achieved.
1. Epochs # 15   --------- Chosen upon the fact that the model reaches a plateau by the 15th epoch
2. Batch Size # 128  --------- Appropriate and efficient batch size     
3. Learning Rate = 0.001 --------- Through many tests, this learning rate was right before overshooting, nonetheless fast and converges
4. Mean = 0.  &  Standard Deviation = 0.1 --------- Values fed for the tf.truncated_normal() function for weight initialization 
5. Dropout = 0.5 --------- The probability for the dropout layers which decreased vastly the overfitting of the dataset

* Used the tf.nn.softmax_cross_entropy_with_logits() function to calculate the logits probabilities using: softmax + the cross entropy 
* Used the Adam Optimizer for training the network with backpropagation and stochastic gradient descent.


My final model results were:
* Validation Set Accuracy = 95.44 % 
* Test Set Accuracy = 93.51 %

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
1. Implemented the LeNet convolutional neural network, for it's simplicity and low computational cost.
    * Produced low accuracies and incompetent performance, due to overfitting of the training dataset.
2. Implemented the VGG "Visual Geometric Group" convolutional network, which is known for it's state-of-art performance
   //INSERT VGG .ipynb file
    * The VGG implementation acquired significantly higher performance than LeNet with colored photos
    * VGG was significantly expensive computationally for its complexity since it had wider layers and it was deeper than LeNet 
           + VGG could only run on an AWS GPU instance.
3. Although VGG is a far more powerful network than LeNet it was costy, therefore I tried manipulating the dataset and LeNet implementation to achieve a close performance to VGG with a way more simplistic model.
     * **Steps**:
          a. Changed images to grayscale, and applied a histogram equalization so the network could focus on the main features in each traffic sign
          b. Through an iterative approach, used several normalization techniques to acquire the highest performance of the network.
          c. Reduced overfitting by adjusting the network and introducing 2 dropout layers following each fully connected layer.
          d. Tried introducing a third convolutional layer to the network, but it didn't affect the network's performance at all.
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

## Testing the Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are thirteen German traffic signs that I found on the web:

![alt text][imgs_test]

The first image might be difficult to classify because it is filling the whole image extensively which looks nothing like the training images, however this tests the networks capabilities of identifying the word 'STOP'.

The second, fourth and tenth images are difficult since all of them have the signs skewed with a different angle, and not straight forward like the training data set.

The fifth, seventh, eighth and ninth images contain intense color noise in the background and had either dirt or an irregularity on top of the sign in the image 

The third, eleventh, twelfth and thirteenth image all had high detailed traffic signs with complicated symbols that could be hard to determine using a simple network as LeNet.

The sixth image is a 'Right Ahead' sign which is blue, and the image shows the sign with a blue similar background, which might be challenging for the network to determine the edges of the sign.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| General Caution     			|  General Caution  										|
| Road Work					| Road Work											|
| Right-of-way at the next intersection |  Right-of-way at the next intersection |
| Bumpy Road | Bumpy Road |
| Turn Right Ahead | Turn Right Ahead |
| Speed Limit (30km/h) | Speed Limit (30km/h) |
| Roundabout Mandatory | Roundabout Mandatory | 
| Priority Road | Priority Road |
| Yield					| Yield											|
| Wild Animals Crossing | Wild Animals Crossing |
| Beware of ice/snow | Beware of ice/snow |
| Pedestrians | Pedestrians |


The model was able to correctly guess 12 of the 13 traffic signs, which gives an accuracy of 92.3 %. This compares favorably to the accuracy on the test set of 93.5 %

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

![alt text][piecharts]

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|




