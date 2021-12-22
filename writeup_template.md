# **Traffic Sign Recognition** 

**Building a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./.images/dataset_visualization.png "Dataset Visualization"
[image2]: ./.images/grayscale.png "Grayscaling"
[image3]: ./.images/accuracy-loss.png "Accuracy/Loss Visualization"
[image4]: ./.images/web_images.png "Traffic Signs"
[image5]: ./.images/activation_maps1.png "Activation Maps1"
[image6]: ./.images/activation_maps2.png "Activation Maps2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it, and here is a link to my [project code](https://github.com/mrrocketraccoon/GermanTrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration
#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is of 34799 samples
* The size of the validation set is of 4410 samples
* The size of test set is of 12630 samples
* The shape of a traffic sign image is of 32x32x3
* The number of unique classes/labels in the data set is 42

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the different classes of traffic signs are represented in the dataset. We clearly see that we are dealing with imbalanced classes, there are many traffic sign classes that have under 500 samples while some have almost 2000. At this point it would make sense to implement a tactic to combat this issue. To mention a couple of alternatives, we could add copies of under-represented classes or remove copies from over-represented classes. We could also augment the dataset by performing small rotations to the images. Since fixing this issue is not part of the rubric it has not been addressed in this project.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques?

As a first step in the preprocessing of the data the images have been converted to grayscale, so that a single channel is fed into the data pipeline, then the samples have been normalized so that the data has mean zero and equal variance, this is important so that we train with samples that contribute equally to our analysis while avoiding a potential bias in our model.
Here is an example of the Speed Limit 20 Km/h sign image before and after grayscaling.

![alt text][image2]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The LeNet classifier has been used as a baseline for the model architecture, the final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   			        | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flattening            | outputs 400                                   |
| Dropout				| keep probability 0.7   						|
| Fully connected		| outputs 120        							|
| RELU					|												|
| Dropout				| keep probability 0.7   						|
| Fully connected		| outputs 84        							|
| RELU					|												|
| Dropout				| keep probability 0.7   						|
| Fully connected		| outputs 43        							|
| RELU					|												|
| Dropout				| keep probability 0.7   						|
| Softmax				|           									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the adam optimizer with following hyperparameters.
| Layer         		|     Description	| 
|:---------------------:|:-----------------:| 
| Batch size         	| 128 	            |
| Epochs				| 60	            |
| Learning rate     	| 0.001             |


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.959
* test set accuracy of 0.948

The chosen architecture to be implemented was LeNet, it was simple enough to code and there is previous work demonstrating its usefulness when classifying traffic signs.
The initial architecture could not get an accuracy greater than 0.7, that is why dropout was added to the fully connected layers to avoid overfitting and thus improve the generalization of the classifier.


![alt text][image3]


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]

I have chosen traffic signs that have similar characteristics and are under-represented in the dataset to test the robustness of the classifier. Images with labels 24, 27 and 26 have a triangular red border that might make some signs difficult to differentiate from each other, especially when working with a reshaped 32x32 pixel image where a considerable amount of information has been lost. Signs 34 and 40 have a blue circle with arrows indicating directions, as in the previous case it is a potential problem for the neural network if signs are relatively similar and few samples are available for the classes.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                |     Prediction	        			| 
|:-----------------------------:|:-------------------------------------:| 
| Road narrows on the right     | Road narrows on the right   			| 
| Pedestrians     		    	| Right-of-way at the next intersection |
| Turn left ahead				| Turn left ahead					    |
| Traffic signals	      		| Traffic signals					 	|
| Roundabout mandatory			| Roundabout mandatory					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. These results seem to be less promising than the results on the test set which yielded an accuracy of 0.947 at first sight. However this can be due to the following reasons, the images originally had a quite higher resolution, when resizing them some pixel information might get lost. The Pedestrians sign gets a wrong prediction from the neural network, it is predicted to be a Right-of-way at the next intersection sign, if we compare them the signs look relatively similar at such a low resolution, plus the class imbalance is quite evident in this case. The Pedestrians class has less than 250 samples, while the Right-of-way at the next intersection has almost 1250 samples.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first four images, the model is absolutely sure that it has predicted the correct traffic signs. However for the Pedestrians sign it predicted a Right-of-way at the next intersection sign. The last prediction is a correct but has relatively low confidence with only 0.42. Again, this behavior could be explained by imbalances in the data classes.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Road narrows on the right   					| 
| .99     				| Pedestrians 									|
| .99					| Turn left ahead								|
| .99	      			| Traffic signals				 				|
| .42				    | Roundabout mandatory      					|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
In the following image we can see the activation maps of the first convolutional layer, we see that the model learns certain features like borders and fillings.
![alt text][image5]
In this next image we can appreciate that the activation maps of the second convolutional layer display high activations for other patterns found in the image that are not as intuitive as in the first convolutional layer.
![alt text][image6]