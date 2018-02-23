# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



### Writeup / README



### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Visualization is shown in Traffic_Sign_Classifier.html file also uploaded with code . 
I have displayed no if images in each class along with a random sample image .

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I choose to keep the image as it is and only performed normalization by bringing vallues under range of 0,1 .
I achieved this by dividing training data and validation data by 255 and storing values as float.
I went with training of data with available data set and didn't try any augmentation techiques.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model is insperied from LeNet architecture.
Have added few drop outs after max pooling and one during fully connected layer, also have choosen different keep probabilities for differnt layers.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   											| 
| Convolution layer1   	| 5x5x3x6 Filter , 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU activation		|																|
| Max pooling	      	| 2x2 stride, valid padding , outputs 14x14x6 					|
| Drop Out		      	| keep probility 0.9											|
| Convolution layer2   	| 5x5x6x16 Filter , 1x1 stride, valid padding, outputs 10x10x16	|
| RELU activation		|																|
| Max pooling	      	| 2x2 stride, valid padding , outputs 5x5x16 					|
| Drop Out		      	| keep probility 0.7											|
| Flatten			    | Input : 5x5x16  output : 400   								|
| Fully connected layer3| Input : 400  output : 120 									|
| RELU activation		|																|
| Drop Out		      	| keep probility 0.5											|
| Fully connected layer4| Input : 120  output : 84 										|
| RELU activation		|																|
| Fully connected layer5| Input : 84  output : 43 										|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an below values of various parmeters and hyperparamter
BATCH_SIZE: 126
EPOCH : 40
Learning Rate: 0.001
mean : 0 
standard dev : 0.2
Optimizer : AdamOptimizer

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I went with the LeNet model which was used in the class room for pridicting numbers from image dataset provided by MINDST.
Initially the validation accuracy went till 0.89 so i introduced drop outs after first , second and third layer with keep probailities as 0.9 , 0.7 ,0.5 respectively , this was only done for training data set , where as for validation and test dataset all were kept as 1.0

My final model results were:
* training set accuracy of 0.93
* validation set accuracy of 0.93


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I choose 5 images from wiki , images are diplayed under file "Traffic_Sign_Classifier.html" submitted with the project.
Images choosen were 

    Turn Left Ahead
    Turn Right Ahead
    No Entry
    Ahead
    Stop

All images were of shape 120x120x4 , these imgaes were later converted to 32x32x3 so that they can be added to the model being trained.
To perform resizing took help of matplot,opencv libraries.

Since these images were clear so model predicited them correclty.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn Left Ahead       | Turn Left Ahead   							| 
| Turn Right Ahead     	| Turn Right Ahead 								|
| No Entry				| No Entry										|
| Ahead	      			| Ahead					 						|
| Stop					| Stop      									|


The model was able to correctly guess all 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a Turn left ahead sign (probability of 0.9), and the image does contain a Turn left ahead sign. The top five soft max probabilities were

| Probability      	|     Prediction	        					| 
|:------------------|:---------------------------------------------:| 
| 9.99996185e-01	| Turn left ahead   							| 
| 3.68013252e-06	| Go straight or left							|
| 9.62443139e-08	| Keep right									|
| 5.80101798e-08	| Ahead only					 				|
| 6.61787958e-10	| Right-of-way at the next intersection			|


For the second image , the model is relatively sure that this is a ahead sign (probability of 0.9), and the image does contain a ahead sign. The top five soft max probabilities were 
| Probability      	|     Prediction	        					| 
|:------------------|:---------------------------------------------:| 
| 9.99886394e-01	| Ahead only   									| 
| 9.33241608e-05	| Go straight or right							|
| 9.39425718e-06	| End of all speed and passing limits			|
| 6.30489558e-06	| Turn left ahead				 				|
| 3.85427438e-06	| Roundabout mandatory		 	 	 	 		|

For the third image , the model is relatively sure that this is a no entry sign (probability of 1.0), and the image does contain a no entry sign. The top five soft max probabilities were 
| Probability      	|     Prediction	        					| 
|:------------------|:---------------------------------------------:| 
| 1.00000000e+00	| No entry   									| 
| 8.96214364e-13	| Stop											|
| 7.64902706e-26	| Speed limit (20km/h)							|
| 2.76909861e-27	| Speed limit (120km/h)					 		|
| 1.37776512e-27	| Speed limit (30km/h)							|

For the fourth image , the model is relatively sure that this is a Turn right ahead sign (probability of 0.9), and the image does contain a Turn right ahead sign. The top five soft max probabilities were 
| Probability      	|     Prediction	        					| 
|:------------------|:---------------------------------------------:| 
| 9.99999762e-01	| Turn right ahead   							| 
| 1.04195585e-07	| Roundabout mandatory							|
| 4.57173677e-08	| Ahead only									|
| 3.09445980e-08	| Keep left					 					|
| 2.48291721e-09	| No vehicles									|

For the fifth image , the model is relatively sure that this is a Turn left stop sign (probability of 0.9), and the image does contain a stop sign. The top five soft max probabilities were 
| Probability      	|     Prediction	        					| 
|:------------------|:---------------------------------------------:| 
| 9.97931242e-01	| Stop   										| 
| 2.04815390e-03	| No entry										|
| 1.69340274e-05	| Speed limit (30km/h)							|
| 2.74702711e-06	| Speed limit (20km/h)					 		|
| 5.80706285e-07	| Speed limit (120km/h)							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


