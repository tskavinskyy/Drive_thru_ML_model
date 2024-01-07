#Identifying Drive-Thru Buildings with Satellite Imagery Recognition Model

Introduction
Commercial real estate (CRE) professionals have to analyze thousands of potential retail sites to know which one has the highest financial potential. It is crucial for them to quickly understand every possible building characteristic that may affect the site's potential. There are hundreds of building features they may be interested in, this article will demonstrate how you can use satellite imagery paired with an image recognition model to find if a building has a Drive-Thru.

Using the power of AI image recognition real estate professionals can save hundreds of hours compiling data about their and potential properties. They can use the data this algorithm produces to speed up crucial decisions, get insights into market conditions, and understand the true financial potential of a site.

Other image recognition algorithms we found can only identify that our images are satellite images. They are not specific enough to recognize building features in the image. We build this specialized algorithm to recognize what is in the satellite image and in this particular example if we can find a drive-thru at this location.
About the Algorithm
The objective of this study is to create an algorithm to substitute human data collection of building features with artificial intelligence satellite image recognition. This convolutional neural network (CNNs) drive-thru predicted with an accuracy of 72% whether a building had a drive-thru or not from only knowing the address of the building.

Data Preparation and Description of Variables
Building data were collected from web scrapers. Satellite images were pulled using Google API. We tested the performance of three different classification model architectures before creating the final model to ensure we had the optimal model for the task. Y

To avoid selection bias, a data sample from a larger dataset for each model was selected randomly. The data were up-sampled to create an even distribution of each category so that the algorithm had enough examples of each class to learn from.

We converted addresses to geocodes to get latitude and longitude coordinates for each address. Those coordinates were used as inputs into Google API to download map and satellite images of size 200x200 for each building in the model samples. We tested the models on images of different sizes. Size 200x200 had enough details for the models to train on and increasing the image size did not improve the performance of the models. The images were then converted to black and white to decrease storage size and training processing requirements which allowed us to use more images to train the model. You can read more about how the data was prepared here.

We also utilized a data augmentation algorithm, which rotates each image three times to increase the number of examples that the models get trained on.

Model Architectures
Classification models were built using a convolutional neural network (CNN) technique. We tested 3 different neural network architectures to see if deeper models with more parameters would improve the prediction. Based on the comparison of the architecture of the models, we created the final model.

The first shallow CNN model we tested was a modification of the handwritten digits classification model based on Modified National Institute of Standards and Technology (MNIST) dataset described by Yalçın (2018). Table A1 describes the architecture of our shallow model. We added several convolutional layers because the patterns of our images are more complicated than the digits that the NMIST model was predicting. The model picked up some of the image patterns but the accuracy was not high and thus, not very useful.

To improve the accuracy of predictions of the first shallow model, we tested a deeper CNN model architecture from Brownlee (2019). Table B describes the architecture of this model.

The Final drive-thru Model
The drive-thru model allows the user to classify whether a building has a drive-thru with a prediction accuracy of 72% on the holdout set. First, you will need to import some dependencies and the data that was prepared in the previous blog article. https://medium.com/@skavinskyy/geocoding-addresses-to-get-latitude-and-longitude-with-google-api-e85f159069a3

The drive-thru model was trained on 3,637 images. Data were up-sampledThe drive-thru model was trained on 3,637 images. Data were up-sampled so that both classes of the dependent variable had a relatively equal count. The train and test ratio for this model was 70:30. Data were augmented with three rotations of each image to increase the training size and improve the performance of the model.

The model was built using both shallow and deep CNN architectures. The performance of the model using these two architectures wasn’t satisfactory due to very low prediction accuracy and overfitting. The final model that improved accuracy and reduced overfitting was built using an architecture that comprises four convolutions, two drop out, two max pool, and two dense layers. The model also utilized early stopping to tackle the problem of overfitting. Table F describes the architecture of the final model.
https://miro.medium.com/v2/resize:fit:786/format:webp/1*2Xj7mJHD51W-iYRnudOb9A.png

The classification confusion matrix in Table G describes in more detail the performance of the model for each class. Chart 2 below shows the increase in training and validation accuracy of the model with each epoch.

Table F: Drive-Thru Final Model Architecture

https://miro.medium.com/v2/resize:fit:828/format:webp/1*IbhXIBqZsewXk7z212N72w.jpeg
https://miro.medium.com/v2/resize:fit:828/format:webp/1*IbhXIBqZsewXk7z212N72w.jpeg




Conclusions
We can identify
We can identify building features from satellite images with reasonable accuracy. These deep learning techniques can replace human site surveys to classify, measure and identify building features for commercial real estate decision-making and analytics. Limitations of geocoding and outdated images bring a fair amount of noise to the data that is outside of the scope of the models. Showing the models more examples will help improve the accuracy and help them generalize.

More advanced specialized image recognition algorithms often reach over 95% accuracy so our prediction accuracy of 72% falls short of that mark. Yet, these models can provide actionable insights to our industry partners. Considering the variety of buildings across the US and some of the limitations of geocodes and image vintages mentioned before, the models outperformed our expectations. The results of the models we built are almost as good as some of the more generic algorithms out there, in identifying these specific building features.

These models can substitute in-person site visits to identify building features at the preliminary stages of real estate selection thus saving companies time and resources. Real estate decision-makers can use the results of this algorithm to quickly filter and find desired buildings. Identifying these building features can be used as inputs to improve.

References.
Brownlee, J. (2019, May 20). How to Develop a Deep CNN for Multi-Label Classification of Photos [Blog post].

Google. Maps Static API.

Yalçın, O. (2018, August 19). Image Classification in 10 Minutes with MNIST Dataset [Blog post].






