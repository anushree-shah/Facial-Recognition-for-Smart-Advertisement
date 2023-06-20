# Facial REcognition for Smart Advertisement
A web application that uses real-time face detection using webcam to extract data like age and gender to provide user with smarter advertisement

Tools used :

•	Python (back-end)
•	HTML (front-end)
•	OpenCV (Face detection API)
•	Caffe Models (Pre-trained models for gender and age recognition)
•	Flask (to deploy as web application)

How to Run the code:
- Have Python, OpenCv and Flask installed on the device
- Have a webcam connected to the device or a camera 
- Have the Caffe models for face detection and age and gender detection and dataset.csv downloaded on the device 
- Additionally, try remaining as still as possible while running the code to get accurate output
- press 'q' to exist the webcam and display the output when satisfied by the estimate

Dataset used:

For the age and gender extraction as the Caffe Models are applied they use the Adience dataset; the dataset is available in the public domain. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. 
	
The advertising dataset(dataset.csv) is manually created just for testing purposes to depict the working of the application. It correlates age and gender to products commonly bought by certain demographics. As this serves as online advertising the age of the user is limited from 8-100 years. (Further explained under challenges)

Caffe Models: 

It is very difficult to accurately guess an exact age from a image because of factors like makeup, lighting, obstructions, and facial expressions . Moreover, a live video implies constant motion this makes the detection and extraction even more volatile and inaccurate.  And so, I made this a classification problem instead of making it one of regression.
This Project uses Deep Learning to accurately identify the gender and age of a person using the Caffe models trained by Tal Hassner and Gil Levi. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- '0-4', '5-8', ' 8-15', '15-20', '20-30', '30-45', '45-60', '60-100'
(8 nodes in the final softmax layer).

Sample output:
![image](https://user-images.githubusercontent.com/77979395/170885674-751cdf0f-cc81-45cf-9db5-c5105dc0902b.png)

