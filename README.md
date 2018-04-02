# Deep-CNN-for-image-recognition
Neural network that learns to classify images. I tested this algorithm on the mnist dataset, The MNIST Dataset contains 70,000 images of handwritten digits (zero through nine), divided into a 60,000-image training set and a 10,000-image testing set. This algorithm was able to correctly clasify the numbers with 98% efficiency. The convolutions_pipeline.png image shows the pipeline of the image being convoluted to a smaller image. The hidden_layer.png pic shows the transformations being ran, and the calculation of the final error. The images help with data visualization and should be looked at brifely to increase the understanding of what the algorithm is doing. 

I get an average of 97 % correct images on my test set using the MNIST taining set.
restore function gives probability of neural network guessing the correct image.

Example output after running restore function on the MNIST training set:                                                        
restore = restore(x_,y_)                                                                                                          
INFO:tensorflow:Restoring parameters from /Users/bennicholl/Desktop/outputfile                                                      
97.0 % chance of correct prediciton                                                                                             
cross entropy errors are: 0.16949451                                                                                            



Ensure that the newest version of tensorflow is downloaded so tensorboard runs correctly
run:  conda install -c conda-forge tensorflow
