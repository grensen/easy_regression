# Easy Regression In C#

Neural networks are often considered to be difficult to understand. This work aims in the long run to change that. Easy regression is an efficient supervised learning technique for classification problems and it is closely related to multiclass logistic regression, the entry to neural networks.

## Easy Regression

<p align="center">
  <img src="https://github.com/grensen/easy_regression/blob/main/figures/easy_regression.png?raw=true">
</p>

Imagine a system to predict a binary output. In the figure, the possible outputs would be either 0 or 1. But it could also be A and B, or a and b, or even cats and dogs. Each frame holds the incoming signals, in this example the frame works with pixels, but it could also be the weight and hair color of a person or other input values. In this case we will work with 28 * 28 = 784 pixels.

The system is new and doesn't know anything yet, not even noise. Nevertheless, it makes a prediction and takes the highest value at class 0, since no higher value can follow with nothing. The prediction starts at class 0, but the input was a 1 represented as an image. So the prediction should be wrong. This is then followed by the learn and update step.

To do this, the incoming signals from the image for the predicted class that predicted incorrectly are simply calculated negatively on the connected weights, as you can see in blue. So if the signal comes into the system again, it would be weighted more negatively for class 0. The target class 1 adds the signals positive, so that this class evaluates the incoming signal more in a positive way next time.

This is actually the whole magic of how the system "learns" or "perceives". The purple zero has nothing to do with the classes. It represents the balancing of the signals that in turn cause the learning.

## The Model

<p align="center">
  <img src="https://github.com/grensen/easy_regression/blob/main/figures/the_model.png?raw=true">
</p>

Perhaps the entire system can be understood a little bit better in this way, because in our case 10 classes from 0 to 9 are used, which are controlled by an image with 784 pixels. But it could also be several hundred classes.

But all this is not so important. The important thing is to react when something goes wrong in a system. Only then easy regression comes into play. And only 2 classes learn instead of 10, the class that was predicted wrong, with negative signals, and the target class that takes the equal signal positive. 

The weights start at 0, but here they are already well trained. The output is similar to a pseudo softmax function. However, this is only a fake as we will see later.  

## Easy Regression

<p align="center">
  <img src="https://github.com/grensen/easy_regression/blob/main/figures/easy_regression_ji.png?raw=true">
</p>

Here you can see how learning works. In the first sample, a 0 is predicted, but the input was a 5. So the signals are added negatively to the weights for class 0. The signals for the desired class 5 are added positively to the weights of the class.

The second sample is a 0, so the signals for both classes are simply reversed now with the new signal. But this already gives a very weak picture of how the learning will continue. After 20 samples, all classes are already filled, the incoming 9 is recognized as an 8, and is distributed according to the signal.

After 100 samples, the picture is formed further and with a clear increase of accuracy. The 1 as a sample is recognized correctly, that means no training. After the entire dataset was trained with 60,000 samples, the network achieved a training accuracy of 85%. And typical here is the image of the weights which look like this or something similar.

## Infinity Regression

<p align="center">
  <img src="https://github.com/grensen/easy_regression/blob/main/figures/infinity_regression_ji.png?raw=true">
</p>

Infinity regression is a very simple modification with which easy regression works even better. Just take out your imaginary machine gun and randomly shoot out half of the input signals. Then you will get what you see. Infinite, because this creates so many different samples for the network, so that it could train forever. It becomes a lifelong learner. Ok, I admit, the picture doesn't look very spectacular yet.

## Fully Trained

<p align="center">
  <img src="https://github.com/grensen/easy_regression/blob/main/figures/tested_regressions.png?raw=true">
</p>

After a few epochs with the entire training data set, a completely different picture emerges. Spectacular, isn't it? The example above was trained with easy regression. The lower example was trained with infinity regression. The noise in the image is clearly visible in the first example. This is the image that I have already seen in so many examples after training.

The bottom picture of the weights trained with the infinity technique was new to me. It is far less noisy, but the picture has been divided more into areas, where the dark and dirty areas give an idea that the corresponding class tends towards 0 here, which means that the class avoids any prediction in this areas. For a better understanding let's take a look at the demo.

## The Demo

<p align="center">
  <img src="https://github.com/grensen/easy_regression/blob/main/figures/demo.png?raw=true">
</p>

The demo first loads the hyperparameters of the model and then checks if the MNIST dataset is available. If not, the dataset is downloaded from my GitHub account and placed in the appropriate folder. After that, the demo already starts with the easy regression. For this purpose, the entire training data set is trained with 60,0000 examples. In addition, after each epoch, a test is performed to determine how high the accuracy is for the test data that was not learned. 
After that, inifinity regression begins, where only the inputs are randomly switched off before the system gets them.

Finally, the trained model is saved and reloaded from the file to test it again. This is to make sure that everything works right, which seems to be the case.
