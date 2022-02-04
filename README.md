# Easy Regression 

<p align="center">
  <img src="https://github.com/grensen/easy_regression/blob/main/figures/ez.png?raw=true">
</p>

Neural networks are often considered to be difficult to understand. This work aims in the long run to change that. Easy regression is an efficient iterative supervised learning technique for classification problems and it is closely related to multiclass logistic regression, the entry to neural networks.

## The Intuition

<p align="center">
  <img src="https://github.com/grensen/easy_regression/blob/main/figures/easy_regression.png?raw=true">
</p>

Imagine a system to predict a binary output. In the figure, the possible output classes would be either 0 or 1. But it could also be A and B, or a and b, or even cats and dogs. Each frame holds the incoming signals, in this example the frame works with pixels, but it could also be the weight and hair color of a person or other input values. In this case we will work with 28 * 28 = 784 pixels.

The system is new and doesn't know anything yet, not even noise. Nevertheless, it makes a prediction and takes the highest value at class 0, since no higher value can follow with nothing. The prediction starts at class 0, but the input was a 1 represented as an image. So the prediction should be wrong. This is then followed by the learn and update step.

To do this, the incoming signals from the image for the predicted class that predicted incorrectly are simply calculated negatively on the connected weights, as you can see in blue. So if the signal comes into the system again, it would be weighted more negatively for class 0. The target class 1 adds the signals positive, so that this class evaluates the incoming signal more in a positive way next time.

This is actually the whole magic of how the system "learns" or "perceives". The purple zero has nothing to do with the classes. It represents the balancing of the signals.

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

Here you can see how the training works. In the first sample, a 0 is predicted, but the input was a 5. So the signals are added negatively to the weights for class 0. And the signals for the desired class 5 are added positively to the weights of the class.

The second sample is a 0, so the signals for both classes are simply reversed now with the new input signal. But this already gives a very weak picture of how the learning will continue. After 20 samples, all classes are already filled, the incoming 9 is recognized as an 8, and is distributed according to the signal.

After 100 samples, the picture is formed further and with a clear increase of accuracy. The 1 as a sample is recognized correctly, that means no training. After the entire dataset was trained with 60,000 samples, the network achieved a training accuracy of 85%. And typical here is the image of the weights which look like this or something similar, eben when you will use logistic regression.

## Infinity Regression

<p align="center">
  <img src="https://github.com/grensen/easy_regression/blob/main/figures/infinity_regression_ji.png?raw=true">
</p>

Infinity regression is a very simple modification to make easy regression even better. Just take out your imaginary machine gun and randomly shoot out half of the input signals. Then you will get what you see. Infinite, because this creates so many different samples for the network, so that it could train forever. It becomes a lifelong learner. But ok, I admit, the picture doesn't look very spectacular yet.

## Fully Trained

<p align="center">
  <img src="https://github.com/grensen/easy_regression/blob/main/figures/tested_regressions.png?raw=true">
</p>

After a few epochs with the entire training data set, a completely different picture emerges. Spectacular, isn't it? The example above was trained with easy regression. The lower example was trained with infinity regression. The noise in the image is clearly visible in the first example. This is the image that I have already seen in so many examples after training.

The bottom picture of the weights trained with the infinity technique was new to me. It is far less noisy. The picture has been divided more into areas, where the dark and dirty areas give an idea that the corresponding class tends towards 0 here. Which means that the class try to avoid a prediction in this areas. 

## The Demo

<p align="center">
  <img src="https://github.com/grensen/easy_regression/blob/main/figures/demo.png?raw=true">
</p>

The demo first loads the hyperparameters of the model and then checks if the MNIST dataset is available. If not, the dataset is downloaded from my GitHub account and placed in the appropriate folder from where the data can be reloaeded. After that, the demo already starts with easy regression. For this purpose, the entire training data set is trained with 60,0000 samples. In addition, after each epoch, a test is performed with 10,000 to determine the accuracy for the test data that was not learned. 

After that, inifinity regression begins, where only the inputs are randomly switched off before the system gets them.

Finally, the trained model is saved and reloaded from the file to test it again. This is to make sure that everything works right, which seems to be the case.

## High Level Code

~~~cs
using System; using System.Linq; using System.IO;
System.Action<string> print = System.Console.WriteLine;

print("Begin easy regression demo on MNIST\n");

var lr1    = 2.0f; // learning rate > 1
var lr2    = 1.0f; // learning rate < 1
var epochs = 10;
var drop   = 0.2f; // input dropout

print("Learning rate1 = " + lr1.ToString("F4") + "\nLearning rate2 = " + lr2.ToString("F4"));
print("Epochs = " + epochs.ToString() + "\nDrop = " + (drop * 100).ToString("F2"));

// loads MNIST to this folder
AutoData d = new(@"C:\mnist\");

print("Run easy regression");
RunDemo(d, lr1, lr2, epochs);

print("\nRun infinity regression");
float[] infWeights = RunDemo(d, lr1, lr2, epochs, drop);

d.SaveWeights(@"myInfinityTest.txt", infWeights);
float[] loadedWeights = d.LoadWeights(@"myInfinityTest.txt");

print("\nRerun infinity regression test");
int correctTest = Test(d, 10000, false, loadedWeights, 0);
print("Test accuracy = " + (correctTest * 100.0 / 10000).ToString("F2") + "%");

print("\nEnd demo");
~~~

The code in the highlevel is fairly intuitive. The most important thing is the specified path from `AutoData d`, where the dataset and networks are stored. I wanted to make this area readable and simple. But under other circumstances, other code designs would be preferable.

## Functions

~~~cs
static float[] RunDemo(AutoData d, float lr1, float lr2, int epochs, float drop = 0)
{
    float lr = 1; // multiplier
    float[] weights = new float[784 * 10];

    for (int ep = 0; ep < epochs; ep++, lr *= lr2)
    {
        // more efficient learning rate - reduced impact each epoch
        for (int i = 0; i < 7840; i++) weights[i] += weights[i] * lr1;

        // get training accuracy
        int cTrain = Test(d, 60000, true, weights, lr, drop, new Random(123 + ep));
       
        // get test accuracy
        int cTest = Test(d, 10000, false, weights, lr);

        System.Console.WriteLine((ep + 1) + " Training = " + (cTrain * 100.0 / 60000).ToString("F2") 
            + "%, Test = " + (cTest * 100.0 / 10000).ToString("F2") + "%");
    }
    return weights;
}

static int Test(AutoData d, int len, bool training, float[] weights, float lr, float drop = 0, Random r = null)
{
    int correct = 0;
    for (int x = 0; x < len; x++)
    {
        // feed sample id from test or training
        Sample s = d.GetSample(x, training);

        // input dropout
        if (drop != 0) for (int i = 0; i < 784; i++)
            if (s.sample[i] != 0 && r.NextDouble() > 1 - drop) s.sample[i] = 0;

        // feed forward
        float[] outputs = new float[10];
        for (int i = 0; i < 784; i++) // each input neuron
            if (s.sample[i] > 0) // skip zero multiplications
                for (int j = 0; j < 10; j++)
                    outputs[j] += s.sample[i] * weights[i * 10 + j];

        int prediction = ArgMax(outputs);

        // (backprop) plus update
        if (training && prediction != s.label)
            for (int i = 0; i < 784; i++)
                if (s.sample[i] != 0) // only non zeros
                {
                    weights[i * 10 + s.label] += s.sample[i] * lr;
                    weights[i * 10 + prediction] -= s.sample[i] * lr;
                }

        // todo : add delta for batch update

        correct += prediction == s.label ? 1 : 0;
    }
    return correct;
    static int ArgMax(float[] arr)
    {
        int prediction = 0;
        float max = arr[0];
        for (int i = 1; i < 10; i++)
            if (arr[i] > max)
            { max = arr[i]; prediction = i; }
        return prediction;
    }
}
~~~

## AutoData

~~~cs
struct Sample
{
    public float[] sample;
    public int label;
}

struct AutoData
{
    public string source;
    public byte[] samplesTest, labelsTest;
    public byte[] samplesTraining, labelsTraining;
    public AutoData(string yourPath)
    {
        this.source = yourPath;

        // hardcoded urls from my github
        string trainDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-images.idx3-ubyte";
        string trainLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-labels.idx1-ubyte";
        string testDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-images.idx3-ubyte";
        string testnLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-labels.idx1-ubyte";

        // change easy names 
        string d1 = @"trainData", d2 = @"trainLabel", d3 = @"testData", d4 = @"testLabel";

        if (!File.Exists(yourPath + d1) 
            || !File.Exists(yourPath + d2)
              || !File.Exists(yourPath + d3)
                || !File.Exists(yourPath + d4))
        {
            System.Console.WriteLine("\nData does not exist");
            if (!Directory.Exists(yourPath)) Directory.CreateDirectory(yourPath);

            // padding bits: data = 16, labels = 8
            System.Console.WriteLine("Download MNIST dataset from GitHub");
            this.samplesTraining = (new System.Net.WebClient().DownloadData(trainDataUrl)).Skip(16).Take(60000 * 784).ToArray();
            this.labelsTraining = (new System.Net.WebClient().DownloadData(trainLabelUrl)).Skip(8).Take(60000).ToArray();
            this.samplesTest = (new System.Net.WebClient().DownloadData(testDataUrl)).Skip(16).Take(10000 * 784).ToArray();
            this.labelsTest = (new System.Net.WebClient().DownloadData(testnLabelUrl)).Skip(8).Take(10000).ToArray();
           
            System.Console.WriteLine("Save cleaned MNIST data into folder " + yourPath + "\n");
            File.WriteAllBytes(yourPath + d1, this.samplesTraining);
            File.WriteAllBytes(yourPath + d2, this.labelsTraining);
            File.WriteAllBytes(yourPath + d3, this.samplesTest);
            File.WriteAllBytes(yourPath + d4, this.labelsTest); return;
        }
        // data on the system, just load from yourPath
        System.Console.WriteLine("\nLoad MNIST data and labels from " + yourPath + "\n");
        this.samplesTraining = File.ReadAllBytes(yourPath + d1).Take(60000 * 784).ToArray();
        this.labelsTraining = File.ReadAllBytes(yourPath + d2).Take(60000).ToArray();
        this.samplesTest = File.ReadAllBytes(yourPath + d3).Take(10000 * 784).ToArray();
        this.labelsTest = File.ReadAllBytes(yourPath + d4).Take(10000).ToArray();       
    }
    public Sample GetSample(int id, bool isTrain)
    {
        Sample s = new();
        s.sample = new float[784];
       
        if (isTrain) for (int i = 0; i < 784; i++)
                s.sample[i] = samplesTraining[id * 784 + i] / 255f;
        else for (int i = 0; i < 784; i++)
                s.sample[i] = samplesTest[id * 784 + i] / 255f;

        s.label = isTrain ? labelsTraining[id] : labelsTest[id];
        return s;
    }
    public void SaveWeights(string name, float[] weights)
    {
        Console.WriteLine("\nSave weights to " + source + name);
        // bring weights into string
        string[] wStr = new string[weights.Length];
        for (int i = 0; i < weights.Length; i++)
            wStr[i] = ((decimal)((double)weights[i])).ToString(); // for precision
        // save weights to file
        File.WriteAllLines(source + name, wStr);
    }
    public float[] LoadWeights(string name)
    {
        Console.WriteLine("\nLoad weights from " + source + name);
        // load weights from file
        string[] wStr = File.ReadAllLines(source + name);
        // string to float
        float[] weights = new float[wStr.Length];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = float.Parse(wStr[i]);
        return weights;
    }
}
~~~

The AutoData struct takes care of the data and handles the whole input/output business. From data to network storage and deployment. Pretty cool and definitely a highlight which makes it very easy for the user to use the demo. Just copy the code and start without much fiddling around. This could be a standard feature for such demos if it goes according to me.

Often enough I could not try demos due to my lack of knowledge.

## Some Experiments

If you change the line for learning rate1: 
~~~cs
for (int i = 0; i < 7840; i++) weights[i] += weights[i] * lr1;
~~~
To that line:
~~~cs
for (int i = 0; i < 7840; i++) weights[i] *= lr1;
~~~
This happens: 
<p align="center">
  <img src="https://github.com/grensen/easy_regression/blob/main/figures/lr_set.png?raw=true">
</p>

This not only shows very nicely that you can scale the weights both larger and smaller, but also how the effort with lr1 is significantly less than with lr2. And also that the learning rate here works completely differently from what we know from neural networks. 

At the beginning I talked about a fake softmax, because depending on the scaling of the weights, the softmax pseudoprobability is influenced by the learning rate. And thus stronger or weaker. However, the result itself remains the same with any scaling. No matter how big or small you have trained the weights, as long as the smallest and biggest weights are not swallowed by the scaling.
