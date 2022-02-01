using System; using System.Linq; using System.IO;
System.Action<string> print = System.Console.WriteLine;
// https://github.com/grensen/easy_regression

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
        System.Console.WriteLine("Load MNIST data and labels from " + yourPath + "\n");
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

    public void SaveWeights(string savePath, float[] weights)
    {
        Console.WriteLine("\nSave weights to " + source + @"myInfinityTest.txt");
        // bring weights into string
        string[] wStr = new string[weights.Length];
        for (int i = 0; i < weights.Length; i++)
            wStr[i] = ((decimal)((double)weights[i])).ToString(); // for precision
        // save weights to file
        File.WriteAllLines(source + savePath, wStr);
    }

    public float[] LoadWeights(string loadPath)
    {
        Console.WriteLine("\nLoad weights from " + source + @"myInfinityTest.txt");
        // load weights from file
        string[] wStr = File.ReadAllLines(source + loadPath);
        // string to float
        float[] weights = new float[wStr.Length];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = float.Parse(wStr[i]);
        return weights;
    }
}
// end 