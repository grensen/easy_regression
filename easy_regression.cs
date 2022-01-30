using System; using System.Linq; using System.IO;
System.Action<string> print = System.Console.WriteLine;
// https://github.com/grensen/easy_regression

// hyperparameter, same results : lr1 = 1, lr2 = 0.5f; lr1 = 2, lr2 = 1; lr1 = 1.5f, lr2 = 0.75f;
float lr1  = 2.0f; // learning rate plus
float lr2  = 1.0f; // learning rate minus
float drop = 0.9f;
int epochs = 15;

// loads MNIST to this folder
AutoData d = new(@"C:\mnist\");

print("Run easy regression\n");
RunDemo(d, lr1, lr2, epochs);

print("\nRun infinity regression\n");
RunDemo(d, lr1, lr2, epochs, drop);

// todo : save and reload net
print("\nEnd demo");

static void RunDemo(AutoData d, float lr1, float lr2, int epochs, float drop = 0)
{
    float lr = 1;
    float[] weights = new float[784 * 10];
    for (int ep = 0, correct; ep < epochs; ep++, lr *= lr2)
    {
        // efficient learning rate
        for (int i = 0; i < 7840; i++) weights[i] *= lr1;

        // get training accuracy
        int cTrain = Test(d, 60000, true, weights, lr, drop, new Random(123 + ep));
       
        // get test accuracy
        int cTest = Test(d, 10000, false, weights, lr);

        System.Console.WriteLine(ep + " Training = " + (cTrain * 100.0 / 60000).ToString("F2") 
            + "%, Test = " + (cTest * 100.0 / 10000).ToString("F2") + "%");
    }
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
            if (s.sample[i] != 0 && r.NextDouble() > drop) s.sample[i] = 0;

        // feed forward
        float[] outputs = new float[10];
        for (int i = 0; i < 784; i++) // input over outputs
            if (s.sample[i] > 0) // skip zero multiplications
                for (int j = 0; j < 10; j++) // layerwise
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
    public bool andTrain;
}
struct AutoData
{
    public byte[] samplesTest, labelsTest;
    public byte[] samplesTraining, labelsTraining;
    public AutoData(string yourPath)
    {
        string trainDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-images.idx3-ubyte";
        string trainLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-labels.idx1-ubyte";
        string testDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-images.idx3-ubyte";
        string testnLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-labels.idx1-ubyte";

        string d1 = @"trainData", d2 = @"trainLabel", d3 = @"testData", d4 = @"testLabel";

        if (!File.Exists(yourPath + d1) 
            || !File.Exists(yourPath + d2)
              || !File.Exists(yourPath + d3)
                || !File.Exists(yourPath + d4))
        {
            System.Console.WriteLine("Data does not exist");
            if (!Directory.Exists(yourPath)) Directory.CreateDirectory(yourPath);

            System.Console.WriteLine("Download MNIST dataset from GitHub with cleanup");
            this.samplesTraining = (new System.Net.WebClient().DownloadData(trainDataUrl)).Skip(16).Take(60000 * 784).ToArray();
            this.labelsTraining = (new System.Net.WebClient().DownloadData(trainLabelUrl)).Skip(8).Take(60000).ToArray();
            this.samplesTest = (new System.Net.WebClient().DownloadData(testDataUrl)).Skip(16).Take(10000 * 784).ToArray();
            this.labelsTest = (new System.Net.WebClient().DownloadData(testnLabelUrl)).Skip(8).Take(10000).ToArray();
           
            System.Console.WriteLine("Save MNIST data into folder " + yourPath + "\n");
            File.WriteAllBytes(yourPath + d1, this.samplesTraining);
            File.WriteAllBytes(yourPath + d2, this.labelsTraining);
            File.WriteAllBytes(yourPath + d3, this.samplesTest);
            File.WriteAllBytes(yourPath + d4, this.labelsTest); return;
        }
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
        s.andTrain = isTrain;
        return s;
    }
}
