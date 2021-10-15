## Objective
To Design a neural network that can take inputs as below and can predict the outputs as below:

**Inputs:** an MNIST image and a Random number.

**Outputs:** MNIST Digit and sum of MNIST digit + random number.

![Problem_Statement](https://github.com/pratiksgole/TSAI-Assignments/tree/main/Session_3/images/problem_statement.png)


## Data Representation
We represent the dataset as shown below:

| Image | Random Number | Label | Sum Label |

|----|----|----|----|

|[28x28]| 3 | 5 | 8 |

## Data Generation
We generated our data using the ``Dataset`` class from ``torch.utils.data``.

```python
class MNISTRandDataset(Dataset):
    def __init__(self, mnist):
        self.mnist = mnist

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]

        rand_num = random.randint(0, 9)

        rand_num_tensor = F.one_hot(torch.tensor(rand_num), num_classes=10)
        sum_label = label + rand_num

        return image, rand_num_tensor, label, sum_label
```

## Image and Random Input Combination
We have concatenated the final probabilities of 10 classes for a MNIST image from fully connected layer (``self.out1``) with the Random number represented in 10-bit one hot encoded vector.

After this we get a tensor of size 20, which we pass through final fully connected layer (``self.out2``) with ``input_features = 20`` and ``output_features = 19``, to get all possible probabilities of sums which are 19.

![Combination](https://github.com/pratiksgole/TSAI-Assignments/tree/main/Session_3/images/combine_image_num.png)


## Model 
![Model_Architecture](https://github.com/pratiksgole/TSAI-Assignments/tree/main/Session_3/images/model.png)

## Loss Function
We used 2 losses, one for each ouput, and averaged them to get total loss.
Since this was classification problem we have used cross entropy. Cross entropy measures the  distance between output probabilities and truth values.

## Evaluation
MNIST test dataset has 10,000 images.

We measured and compared the Total loss and Accuracy on Train set. 

### Results
Below are the results of our model predictions at each epoch. We have trained the model for ``100 epochs``. The training was done on **GPU**.
Accuracy on Test set for digit recognition is 98.5% and sum prediction accuracy is 92.2%.

![Training_Logs](https://github.com/pratiksgole/TSAI-Assignments/tree/main/Session_3/images/model.png)

#### Accuracy vs No. of Epochs
![Accuracy](https://github.com/pratiksgole/TSAI-Assignments/tree/main/Session_3/images/accuracy.png)
