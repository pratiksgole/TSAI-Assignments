# Text Classification on datasets available in torchtext using embeddingbag

Torchtext is amazing library that handles preprocessing of text were well. It has several modules which helps in speeding NLP pipeline.

## torchtext.datasets

Has sample datasets for text classification, language modeling, machine translation and question-answering. It returns iterable object.

```How to use
from torchtext.datasets import YelpReviewFull
train_iter, test_iter = YelpReviewFull()
```

## Datasets
In this example, we will be building our model based on YelpReviewPolarity and AmazonReviewPolarity datasets. More the value of label, positive is the review.

## Pipeline
1. Load Data
2. Build Vocab, set index for unknown and special characters.
3. Build text classification model. Model used embeddingbag layer followed by fully connected layer. Embeddingbags have lesser accuracy than embedding layer  followed by LSTM as it looses sequential information while forming embeddingbag. Embedding bag is good way to build model for shorter sentences as it don't need padding as Embedding followed by LSTM itself.
4. Define optimizer
5. Train model
6. Evaluate

For each class in both dataset we get ~90% accuracy, which is good for such simple model.

## YelpReviewPolarity  
### Training logs
![YelpReviewPolarity Accuracy](https://github.com/pratiksgole/TSAI-Assignments/blob/main/Session_5/images/yelp_accuracy.PNG)

### Classwise test accuraty
![Yelp Classwise Accuracy](https://github.com/pratiksgole/TSAI-Assignments/blob/main/Session_5/images/yelp_cls_accuracy.PNG)

## AmazonReviewPolarity 
### Training logs
![AmazonReviewPolarity Accuracy](https://github.com/pratiksgole/TSAI-Assignments/blob/main/Session_5/images/amz_accuracy.PNG)

### Classwise test accuraty
![Amazon Classwise Accuracy](https://github.com/pratiksgole/TSAI-Assignments/blob/main/Session_5/images/amz_cls_accuracy.PNG)