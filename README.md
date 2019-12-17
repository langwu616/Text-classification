# Text-classification
Task: classify the tweet as zero or more of eleven emotions that best represent the mental state of the tweeter

The primary evaluation metric is multi-label accuracy (or Jaccard index). Since this is a multi-label classification task, each tweet can have one or more gold emotion labels, and one or more predicted emotion labels. Multi-label accuracy is defined as the size of the intersection of the predicted and gold label sets divided by the size of their union. This measure is calculated for each tweet, and then is averaged over all tweets in the dataset.
