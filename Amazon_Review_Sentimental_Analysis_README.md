# Sentiment-Analysis
Deep Learning, NLP, Keras 


#Sentiment Analysis On Letter-Basis

This notebook illustrates how to predict the sentiment of a text by using a sequence of letters as input. This approach bypasses the need of textual preprocessing and enables language independent predictions.
The dataset consists of Amazon Reviews and their corresponding positive or negative assessment.


# Content
The fastText supervised learning tutorial requires data in the following format:
__label__<X> __label__<Y> ... <Text>

where X and Y are the class names. No quotes, all on one line.

In this case, the classes are __label__1 and __label__2, and there is only one class per row.

__label__1 corresponds to 1- and 2-star reviews, and __label__2 corresponds to 4- and 5-star reviews.

(3-star reviews i.e. reviews with neutral sentiment were not included in the original),

The review titles, followed by ':' and a space, are prepended to the text.

Most of the reviews are in English, but there are a few in other languages, like Spanish.

# Source

The data was lifted from Xiang Zhang's Google Drive dir, but it was in .csv format, not suitable for fastText.

# Training and Testing

Followed the basic instructions at fastText supervised learning tutorial to set up the directory.

