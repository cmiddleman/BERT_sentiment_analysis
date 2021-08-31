#Sentiment analysis with BERT

In this project I use Google's BERT architecture to tackle two NLP related problems: classifying Amazon product reviews as positive or negative, and analyzing chat of the live-streaming platform Twitch. 

BERT is a powerful NLP model developed by Google. Without any additional training it was able to achieve 90% accuracy on a test set of 400K amazon reviews labeled positive or negative. After training on the other 3.6M amazon reviews an accuracy of 97% was achieved (70% decrease in error). 

A downside of BERT is that it is a large model (>100M trainable paramaters for the particular BERT model used in this project). Training time is over a week per epoch on a CPU, by using Google's state of the art TPUs training time was reduced to 2.5 hours per epoch.
