<h1 align="center">Movie Sentiment Analyzer</h1>
A very simple movie sentiment analyzer based on Naive Bayes classification. It uses Natural Language Processing to process movie reviews and then send the processed data to a Multinomial Naive Bayes to generate the sentiment. Sentiment is either 'positive' or 'negative', based on the review you write.

## Purpose of this project:
<p align="justify">Say, you want to write a movie review. But thing is you are not sure how you feel about the movie (Maybe the movie is, well, meh...) but you want to write a positive review. Problem is, if you write a review of that particular movie, chances are the review will be partly negative, or entirely negative. To prevent that, you can write your review to this AI model and check whether the review will be positive or negative, based on that you can edit your review and give a positive review regardless of how you feel about the movie (Obviously if you don't feel bad that is). It may also work vice versa, if you want to write a bad review for a movie, but you are too polite to write negative stuff and the review turns out slightly positive, or entirely positive (If you are dumb polite 🫡).</p>

**Note:** The Bayes model is trained with 50k IMDb movie reviews.

## How to use:
Like the feel of it? Okie dokie, let's get to it then.
### Prerequisites: 
- You got to have `Python` installed with `venv` module.
- Active stable internet connection, cause a large file will be downloaded (The whole 50k movie reviews in IMDb).

Okay, you are ready to rock!

### Get the ball rolling:
 - Create a Python virtual environment using python `venv` module. Learn more about it, [here](https://docs.python.org/3/library/venv.html).
 - After entering the virtual environment, install all the required Python packages using `pip install -r requirements.txt`.
 - Then you have to generate the Dataset. After entering the virtual environment, use `python gends.py`. It will take some time, as it will download all the reviews and generate the datasets.
 - Finally, after the dataset generation is complete, use `python repl.py` to get to the REPL (Read-Evaluate-Print-Loop), where you type your review to get the sentiment.
 
 ```bash
$ python repl.py
Training Naive Bayes...
Initializing vectorizer...
Preparing binarizer...
Entering REPL...

>>> When I started watching the movie, I thought, well you know what it won't be that much fun. But, god wasn't I wrong! After 25min of the movie, I got completely hooked. The character development in the middle and story progression is simple mesmerizing. The actors also played an awesome role playing those characters, they seemed real. Completely worth watching!
-> Your review is positive! Good job.
 ```

**Note:** Type `.exit` to exit the REPL.

That's all folks 🫡🫡.
<b><i>Peace.</i></b>
