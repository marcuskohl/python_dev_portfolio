import tweepy
import csv
from twitter_auth import BEARER_TOKEN

#Defining CSV path
csv_file_path = '../data/tweets.csv'

#Defining callback for the stream
class MyStreamListener(tweepy.StreamingClient):
    def on_tweet(self, tweet):
        #Extracting text, user ID, and creation time
        tweet_text = tweet.text
        user_id = tweet.author_id
        tweet_time = tweet.created_at

        #Opening CSV file and appending data
        with open(csv_file_path, 'a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([tweet_text, user_id, tweet_time])

#Initializing stream
myStream = MyStreamListener(BEARER_TOKEN)

#Defining rules
myStream.add_rules(tweepy.StreamRule("$BTC"), dry_run=False)

#Starting stream
myStream.filter(expansions=["author_id"], tweet_fields=["created_at", "text"])



