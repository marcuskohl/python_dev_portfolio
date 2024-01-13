import os
import tweepy

#Accessing credentials
API_KEY = os.environ.get('TWITTER_API_KEY')
API_SECRET_KEY = os.environ.get('TWITTER_API_SECRET')
BEARER_TOKEN = os.environ.get('TWITTER_BEARER_TOKEN')
ACCESS_TOKEN = os.environ.get('TWITTER_ACCESS_TOKEN')
ACCESS_TOKEN_SECRET = os.environ.get('TWITTER_ACCESS_SECRET')

#Authenticating to X/Twitter
auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

#Creating API object
api = tweepy.API(auth)

#Verifying authentication
try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")


