#!/usr/bin/env python
from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

# Go to http://apps.twitter.com and create an app.
# The consumer key and secret will be generated for you after
consumer_key="MWwQL4zOGqOOHJ2mOCT51YLpu"
consumer_secret="uAWUy0qdHUgQ7Q0MMH8U2jNNdILFHuV4nr16Chpw3U87wTrt3V"

# After the step above, you will be redirected to your app's page.
# Create an access token under the the "Your access token" section
access_token="938738992111120384-33fOg5XPHS8rBMVSs06mRMfM0PMT2d9"
access_token_secret="VS16OHoqLtoQl5AFsn89FMpdRFOHy5vUepeeKygcLrwIo"

class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)

if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)
    stream.filter(track=["macri"], languages=["es"])
