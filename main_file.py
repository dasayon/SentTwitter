from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s


ckey="kjEkgllv84liDohTNBz9FmUQ9"
csecret="SzJAIAt0MTJ6ZDQOsytzW3P0DHeC1lVyqEacWlXpThdGltsJKY"
atoken="3124394218-Q7X77MmoM65Z0ELaj9uDR5lffrQSVAzlzEXtv2q"
asecret="AsaEMS6fkVkKR9qHgYe0Y1c8pGc6fjUhP4lxj5E8ZSVFi"

import re
import sys

def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end
class listener(StreamListener):
    count=0
    stopat=10
    def on_data(self,data):
            try:
                all_data=json.loads(data)
                tweet=all_data["text"]
                tweet=processTweet(tweet)
                non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
                tweet=tweet.translate(non_bmp_map)
                sentiment_value,confidence=s.sentiment(tweet)
                print(tweet,sentiment_value,confidence)
                self.count+=1
                if (confidence*100>=75):
                        output = open("twitter-out.txt","a")
                        output.write(sentiment_value)
                        output.write('\n')
                        output.close()
                if(self.count>self.stopat):
                        return False
                else:
                        return True
            
            except:
                    return True
            
    def on_error(self, status):
         print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["conjuring"],languages=["en"])
