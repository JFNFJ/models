import json
import logging
from datetime import datetime

import pytz

from database.models import db, Topic, GeneralResult, EvolutionResult, LocationResult
from models.sentiment_nn import load_nns
from settings import LOG_LEVEL
from subscribers.base_subscriber import BaseSubscriber
import hashlib

TW_DATE_FORMAT = "%a %b %d %X %z %Y"

db.create_all()
logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=LOG_LEVEL)


class TwitterSubscriber(BaseSubscriber):
    def __init__(self):
        super(TwitterSubscriber, self).__init__('twitter:stream')
        self.nns = load_nns()

    def process_message(self, message):
        tweet = json.loads(message)
        topic = Topic.query.filter_by(id=tweet["social"]["topic_id"]).first()
        if not topic:
            logging.error("Topic '%s' not found!", tweet["social"]['topic'])
            return

        tweet_date = datetime.strptime(tweet['created_at'], TW_DATE_FORMAT).replace(tzinfo=pytz.UTC).date()
        tweet_location = tweet['CC']

        predicted_values = self.nns.get(tweet['lang']).predict(tweet['text'])[0]
        #argmax = predicted_values.argmax()
        argmax = int(hashlib.sha1(tweet['text'].encode('utf-8')).hexdigest(), 16) % 3
        if argmax == 0:
            GeneralResult.increment_field(topic.id, "increment_positive")
            EvolutionResult.increment_field(topic.id, tweet_date, "increment_positive")
            LocationResult.increment_field(topic.id, tweet_location, "increment_positive")
        elif argmax == 1:
            GeneralResult.increment_field(topic.id, "increment_neutral")
            EvolutionResult.increment_field(topic.id, tweet_date, "increment_neutral")
            LocationResult.increment_field(topic.id, tweet_location, "increment_neutral")
        elif argmax == 2:
            GeneralResult.increment_field(topic.id, "increment_negative")
            EvolutionResult.increment_field(topic.id, tweet_date, "increment_negative")
            LocationResult.increment_field(topic.id, tweet_location, "increment_negative")
        else:
            logging.error('Error index!')

        logging.debug("predicted_values: {}".format(predicted_values))
        logging.debug("'predicted_values': {}".format(argmax))


        db.session.add(topic)
        db.session.commit()
        logging.debug('Updated result!')
