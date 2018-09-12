import json
import logging
from datetime import datetime

import pytz

from models.sentiment_nn import load_nns
from subscribers.base_subscriber import BaseSubscriber

from settings import LOG_LEVEL
from database.models import db, Topic, GeneralResult, EvolutionResult, LocationResult

db.create_all()
logging.basicConfig(format='[%(levelname)s:%(asctime)s] %(message)s', level=LOG_LEVEL)


class TwitterSubscriber(BaseSubscriber):
    def __init__(self):
        super(TwitterSubscriber, self).__init__('twitter:stream')
        self.nns = load_nns()

    def process_message(self, message):
        tweet = json.loads(message)
        topic = Topic.query.filter_by(name=tweet['topic']).first()
        if not topic:
            logging.error("Topic '%s' not found!", tweet['topic'])
            return

        if not topic.general_result:
            topic.general_result = GeneralResult(topic=topic, positive=0, neutral=0, negative=0)

        tweet_date = datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y').replace(
            tzinfo=pytz.UTC).date()
        evolution_result = next(iter([er for er in topic.evolution_results if er.day == tweet_date]), None)
        if not evolution_result:
            evolution_result = EvolutionResult(topic=topic, day=tweet_date, positive=0, neutral=0, negative=0)
            topic.evolution_results.append(evolution_result)

        tweet_location = tweet['user']['location']
        location_result = next(iter([lr for lr in topic.location_results if lr.location == tweet_location]), None)
        if not location_result:
            location_result = LocationResult(topic=topic, location=tweet_location, positive=0, neutral=0, negative=0)
            topic.location_results.append(location_result)

        predicted_values = self.nns.get(tweet['lang']).predict(tweet['full_text'])[0]
        argmax = predicted_values.argmax()
        if argmax == 0:
            topic.general_result.increment_positive()
            evolution_result.increment_positive()
            location_result.increment_positive()
        if argmax == 1:
            topic.general_result.increment_neutral()
            evolution_result.increment_neutral()
            location_result.increment_neutral()
        if argmax == 2:
            topic.general_result.increment_negative()
            evolution_result.increment_negative()
            location_result.increment_negative()
        else:
            logging.error('Error index!')

        logging.debug("predicted_values: {}".format(predicted_values))

        logging.debug(topic.general_result)
        logging.debug(topic.evolution_results)
        logging.debug(topic.location_results)

        db.session.add(topic)
        db.session.commit()
        logging.debug('Updated result!')
