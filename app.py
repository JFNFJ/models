from redis import StrictRedis
import settings as s

redis = StrictRedis(host=s.REDIS_HOST, port=s.REDIS_PORT)
ps = redis.pubsub()
ps.subscribe("twitter:stream")

for message in ps.listen():
    print(message)
