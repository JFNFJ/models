import json
import random

from flask import Flask, request

#from models.sentiment_nn import Model

app = Flask(__name__)


class Response:
    def __init__(self, positive, neutral, negative):
        self.positive = positive
        self.neutral = neutral
        self.negative = negative


#model = Model()
#model.load_from_saved_weights('/tmp/best_weights_%s.hdf5')


@app.route("/process_data", methods=['POST'])
def process_data():
    body = request.get_data()
    lang = request.args.get('lang')
    if lang not in ['en', 'es']:
        return 'Unsupported language, only "es" and "en" available', 400
    r1 = random.random()
    r2 = random.random()
    r3 = random.random()
    sum = r1+r2+r3
    return json.dumps(Response(r1/sum, r2/sum, r3/sum).__dict__)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
