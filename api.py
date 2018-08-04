from flask import Flask
import random

app = Flask(__name__)


@app.route("/")
def just_a_random():
    return str(random.uniform(-1, 1))
