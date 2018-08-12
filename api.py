from flask import Flask, request
import random

app = Flask(__name__)

@app.route("/rank", methods=['GET'])
def just_a_random():
    text = request.args.get('text')
    app.logger.debug("text: %s", text)
    if not text:
        return "0"
    return str(random.uniform(-1, 1))

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')