# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from recommender import Recommender

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False  # do not sort keys in the returning JSON

rc = Recommender('data.csv')  # instantiate recommender


@app.route('/', methods=['GET'])
def home():
    return "Recommendify Model Server"


@app.route('/recommend', methods=['POST'])
def recommend():
    ids = request.json['id']
    recommends = rc.recommend(ids)
    print(recommends)
    return jsonify({
        "status": True,
        "data": recommends
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0")
