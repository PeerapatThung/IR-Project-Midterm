from flask import Flask, request, jsonify
from flask_restful import reqparse, abort, Api, Resource
import main as module

app = Flask(__name__)

parser = reqparse.RequestParser()
parser.add_argument('query', type=str)
parser.add_argument('score', type=str)

@app.route('/search', methods=['POST','GET'])
def search():
    if request.method == 'POST':
        body = request.get_json()
        if body['score'] == 'tf':
            return module.tf_score(body['query'])
        if body['score'] == 'tf-idf':
            return module.tfidf_score(body['query'])
        if body['score'] == 'bm25':
            return module.bm25_score(body['query'])
        else:
            return {"message": "No scoring system"},200

# class CalculateScore(Resource):
#     def post(self):
#         args = parser.parse_args()
#         query = str(args['query'])
#         score = str(args['score'])
#         if score =='tf':
#             return jsonify(module.tf_score(query))
#         if score == 'tf-idf':
#             return jsonify(module.tfidf_score(query))
#         if score == 'bm25':
#             return jsonify(module.bm25_score(query))
#         else:
#             return None

@app.route('/hello')
def hello():
    q = request.args.get('q')
    print(q)
    return { "message": "Hello!"}, 201

# api.add_resource(CalculateScore, '/search')

if __name__ == '__main__':
    app.run()