from flask import Flask, request, jsonify
from flask_cors import cross_origin
import main as module

app = Flask(__name__)
# cors = CORS(app, resources={r'/api/*', {'origins': "http://localhost:3000"}})
print("hello")

@app.route('/search', methods=['POST','GET'])
@cross_origin(origins=['http://localhost:3000'])
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

@app.route('/hello')
def hello():
    q = request.args.get('q')
    print(q)
    return { "message": "Hello!"}, 201

if __name__ == '__main__':
    app.run()