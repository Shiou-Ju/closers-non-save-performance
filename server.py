from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not all(feature in data for feature in ['OPS', 'BAbip', 'tOPS+', 'BA', 'SLG', 'OBP', 'Year', 'IsModernCloser']):
        return 'Bad Request', 400

    X = pd.DataFrame([data])

    y_pred = model.predict(X)

    return jsonify(y_pred.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007)

# curl -X POST -H "Content-Type: application/json" -d '{
#     "OPS": 0.737,
#     "BAbip": 0.300,
#     "tOPS+": 95,
#     "BA": 0.250,
#     "SLG": 0.400,
#     "OBP": 0.337,
#     "Year": 2019,
#     "IsModernCloser": 1
# }' http://127.0.0.1:5007/predict

# response
# [[0.7677010597290796,0.30662031741529017,105.27713511112603,0.25760737827629,0.4218073139370784,0.34593892246940217,2019.0,1.000000000000009]]
