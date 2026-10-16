import numpy as np
import pickle
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model
with open('linear2.pkl', 'rb') as f:
  model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Insurance Cost Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: #ffffff;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 550px;
        }
        h2 {
            color: #1a202c;
            margin-bottom: 8px;
            font-weight: 700;
            text-align: center;
        }
        .subtitle {
            color: #718096;
            font-size: 14px;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #4a5568;
            font-weight: 500;
            font-size: 14px;
        }
        input, select {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            font-size: 15px;
            color: #2d3748;
            background: #f8fafc;
            transition: all 0.3s ease;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
            background: #ffffff;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15);
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s ease, transform 0.1s ease;
            margin-top: 10px;
        }
        button:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }
        .result-box {
            margin-top: 25px;
            padding: 20px;
            background: #ebf8ff;
            border: 1px solid #bee3f8;
            border-radius: 8px;
            text-align: center;
        }
        .result-box h3 {
            color: #2b6cb0;
            font-size: 16px;
            font-weight: 600;
        }
        .result-price {
            font-size: 28px;
            color: #2c5282;
            font-weight: 700;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Insurance Cost Predictor</h2>
        <p class="subtitle">Enter details below to estimate medical charges</p>
        
        <form method="POST">
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" placeholder="e.g., 30" required min="1" max="120">
            </div>

            <div class="form-group">
                <label for="sex">Sex</label>
                <select id="sex" name="sex" required>
                    <option value="0">Female</option>
                    <option value="1">Male</option>
                </select>
            </div>

            <div class="form-group">
                <label for="bmi">BMI</label>
                <input type="number" step="0.1" id="bmi" name="bmi" placeholder="e.g., 24.5" required min="10" max="60">
            </div>

            <div class="form-group">
                <label for="children">Number of Children</label>
                <input type="number" id="children" name="children" placeholder="e.g., 1" required min="0" max="10">
            </div>

            <div class="form-group">
                <label for="smoker">Smoker</label>
                <select id="smoker" name="smoker" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>

            <div class="form-group">
                <label for="region">Region</label>
                <select id="region" name="region" required>
                    <option value="0">Northeast</option>
                    <option value="1">Northwest</option>
                    <option value="2">Southeast</option>
                    <option value="3">Southwest</option>
                </select>
            </div>

            <button type="submit">Predict Cost</button>
        </form>

        {% if prediction_text %}
        <div class="result-box">
            <h3>Estimated Cost</h3>
            <div class="result-price">{{ prediction_text }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def home():
  prediction_text = None
  if request.method == 'POST':
    try:
      age = float(request.form['age'])
      sex = float(request.form['sex'])
      bmi = float(request.form['bmi'])
      children = float(request.form['children'])
      smoker = float(request.form['smoker'])
      region = float(request.form['region'])

      features = np.array([[age, sex, bmi, children, smoker, region]])
      prediction = model.predict(features)[0]
      prediction_text = f'${prediction:,.2f}'
    except Exception as e:
      prediction_text = f'Error: {str(e)}'

  return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text)


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
