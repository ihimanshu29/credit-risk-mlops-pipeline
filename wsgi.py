# Save this as wsgi.py in your project root

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
# Import the specific class needed
from mlProject.pipeline.prediction import PredictionPipeline 

# Initialize the pipeline object ONCE globally when the application starts
# This makes subsequent predictions much faster!
PREDICTOR_OBJ = PredictionPipeline()

app = Flask(__name__) # initializing a flask app


@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

# The /train route is removed. Training should be done offline/via CI/CD.
# Do NOT run os.system() in a production web app route.

@app.route('/predict', methods=['POST', 'GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            # Reads incoming form variables mapped to our credit risk schema
            data = [
                float(request.form['person_age']),
                float(request.form['person_income']),
                float(request.form['loan_amnt']),
                float(request.form['loan_int_rate']),
                float(request.form['cb_person_cred_hist_length'])
            ]
            
            # Reshapes into flat row vectors matching standard modeling contracts (1 row, 5 cols)
            data = np.array(data).reshape(1, 5)
            predict = PREDICTOR_OBJ.predict(data)

            # Map the binary prediction to a human-readable result
            result_text = "High Risk (Default)" if predict[0] == 1 else "Safe (No Default)"

            return render_template('results.html', prediction = result_text)            

        except Exception as e:
            print(f'Prediction Exception: {e}')
            return render_template('error.html', message='An error occurred during prediction.')

    else:
        # GET request or initial load
        return render_template('index.html')


# Gunicorn will handle the production run, so the __main__ block is no longer needed.
# For local testing, you can add it back:
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)