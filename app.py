from flask import Flask, request, render_template
import joblib
import numpy as np

#Create Flask app instance
app = Flask(__name__) #special built-in variable, name of the current module

# Load your model
model = joblib.load("svm_iris_model.joblib")  # Replace with your actual file name

# Map prediction output to Iris species names
iris_classes = ['setosa', 'versicolor', 'virginica']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form input
        sl = float(request.form['sepal_length'])
        sw = float(request.form['sepal_width'])
        pl = float(request.form['petal_length'])
        pw = float(request.form['petal_width'])

        # Prepare the input as a 2D array
        input_data = np.array([[sl, sw, pl, pw]])
        

        # Predict the class (index)
        prediction = model.predict(input_data)

        # Get class name from index
        result = iris_classes[prediction[0]]

        image_filename = f"{result}.jpg"  # e.g., setosa.jpg

        return render_template('index.html', prediction_text=f'Predicted Iris Class: {result}', image_file=image_filename)

    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
