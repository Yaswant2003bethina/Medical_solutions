from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the ML models
name_model = joblib.load('name_model.pkl')
uses_model = joblib.load('uses_model.pkl')
side_effects_model = joblib.load('side_effects_model.pkl')
Composition_model = joblib.load('Composition.pkl')
Manufacturer_model = joblib.load('Manufacturer.pkl')

# Load the dataset
data = pd.read_csv('Medicine_Details.csv')

@app.route('/')
def index():
    return render_template('index.html', uses=get_uses(), medicines=get_medicines())

@app.route('/predict_medicine_name', methods=['POST'])
def predict_medicine_name():
    uses = request.form['uses']
    predictions = predict_name_and_side_effects(uses)
    if not predictions:
        return render_template('error.html', message="Invalid input. Please enter a valid use case.")
    return render_template('output_medicine_name.html', predictions=predictions)


@app.route('/predict_uses', methods=['POST'])
def predict_uses():
    medicine_name = request.form['medicine_name']
    predictions = predict_uses_and_side_effects(medicine_name)
    if not predictions:
        return render_template('error.html', message="Invalid input. Please enter a valid medicine name.")
    return render_template('output_uses.html', predictions=predictions)


def predict_name_and_side_effects(uses):
    predictions = []
    for index, row in data[data['Uses'] == uses].iterrows():
        name_prediction = name_model.predict([uses])
        side_effects = predict_side_effects(uses)
        predictions.append({'medicine_name':row['Medicine Name'] , 'image_url': row['Image URL'], 'side_effects': row['Side_effects']})
    return predictions

def predict_uses_and_side_effects(medicine_name):
    predictions = []
    for index, row in data[data['Medicine Name'] == medicine_name].iterrows():
        uses_prediction = uses_model.predict([medicine_name])
        side_effects = predict_side_effects(medicine_name)
        composition = predict_composition(medicine_name)
        Manufacturer = predict_Manufacturer(medicine_name)
        Uses = predict_Manufacturer(medicine_name)
        predictions.append({'medicine_name':row['Medicine Name'],'uses': row['Uses'] , 'image_url': row['Image URL'], 'side_effects':  row['Side_effects'], 'composition': row['Composition'], 'Manufacturer': row['Manufacturer'],'Uses':Uses})
    return predictions

def predict_side_effects(input_data):
    side_effects_prediction = side_effects_model.predict([input_data])
    return side_effects_prediction[0]

def predict_composition(medicine_name):
    index = data[data['Medicine Name'] == medicine_name].index
    if len(index) == 0:
        return None
    index = index[0]
    composition = Composition_model[index]
    return composition

def predict_Manufacturer(medicine_name):
    index = data[data['Medicine Name'] == medicine_name].index
    if len(index) == 0:
        return None
    index = index[0]
    Manufacturer = Manufacturer_model[index]
    return Manufacturer

def get_uses():
    return data['Uses'].unique().tolist()

def get_medicines():
    return data['Medicine Name'].unique().tolist()

if __name__ == '__main__':
    app.run(debug=True)