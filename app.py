

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
#jesai url come it will render to index page
@app.route('/')
def home():
    return render_template('index.html')
#######providing features to my model.pkl so that my model can give us some output. /predict call the predict func
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #take all values from form as input using request lib
    #int_features = [int(x) for x in request.form.values()]
    #print(int_features)
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    #output = (prediction[0])
    int_features=[]
    for x in request.form.values():
        if x=="male":
            int_features.append(1)
        elif x=="female":
             int_features.append(0)
        else:
            int_features.append(int(x))


    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = (prediction[0])

    return render_template('index.html', prediction_text='Predicted Personality is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
