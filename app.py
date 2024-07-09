import pickle
from flask import Flask,render_template,request,app,jsonify,url_for
import numpy as np
import pandas as pd


app=Flask(__name__)
#Load the Model
#our_model=pickle.load(open('ourmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
our_model1=pickle.load(open('our_model1.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    out=our_model1.predict(new_data)
    print(out[0])
    return jsonify(out[0])

@app.route('/predict_api',methods=['POST'])
def predict_api():
    my_data=[float(x) for x in request.form.values()]
    print(my_data)
    final_input=scalar.transform(np.array(list(my_data)).reshape(1,-1))
    final_output=our_model1.predict(final_input)[0]
    print(final_output)
    return render_template('index.html',prediction_text='The Predicted House Price is {}'.format(final_output))
     
if __name__=="__main__":
    app.run(debug=True)

    
