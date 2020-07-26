from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        Age=int(request.form['Age'])
        Income=float(request.form['Income'])
        Family=int(request.form['Family'])
        CCAvg=float(request.form['CCAvg'])
        Education=int(request.form['Education'])
        Mortgage=float(request.form['Mortgage'])
        SecuritiesAccount=int(request.form['SecuritiesAccount'])
        CDAccount=int(request.form['CDAccount'])
        Online=int(request.form['Online'])
        CreditCard=int(request.form['CreditCard'])
       
        final_features = [Age,Income,Family,CCAvg,Education,Mortgage,SecuritiesAccount,CDAccount,Online,CreditCard]
        prediction = model.predict(final_features)

        #output = round(prediction[0], 2)
        output = prediction.astype('int')
        return render_template('result.html', prediction=output)


if __name__ == "__main__":
    app.run(debug=True)