from flask import Flask , render_template , request
import pandas as pd
import pickle as pkl
from sklearn.base import BaseEstimator , TransformerMixin
from sklearn.preprocessing import StandardScaler
class Standard_Car(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        sc= StandardScaler()
        X['kms_driven'] = sc.fit_transform(X[['kms_driven']])
        return X
app = Flask(__name__)
model,cols = pkl.load(open("model.pkl", "rb"))
@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        data = pd.DataFrame({})
        for k in  cols:
            data[k] = [request.form[k]]
        pre = model.predict(data)
        return render_template('index.html', output=f"output : {pre[0]}")
    return render_template('index.html', output="")

app.run(debug=True)

# pkl.dump([model , ["name","company","year","kms_driven","fuel_type"]], open("model.pkl","wb"))
