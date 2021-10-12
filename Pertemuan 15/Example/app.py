import flask
import numpy as np
import pickle
from markupsafe import escape

app = flask.Flask(__name__, template_folder='templates')

model = pickle.load(open('model/dtree.pkl', 'rb'))


@app.route('/')
def index():
    return (flask.render_template('main.html'))


@app.route('/predict', methods=["POST"])
def predict():

    features = [int(x) for x in flask.request.form.values()]
    features = [np.array(features)]
    prediction = model.predict(features)

    output = {0: 'not placed', 1: 'placed'}

    return flask.render_template('main.html', prediction_text='Student must be {} to workplace'.format(output[prediction[0]]))


@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % escape(username)


if __name__ == '__main__':
    app.run(debug=True)
