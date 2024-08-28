import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pickle
import re
# Load the model and vectorizer
with open("naive_bayes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(
    style={'textAlign': 'center', 'width': '50%', 'margin': 'auto'},
    children=[
        dcc.Textarea(id="text-input", placeholder="Write your email here", style={"width": "100%", "height": "350px", "padding": "10px", "resize": "none"}),
        html.Div(id="prediction-output", style={"marginTop": "20px"})
        
    ]
)

# Define the callback for making predictions
@app.callback(
    Output('prediction-output', 'children'),
    Input('text-input', 'value'),
)
def update_output(value):
    if value:
        try:
            # Transform the input text using the loaded vectorizer
            value = re.sub(r'[^a-zA-Z0-9 ]', ' ', value) # remove punctuation
            value = value.lower() # lower
            value = re.sub(r"\s+", " ", value).strip() # merge spaces together
            X_new_tfidf = vectorizer.transform([value]).toarray()
            
            # Make prediction
            prediction = model.predict(X_new_tfidf)
            return f'{"Fraud/Spam email :(" if prediction else "Safe email :)"}'
        except Exception as e:
            return f'Error: {str(e)}'
    return "Begin typing"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
