import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pickle
import re

with open("naive_bayes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = dash.Dash(__name__)
server=app.server

app.layout = html.Div(
    style={'textAlign': 'center', 'width': '50%', 'margin': 'auto'},
    children=[
        html.Div(id="prediction-output", style={"marginTop": "10px", "marginBottom": "10px"}),

        dcc.Textarea(id="text-input", placeholder="Write your email here", 
                     style={"width": "100%", "height": "350px", "padding": "10px", "resize": "none"}),
        html.A(
            "Click here to download the infographic",
            href="/assets/infographic_poster.png",
            download="infographic_poster.png",
            className="download-link" 
        )
    ]
)


@app.callback(
    Output('prediction-output', 'children'),
    Input('text-input', 'value'),
)
def update_output(value):
    if value:
        try:
            # preprocessing
            value = re.sub(r'[^a-zA-Z0-9 ]', ' ', value) # remove punctuation
            value = value.lower() # lower
            value = re.sub(r"\s+", " ", value).strip() # merge spaces together
            X_new_tfidf = vectorizer.transform([value]).toarray()
            
            prediction = model.predict(X_new_tfidf)
            return f'{"Fraud/Spam email :(" if prediction else "Safe email :)"}'
        except Exception as e:
            return f'Error: {str(e)}'
    return "Begin typing"


if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
