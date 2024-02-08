from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No selected file')
    if file:
        df = pd.read_csv(file)
        # Generate plots and related output
        fig, ax = plt.subplots()
        # Example plot
        df.plot(ax=ax)
        # Save plot to a BytesIO object
        plot_buffer = BytesIO()
        plt.savefig(plot_buffer, format='png')
        plot_buffer.seek(0)
        plot_data_uri = base64.b64encode(plot_buffer.read()).decode('utf-8')
        # Convert DataFrame to HTML table
        table_html = df.to_html()
        return render_template('index.html', plot_data_uri=plot_data_uri, table_html=table_html)

if __name__ == '__main__':
    app.run(debug=True)
