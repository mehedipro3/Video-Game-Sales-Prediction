import gradio as gr
import pandas as pd
import pickle
import numpy as np

# Load the trained model
with open("video_game_sales_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the trained model
def predict_sales(
    platform,
    genre,
    publisher,
    year,
    na_sales,
    eu_sales,
    jp_sales,
    other_sales
):
    input_data = pd.DataFrame([{
        'Platform': platform,
        'Genre': genre,
        'Publisher': publisher,
        'Year': year,
        'NA_Sales': na_sales,
        'EU_Sales': eu_sales,
        'JP_Sales': jp_sales,
        'Other_Sales': other_sales
    }])

    prediction = model.predict(input_data)[0]
    return f"Predicted Global Sales: {prediction:.2f} million units"

# Define Gradio interface

interface = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Dropdown(['Wii', 'PS4', 'PS3', 'X360', 'PC'], label="Platform"),
        gr.Dropdown(['Action', 'Sports', 'Racing', 'Shooter', 'Role-Playing'], label="Genre"),
        gr.Dropdown(['Nintendo', 'Electronic Arts', 'Activision', 'Sony'], label="Publisher"),
        gr.Number(label="Release Year"),
        gr.Number(label="NA Sales"),
        gr.Number(label="EU Sales"),
        gr.Number(label="JP Sales"),
        gr.Number(label="Other Sales"),
    ],
    outputs="text",
    title="🎮 Video Game Global Sales Predictor",
    description="Enter game details to predict global sales using a trained ML model."
)

interface.launch()
