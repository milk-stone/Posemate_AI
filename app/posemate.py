import os
import pandas as pd
import numpy as np
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel

# 1. Import Modules
import recorder_ui
import data_processing

# --- Configuration (Update with your own environment settings) ---
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../vertex_key.json"
PROJECT_ID = "flowing-depot-465522-d5"
REGION = "us-central1"
ENDPOINT_ID = "447964126900125696"

# --- AI Model Initialization ---
# Initialize Vertex AI for both endpoint and Gemini
aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

# Initialize Gemini Model (uses Vertex AI authentication)
gemini_model = GenerativeModel("gemini-2.0-flash")


def predict_squat_posture(instances):
    """Sends an inference request to the Vertex AI endpoint and returns the results."""
    try:
        response = endpoint.predict(instances=instances)
        predictions = []
        for pred in response.predictions:
            scores = pred['scores']
            classes = pred['classes']
            best_index = np.argmax(scores)
            predictions.append({
                "label": classes[best_index],
                "score": scores[best_index]
            })
        return predictions
    except Exception as e:
        print(f"❌ Vertex AI Prediction Failed: {e}")
        return None


def get_feedback_from_gemini(squat_data, prediction_result):
    """Calls the Gemini API to generate feedback on the posture."""
    print("   - 🤖 Calling Gemini API...")
    try:
        # Select only the features used for inference (exclude squat_idx)
        feature_data = squat_data.drop('squat_idx')

        # Convert the 20 feature values into a readable string format
        feature_text = "\n".join([f"- {key}: {value:.4f}" for key, value in feature_data.items()])

        # Construct the full prompt in English
        prompt = f"""You are a squat exercise expert. Your role is to advise the user on how to achieve a correct posture in the future, based on their body angles and the evaluation result of their posture.

Here is the additional information you have:
(Evaluation Result)
{prediction_result['label']}

(Confidence Score)
{prediction_result['score']:.2f}

(20 Feature Values Used for Inference)
{feature_text}

Based on the information above, please provide friendly and specific advice in clear English that the user can easily understand.
"""

        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Gemini API Call Failed: {e}")
        return "Failed to generate AI feedback. Please check your API settings."


def main():
    """Executes the main workflow."""
    # --- Step 1: Record data from webcam ---
    recorder = recorder_ui.SquatRecorder()
    raw_data_df = recorder.start()

    if raw_data_df is None:
        print("\nNo data was recorded. Exiting the program.")
        return

    # --- Step 2: Process and normalize data ---
    manipulated_df = data_processing.transform_raw_data_to_manipulated(raw_data_df)
    final_df = data_processing.process_manipulated_data_to_final(manipulated_df)

    if final_df is None:
        print("\nAn error occurred during data processing. Exiting the program.")
        return

    # --- Step 3: Vertex AI Inference ---
    print(f"\n--- Step 3: Starting Vertex AI Inference ---")

    # Create prediction instances using all features ('stand_', 'sit_')
    prediction_instances = []
    feature_columns = [col for col in final_df.columns if col != 'squat_idx']
    for index, row in final_df.iterrows():
        # Convert each value to a string for the API request
        instance = {key: str(value) for key, value in row[feature_columns].items()}
        prediction_instances.append(instance)

    # Send inference request to the endpoint
    predictions = predict_squat_posture(prediction_instances)

    # --- Step 4: Display results and feedback ---
    print("\n--- Final Analysis Results ---")
    if predictions:
        for i, prediction in enumerate(predictions):
            squat_idx = final_df.loc[i, 'squat_idx']
            label = prediction['label']
            score = prediction['score']

            print(f"✅ Squat #{squat_idx}: {label.upper()} (Confidence: {score:.2f})")

            if label == 'incorrect':
                # Pass the full data for that squat and the prediction result
                feedback = get_feedback_from_gemini(final_df.loc[i], prediction)
                print(f"   - 🤖 AI Feedback: {feedback}")
    else:
        print("Could not retrieve inference results.")


if __name__ == '__main__':
    main()
