import requests
from pathlib import Path


def transcribe_file(
    file_path, model=None, language=None, response_format=None, temperature=None
):
    url = "http://localhost:8080/v1/audio/transcriptions"

    # Create multipart form data
    files = {
        # Use actual filename and open file in binary mode
        "audio": (Path(file_path).name, open(file_path, "rb"), "audio/raw")
    }

    # Create form data dictionary
    form_data = {}
    if model:
        form_data["model"] = (None, model)
    if language:
        form_data["language"] = (None, language)
    if response_format:
        form_data["response_format"] = (None, response_format)
    if temperature is not None:
        form_data["temperature"] = (None, str(temperature))

    # Merge files and form_data
    files.update(form_data)

    # Make the POST request
    response = requests.post(url, files=files)  # Send everything as files parameter

    # Check if request was successful
    response.raise_for_status()
    return response.json()


# Example usage
try:
    result = transcribe_file(
        file_path="./talking_about_a_hike.raw",
        model="Systran/faster-whisper-large-v3",
        response_format="json",
        temperature=0.0,
    )
    print("Transcription Response:", result)
except requests.exceptions.RequestException as e:
    print("Error:", e)
