import requests


def transcribe_file(
    file_path, model=None, language=None, response_format=None, temperature=None
):
    # URL of your FastAPI endpoint
    url = "http://localhost:8080/v1/audio/transcriptions"

    # Prepare the files and form data
    files = {"audio": ("audio.wav", open(file_path, "rb"))}

    # Prepare form data
    data = {}
    if model:
        data["model"] = model
    if language:
        data["language"] = language
    if response_format:
        data["response_format"] = response_format
    if temperature is not None:
        data["temperature"] = str(temperature)

    # Make the POST request
    response = requests.post(url, files=files, data=data)

    # Check if request was successful
    response.raise_for_status()

    # Return the JSON response
    return response.json()


# Example usage
try:
    result = transcribe_file(
        file_path="./talking_about_a_hike.raw",
        model="whisper-1",
        response_format="json",
        temperature=0.0,
    )
    print("Transcription Response:", result)
except requests.exceptions.RequestException as e:
    print("Error:", e)
