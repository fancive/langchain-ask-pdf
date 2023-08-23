# Langchain Ask PDF (Tutorial)

>You may find the step-by-step video tutorial to build this application [on Youtube](https://youtu.be/wUAUdEw5oxM).

This is a Python application that allows you to upload a PDF, ask questions about its content, and also pose pre-configured questions. The questions and their answers will be converted into audio format, which can be played directly within the app. The application leverages an LLM to generate the responses based on the content of the uploaded PDF.

## Features
- **Upload PDF**: Users can upload a PDF file to the application.
- **Ask Custom Questions**: After uploading, users can type in custom questions about the content of the PDF.
- **Pre-configured Questions**: The app also provides a set of pre-configured questions that users can edit. Once submitted, the app will automatically ask these questions about the PDF content.
- **Audio Responses**: Questions and their corresponding answers are converted into audio. Users can play the audio directly within the app.
- **History**: The app maintains a history of uploaded PDFs and their corresponding audio responses. Users can replay the audio or download the PDFs from the history.

## How it works

The application reads the uploaded PDF and extracts its text. This text is then split into manageable chunks suitable for processing by the LLM. The app utilizes OpenAI embeddings to create vector representations of these chunks. When a user poses a question, the application identifies text chunks that are semantically similar to the question. These relevant chunks are then provided to the LLM, which generates a response.

To facilitate user interaction, the application uses Streamlit for its GUI. The core logic and LLM interactions are handled by the Langchain library.

## Prerequisites

Before running the application, ensure you have `ffmpeg` and `ffprobe` installed as they are required for audio processing.

- **macOS** (using Homebrew):
  ```bash
  brew install ffmpeg
  ```
- **Linux** (e.g., using apt for Ubuntu):
  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```
- **Windows** 
  Download the precompiled binaries from the [ffmpeg official website](https://ffmpeg.org/download.html). After downloading and extracting, add the path to the binaries to your system's PATH environment variable.

## Installation
To set up the application:

1. Clone this repository.
1. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
1. Ensure you have the `uploads` and `audios` directories created in the root of the project to store the uploaded PDFs and generated audio files, respectively.
1. Add your OpenAI API key to the .env file for the LLM interactions.

## Usage
To use the application, run the `app.py` file with the streamlit CLI (after having installed streamlit):
```bash
streamlit run app.py
```

## Contributing
This repository is primarily for educational purposes. While it is an enhancement of the original tutorial project, it is still intended to serve as reference material rather than an actively developed project. Contributions are not actively sought, but insights and suggestions are always welcome.

