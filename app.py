import os
import io
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from gtts import gTTS
from pydub import AudioSegment
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from pydub import AudioSegment
import tempfile

# Global history list to keep track of uploaded files and generated audios
history = []

# Ensure the required directories exist
for directory in ["uploads", "audios"]:
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_audio_for_question_and_answer(q, response):
    combined_audio = AudioSegment.empty()

    # Convert question to audio and save to temp file
    question_audio = gTTS(text=q, lang="zh")
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        question_audio.save(fp.name)
        combined_audio += AudioSegment.from_mp3(fp.name)

    # Convert answer to audio and save to temp file
    answer_audio = gTTS(text=response, lang="zh")
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        answer_audio.save(fp.name)
        combined_audio += AudioSegment.from_mp3(fp.name)

    return combined_audio


def extract_text_from_pdf(pdf):
    """Extract text content from given PDF."""
    pdf_reader = PdfReader(pdf)
    return "".join(page.extract_text() for page in pdf_reader.pages)


def display_history():
    # 获取 uploads 和 audios 目录中的所有文件
    uploaded_pdfs = os.listdir("uploads")
    generated_audios = os.listdir("audios")

    # 自定义 CSS 为按钮增加垂直填充
    custom_css = """
        <style>
            .stButton>button {
                padding-top: 0.05em;
                padding-bottom: 0.05em;
            }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # 在侧边栏上添加一个标题
    st.sidebar.title("History")

    # 展示每一个文件
    for pdf_name, audio_name in zip(uploaded_pdfs, generated_audios):
        pdf_path = os.path.join("uploads", pdf_name)
        audio_path = os.path.join("audios", audio_name)

        # 使用 columns 创建两个并排的部分
        col1, col2 = st.sidebar.beta_columns([3, 1])

        # 在第一列放置文件名（并加粗）和下载链接
        with col1:
            st.markdown(f"**{pdf_name}** [Down]({pdf_path})")

        # 在第二列放置删除按钮
        with col2:
            if st.button(f"Del"):
                os.remove(pdf_path)
                os.remove(audio_path)
                st.sidebar.success(f"Deleted {pdf_name}")

        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.sidebar.audio(audio_bytes, format="audio/mp3")


def save_uploaded_pdf(pdf):
    """Save the uploaded PDF and return its path."""
    pdf_path = os.path.join("uploads", pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(pdf.getvalue())
    return pdf_path


def process_questions_and_generate_audio(pdf_path, questions):
    """Process the uploaded PDF, ask questions, and generate combined audio for answers."""
    text = extract_text_from_pdf(pdf_path)
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")

    overall_audio = AudioSegment.empty()
    for q in questions:
        docs = knowledge_base.similarity_search(q)
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=q)

        question_and_answer_audio = generate_audio_for_question_and_answer(q, response)
        question_and_answer_audio = adjust_playback_speed(
            question_and_answer_audio, speed=1.25
        )

        overall_audio += question_and_answer_audio

        # Display each question, answer, and the corresponding audio
        st.write("Question:", q)
        st.write("Answer:", response)

        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
            question_and_answer_audio.export(fp.name, format="mp3")
            st.audio(fp.name, format="audio/mp3")

    # Save the combined audio
    audio_path = os.path.join("audios", os.path.basename(pdf_path))
    overall_audio.export(audio_path, format="mp3")
    st.write("Combined Audio:")
    st.audio(audio_path, format="audio/mp3")

    return audio_path


def adjust_playback_speed(audio, speed=1.0):
    """
    Adjust the playback speed of the audio.
    :param audio: AudioSegment object
    :param speed: desired speed (e.g., 1.25 for 1.25x speed)
    :return: AudioSegment object with adjusted speed
    """
    return audio._spawn(
        audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * speed)}
    ).set_frame_rate(audio.frame_rate)


def main():
    global history

    load_dotenv()
    st.set_page_config(page_title=" ")
    st.header(" ")

    display_history()

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # Load pre-configured questions from the file
    with open("questions.txt", "r") as f:
        default_questions = f.readlines()

    # Allow user to edit pre-configured questions
    questions = st.text_area(
        "Edit questions if needed:",
        value="".join(default_questions),
        height=200,
    ).splitlines()

    if st.button("Submit and Process") and pdf:
        pdf_path = save_uploaded_pdf(pdf)
        audio_path = process_questions_and_generate_audio(pdf_path, questions)

        history.append(
            {
                "filename": os.path.basename(pdf_path),
                "pdf_path": pdf_path,
                "audio_path": audio_path,
            }
        )


if __name__ == "__main__":
    main()
