import os


class Config:

    YOUTUBE_VIDEO_URL: str = ""
    YOUTUBE_AUDIO_SAVE_DIRECTORY: str = "docs/youtube/"

    PDF_SOURCE_DIRECTORY: str = "data"
    CHROMA_PERSIST_DIRECTORY: str = "docs/chroma"

    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
    CHUNK_SIZE = 2028
    CHUNK_OVERLAP = 250

    def __init__(self):
        os.makedirs(self.PDF_SOURCE_DIRECTORY, exist_ok=True)
        print(f"Configuration loaded. PDF documents should be placed in '{self.PDF_SOURCE_DIRECTORY}'.")


config = Config()