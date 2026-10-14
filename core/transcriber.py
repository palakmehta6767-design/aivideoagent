import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None


def load_model():
    global _model

    if _model is None:
        try:
            import whisper
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Whisper could not be loaded. Install a compatible CPU PyTorch build "
                "for this Python environment, then restart the app."
            ) from exc
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")

    return _model


def transcribe_chunk_whisper(chunk_path: str, language: str = "english") -> str:
    model = load_model()

    language_code = {"english": "en", "hinglish": "hi"}.get(language.lower(), language)
    result = model.transcribe(
        chunk_path,
        task="transcribe",
        language=language_code,
    )

    return result["text"]


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Transcribe one audio chunk using local Whisper.
    """
    return transcribe_chunk_whisper(chunk_path, language)


def transcribe_all(chunks: list, language: str = "english") -> str:
    """
    Transcribe all audio chunks using local Whisper.
    """

    full_transcript = ""

    print("Using Whisper for transcription.")

    for i, chunk in enumerate(chunks):

        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        text = transcribe_chunk(chunk, language)

        full_transcript += text + " "

    print("Transcription complete.")

    return full_transcript.strip()