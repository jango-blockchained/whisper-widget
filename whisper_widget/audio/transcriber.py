"""Audio transcription module for both local and OpenAI methods."""

import unicodedata
from typing import Optional, TypedDict
import openai
from faster_whisper import WhisperModel


class OpenAIResponse(TypedDict):
    """Response type for OpenAI transcription API."""
    text: str


class Transcriber:
    """Handles audio transcription using either local Whisper model or OpenAI API."""

    def __init__(
        self,
        mode: str = "local",
        model_size: str = "base",
        language: str = "en",
        add_punctuation: bool = True,
        openai_api_key: Optional[str] = None
    ):
        """Initialize the transcriber with specified settings."""
        self.mode = mode
        self.model_size = model_size
        self.language = language
        self.add_punctuation = add_punctuation
        self.model = None
        
        if mode == "local":
            try:
                self.model = WhisperModel(
                    model_size_or_path=model_size,
                    device="cpu",
                    compute_type="int8"
                )
            except Exception as e:
                print(f"Error initializing Whisper model: {e}")
                self.model = None
        elif mode == "openai":
            if openai_api_key:
                openai.api_key = openai_api_key

    def transcribe(self, audio_filename: str) -> str:
        """
        Transcribe the audio file using the selected method.
        
        Args:
            audio_filename: Path to the audio file to transcribe
            
        Returns:
            str: Transcribed text
        """
        try:
            if self.mode == "local":
                return self._transcribe_local(audio_filename)
            elif self.mode == "openai":
                return self._transcribe_openai(audio_filename)
            else:
                print(f"Error: Invalid transcription mode '{self.mode}'")
                return ""
                
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def _transcribe_local(self, audio_filename: str) -> str:
        """Transcribe using local Whisper model."""
        if not self.model:
            print("Error: Whisper model not initialized")
            return ""
            
        segments, info = self.model.transcribe(
            audio_filename,
            beam_size=5,
            word_timestamps=True,
            language=self.language,
            condition_on_previous_text=True,
            no_speech_threshold=0.6
        )
        
        try:
            # First try to normalize the text
            text = " ".join([
                unicodedata.normalize('NFKC', seg.text.strip())
                for seg in segments
            ])
            # Then encode and decode with error handling
            text = text.encode('utf-8', 'ignore').decode('utf-8')
        except Exception as e:
            print(f"Warning: Text encoding issue: {e}")
            # Fallback to a more aggressive normalization
            try:
                text = " ".join([
                    unicodedata.normalize('NFKD', seg.text.strip())
                    .encode('ascii', 'ignore')
                    .decode('ascii')
                    for seg in segments
                ])
            except Exception as e:
                print(f"Error: Failed to normalize text: {e}")
                return ""
        
        text = text.strip()
        
        # Add basic punctuation if enabled
        if text and self.add_punctuation and not text[-1] in '.!?':
            text += '.'
            
        return text

    def _transcribe_openai(self, audio_filename: str) -> str:
        """Transcribe using OpenAI's API."""
        if not openai.api_key:
            print("Error: OpenAI API key not set")
            return ""
            
        try:
            with open(audio_filename, "rb") as audio_file:
                result: OpenAIResponse = openai.Audio.transcribe(
                    "whisper-1",
                    audio_file
                )
                return result["text"].strip()
        except Exception as e:
            print(f"OpenAI transcription error: {e}")
            return ""

    def update_settings(
        self,
        mode: Optional[str] = None,
        model_size: Optional[str] = None,
        language: Optional[str] = None,
        add_punctuation: Optional[bool] = None,
        openai_api_key: Optional[str] = None
    ) -> None:
        """Update transcriber settings."""
        if mode is not None and mode != self.mode:
            self.mode = mode
            # Reinitialize model if switching to local mode
            if mode == "local":
                try:
                    self.model = WhisperModel(
                        model_size_or_path=self.model_size,
                        device="cpu",
                        compute_type="int8"
                    )
                except Exception as e:
                    print(f"Error initializing Whisper model: {e}")
                    self.model = None
                    
        if model_size is not None and model_size != self.model_size:
            self.model_size = model_size
            if self.mode == "local":
                try:
                    self.model = WhisperModel(
                        model_size_or_path=model_size,
                        device="cpu",
                        compute_type="int8"
                    )
                except Exception as e:
                    print(f"Error initializing Whisper model: {e}")
                    self.model = None
                    
        if language is not None:
            self.language = language
            
        if add_punctuation is not None:
            self.add_punctuation = add_punctuation
            
        if openai_api_key is not None and self.mode == "openai":
            openai.api_key = openai_api_key 