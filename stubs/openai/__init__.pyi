from typing import Any, Dict, Optional, Union, BinaryIO


class Audio:
    @staticmethod
    def transcribe(
        model: str,
        file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, str]: ... 