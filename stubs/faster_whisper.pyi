from typing import List, Optional, Tuple, Union, Dict, Any


class WhisperModel:
    def __init__(
        self,
        model_size_or_path: str,
        device: str = "auto",
        device_index: Optional[int] = None,
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None: ...

    def transcribe(
        self,
        audio: Union[str, bytes, List[float]],
        beam_size: int = 5,
        word_timestamps: bool = False,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
        temperature: Union[float, Tuple[float, ...]] = 0.0,
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text_size: int = 0,
        initial_prompt_size: int = 0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]: ... 