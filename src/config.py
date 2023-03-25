import urllib

import os
from typing import List
from urllib.parse import urlparse
import json5
import torch

from tqdm import tqdm

from src.conversion.hf_converter import convert_hf_whisper

class ModelConfig:
    def __init__(self, name: str, url: str, path: str = None, type: str = "whisper"):
        """
        Initialize a model configuration.

        name: Name of the model
        url: URL to download the model from
        path: Path to the model file. If not set, the model will be downloaded from the URL.
        type: Type of model. Can be whisper or huggingface.
        """
        self.name = name
        self.url = url
        self.path = path
        self.type = type

    def download_url(self, root_dir: str):
        import whisper

        # See if path is already set
        if self.path is not None:
            return self.path
        
        if root_dir is None:
            root_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")

        model_type = self.type.lower() if self.type is not None else "whisper"

        if model_type in ["huggingface", "hf"]:
            self.path = self.url
            destination_target = os.path.join(root_dir, self.name + ".pt")

            # Convert from HuggingFace format to Whisper format
            if os.path.exists(destination_target):
                print(f"File {destination_target} already exists, skipping conversion")
            else:
                print("Saving HuggingFace model in Whisper format to " + destination_target)
                convert_hf_whisper(self.url, destination_target)

            self.path = destination_target

        elif model_type in ["whisper", "w"]:
            self.path = self.url

            # See if URL is just a file
            if self.url in whisper._MODELS:
                # No need to download anything - Whisper will handle it
                self.path = self.url
            elif self.url.startswith("file://"):
                # Get file path
                self.path = urlparse(self.url).path
            # See if it is an URL
            elif self.url.startswith("http://") or self.url.startswith("https://"):
                # Extension (or file name)
                extension = os.path.splitext(self.url)[-1]
                download_target = os.path.join(root_dir, self.name + extension)

                if os.path.exists(download_target) and not os.path.isfile(download_target):
                    raise RuntimeError(f"{download_target} exists and is not a regular file")

                if not os.path.isfile(download_target):
                    self._download_file(self.url, download_target)
                else:
                    print(f"File {download_target} already exists, skipping download")

                self.path = download_target
            # Must be a local file
            else:
                self.path = self.url

        else:
            raise ValueError(f"Unknown model type {model_type}")

        return self.path

    def _download_file(self, url: str, destination: str):
        with urllib.request.urlopen(url) as source, open(destination, "wb") as output:
            with tqdm(
                total=int(source.info().get("Content-Length")),
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

class ApplicationConfig:
    def __init__(self, models: List[ModelConfig] = [], input_audio_max_duration: int = 600, 
                 share: bool = False, server_name: str = None, server_port: int = 7860, 
                 queue_concurrency_count: int = 1, delete_uploaded_files: bool = True,
                 default_model_name: str = "medium", default_vad: str = "silero-vad", 
                 vad_parallel_devices: str = "", vad_cpu_cores: int = 1, vad_process_timeout: int = 1800, 
                 auto_parallel: bool = False, output_dir: str = None,
                 model_dir: str = None, device: str = None, 
                 verbose: bool = True, task: str = "transcribe", language: str = None,
                 vad_merge_window: float = 5, vad_max_merge_size: float = 30,
                 vad_padding: float = 1, vad_prompt_window: float = 3,
                 temperature: float = 0, best_of: int = 5, beam_size: int = 5,
                 patience: float = None, length_penalty: float = None,
                 suppress_tokens: str = "-1", initial_prompt: str = None,
                 condition_on_previous_text: bool = True, fp16: bool = True,
                 temperature_increment_on_fallback: float = 0.2, compression_ratio_threshold: float = 2.4,
                 logprob_threshold: float = -1.0, no_speech_threshold: float = 0.6):
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.models = models
        
        # WebUI settings
        self.input_audio_max_duration = input_audio_max_duration
        self.share = share
        self.server_name = server_name
        self.server_port = server_port
        self.queue_concurrency_count = queue_concurrency_count
        self.delete_uploaded_files = delete_uploaded_files

        self.default_model_name = default_model_name
        self.default_vad = default_vad
        self.vad_parallel_devices = vad_parallel_devices
        self.vad_cpu_cores = vad_cpu_cores
        self.vad_process_timeout = vad_process_timeout
        self.auto_parallel = auto_parallel
        self.output_dir = output_dir

        self.model_dir = model_dir
        self.device = device
        self.verbose = verbose
        self.task = task
        self.language = language
        self.vad_merge_window = vad_merge_window
        self.vad_max_merge_size = vad_max_merge_size
        self.vad_padding = vad_padding
        self.vad_prompt_window = vad_prompt_window
        self.temperature = temperature
        self.best_of = best_of
        self.beam_size = beam_size
        self.patience = patience
        self.length_penalty = length_penalty
        self.suppress_tokens = suppress_tokens
        self.initial_prompt = initial_prompt
        self.condition_on_previous_text = condition_on_previous_text
        self.fp16 = fp16
        self.temperature_increment_on_fallback = temperature_increment_on_fallback
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        
    def get_model_names(self):
        return [ x.name for x in self.models ]

    def update(self, **new_values):
        result = ApplicationConfig(**self.__dict__)

        for key, value in new_values.items():
            setattr(result, key, value)
        return result

    @staticmethod
    def create_default(**kwargs):
        app_config = ApplicationConfig.parse_file(os.environ.get("WHISPER_WEBUI_CONFIG", "config.json5"))

        # Update with kwargs
        if len(kwargs) > 0:
            app_config = app_config.update(**kwargs)
        return app_config

    @staticmethod
    def parse_file(config_path: str):
        import json5

        with open(config_path, "r") as f:
            # Load using json5
            data = json5.load(f)
            data_models = data.pop("models", [])

            models = [ ModelConfig(**x) for x in data_models ]

            return ApplicationConfig(models, **data)
