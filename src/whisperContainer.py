# External programs
import os
from typing import List
import whisper
from src.config import ModelConfig

from src.modelCache import GLOBAL_MODEL_CACHE, ModelCache

class WhisperContainer:
    def __init__(self, model_name: str, device: str = None, download_root: str = None, 
                 cache: ModelCache = None, models: List[ModelConfig] = []):
        self.model_name = model_name
        self.device = device
        self.download_root = download_root
        self.cache = cache

        # Will be created on demand
        self.model = None

        # List of known models
        self.models = models
    
    def get_model(self):
        if self.model is None:

            if (self.cache is None):
                self.model = self._create_model()
            else:
                model_key = "WhisperContainer." + self.model_name + ":" + (self.device if self.device else '')
                self.model = self.cache.get(model_key, self._create_model)
        return self.model

    def ensure_downloaded(self):
        """
        Ensure that the model is downloaded. This is useful if you want to ensure that the model is downloaded before
        passing the container to a subprocess.
        """
        # Warning: Using private API here
        try:
            root_dir = self.download_root
            model_config = self.get_model_config()

            if root_dir is None:
                root_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")

            if self.model_name in whisper._MODELS:
                whisper._download(whisper._MODELS[self.model_name], root_dir, False)
            else:
                # If the model is not in the official list, see if it needs to be downloaded
                model_config.download_url(root_dir)
            return True
        
        except Exception as e:
            # Given that the API is private, it could change at any time. We don't want to crash the program
            print("Error pre-downloading model: " + str(e))
            return False

    def get_model_config(self) -> ModelConfig:
        """
        Get the model configuration for the model.
        """
        for model in self.models:
            if model.name == self.model_name:
                return model
        return None

    def _create_model(self):
        print("Loading whisper model " + self.model_name)
        
        model_config = self.get_model_config()
        # Note that the model will not be downloaded in the case of an official Whisper model
        model_path = model_config.download_url(self.download_root)

        return whisper.load_model(model_path, device=self.device, download_root=self.download_root)

    def create_callback(self, language: str = None, task: str = None, initial_prompt: str = None, **decodeOptions: dict):
        """
        Create a WhisperCallback object that can be used to transcript audio files.

        Parameters
        ----------
        language: str
            The target language of the transcription. If not specified, the language will be inferred from the audio content.
        task: str
            The task - either translate or transcribe.
        initial_prompt: str
            The initial prompt to use for the transcription.
        decodeOptions: dict
            Additional options to pass to the decoder. Must be pickleable.

        Returns
        -------
        A WhisperCallback object.
        """
        return WhisperCallback(self, language=language, task=task, initial_prompt=initial_prompt, **decodeOptions)

    # This is required for multiprocessing
    def __getstate__(self):
        return { "model_name": self.model_name, "device": self.device, "download_root": self.download_root, "models": self.models }

    def __setstate__(self, state):
        self.model_name = state["model_name"]
        self.device = state["device"]
        self.download_root = state["download_root"]
        self.models = state["models"]
        self.model = None
        # Depickled objects must use the global cache
        self.cache = GLOBAL_MODEL_CACHE


class WhisperCallback:
    def __init__(self, model_container: WhisperContainer, language: str = None, task: str = None, initial_prompt: str = None, **decodeOptions: dict):
        self.model_container = model_container
        self.language = language
        self.task = task
        self.initial_prompt = initial_prompt
        self.decodeOptions = decodeOptions
        
    def invoke(self, audio, segment_index: int, prompt: str, detected_language: str):
        """
        Peform the transcription of the given audio file or data.

        Parameters
        ----------
        audio: Union[str, np.ndarray, torch.Tensor]
            The audio file to transcribe, or the audio data as a numpy array or torch tensor.
        segment_index: int
            The target language of the transcription. If not specified, the language will be inferred from the audio content.
        task: str
            The task - either translate or transcribe.
        prompt: str
            The prompt to use for the transcription.
        detected_language: str
            The detected language of the audio file.

        Returns
        -------
        The result of the Whisper call.
        """
        model = self.model_container.get_model()

        return model.transcribe(audio, \
                 language=self.language if self.language else detected_language, task=self.task, \
                 initial_prompt=self._concat_prompt(self.initial_prompt, prompt) if segment_index == 0 else prompt, \
                 **self.decodeOptions)

    def _concat_prompt(self, prompt1, prompt2):
        if (prompt1 is None):
            return prompt2
        elif (prompt2 is None):
            return prompt1
        else:
            return prompt1 + " " + prompt2
