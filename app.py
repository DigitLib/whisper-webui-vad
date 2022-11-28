import math
from typing import Iterator
import argparse

from io import StringIO
import os
import pathlib
import tempfile

import torch
from src.modelCache import ModelCache
from src.vadParallel import ParallelContext, ParallelTranscription

# External programs
import ffmpeg

# UI
import gradio as gr

from src.download import ExceededMaximumDuration, download_url
from src.utils import slugify, write_srt, write_vtt
from src.vad import AbstractTranscription, NonSpeechStrategy, PeriodicTranscriptionConfig, TranscriptionConfig, VadPeriodicTranscription, VadSileroTranscription
from src.whisperContainer import WhisperContainer

# Limitations (set to -1 to disable)
DEFAULT_INPUT_AUDIO_MAX_DURATION = 600 # seconds

# Whether or not to automatically delete all uploaded files, to save disk space
DELETE_UPLOADED_FILES = True

# Gradio seems to truncate files without keeping the extension, so we need to truncate the file prefix ourself 
MAX_FILE_PREFIX_LENGTH = 17

# Limit auto_parallel to a certain number of CPUs (specify vad_cpu_cores to get a higher number)
MAX_AUTO_CPU_CORES = 8

LANGUAGES = [ 
 "English", "Chinese", "German", "Spanish", "Russian", "Korean", 
 "French", "Japanese", "Portuguese", "Turkish", "Polish", "Catalan", 
 "Dutch", "Arabic", "Swedish", "Italian", "Indonesian", "Hindi", 
 "Finnish", "Vietnamese", "Hebrew", "Ukrainian", "Greek", "Malay", 
 "Czech", "Romanian", "Danish", "Hungarian", "Tamil", "Norwegian", 
 "Thai", "Urdu", "Croatian", "Bulgarian", "Lithuanian", "Latin", 
 "Maori", "Malayalam", "Welsh", "Slovak", "Telugu", "Persian", 
 "Latvian", "Bengali", "Serbian", "Azerbaijani", "Slovenian", 
 "Kannada", "Estonian", "Macedonian", "Breton", "Basque", "Icelandic", 
 "Armenian", "Nepali", "Mongolian", "Bosnian", "Kazakh", "Albanian",
 "Swahili", "Galician", "Marathi", "Punjabi", "Sinhala", "Khmer", 
 "Shona", "Yoruba", "Somali", "Afrikaans", "Occitan", "Georgian", 
 "Belarusian", "Tajik", "Sindhi", "Gujarati", "Amharic", "Yiddish", 
 "Lao", "Uzbek", "Faroese", "Haitian Creole", "Pashto", "Turkmen", 
 "Nynorsk", "Maltese", "Sanskrit", "Luxembourgish", "Myanmar", "Tibetan",
 "Tagalog", "Malagasy", "Assamese", "Tatar", "Hawaiian", "Lingala", 
 "Hausa", "Bashkir", "Javanese", "Sundanese"
]

class WhisperTranscriber:
    def __init__(self, input_audio_max_duration: float = DEFAULT_INPUT_AUDIO_MAX_DURATION, vad_process_timeout: float = None, vad_cpu_cores: int = 1, delete_uploaded_files: bool = DELETE_UPLOADED_FILES):
        self.model_cache = ModelCache()
        self.parallel_device_list = None
        self.gpu_parallel_context = None
        self.cpu_parallel_context = None
        self.vad_process_timeout = vad_process_timeout
        self.vad_cpu_cores = vad_cpu_cores

        self.vad_model = None
        self.inputAudioMaxDuration = input_audio_max_duration
        self.deleteUploadedFiles = delete_uploaded_files

    def set_parallel_devices(self, vad_parallel_devices: str):
        self.parallel_device_list = [ device.strip() for device in vad_parallel_devices.split(",") ] if vad_parallel_devices else None

    def set_auto_parallel(self, auto_parallel: bool):
        if auto_parallel:
            if torch.cuda.is_available():
                self.parallel_device_list = [ str(gpu_id) for gpu_id in range(torch.cuda.device_count())]

            self.vad_cpu_cores = min(os.cpu_count(), MAX_AUTO_CPU_CORES)
            print("[Auto parallel] Using GPU devices " + str(self.parallel_device_list) + " and " + str(self.vad_cpu_cores) + " CPU cores for VAD/transcription.")

    def transcribe_webui(self, modelName, languageName, urlData, uploadFile, microphoneData, task, vad, vadMergeWindow, vadMaxMergeSize, vadPadding, vadPromptWindow):
        try:
            source, sourceName = self.__get_source(urlData, uploadFile, microphoneData)
            
            try:
                selectedLanguage = languageName.lower() if len(languageName) > 0 else None
                selectedModel = modelName if modelName is not None else "base"

                model = WhisperContainer(model_name=selectedModel, cache=self.model_cache)

                # Execute whisper
                result = self.transcribe_file(model, source, selectedLanguage, task, vad, vadMergeWindow, vadMaxMergeSize, vadPadding, vadPromptWindow)

                # Write result
                downloadDirectory = tempfile.mkdtemp()
                
                filePrefix = slugify(sourceName, allow_unicode=True)
                download, text, vtt = self.write_result(result, filePrefix, downloadDirectory)

                return download, text, vtt

            finally:
                # Cleanup source
                if self.deleteUploadedFiles:
                    print("Deleting source file " + source)
                    os.remove(source)
        
        except ExceededMaximumDuration as e:
            return [], ("[ERROR]: Maximum remote video length is " + str(e.maxDuration) + "s, file was " + str(e.videoDuration) + "s"), "[ERROR]"

    def transcribe_file(self, model: WhisperContainer, audio_path: str, language: str, task: str = None, vad: str = None, 
                        vadMergeWindow: float = 5, vadMaxMergeSize: float = 150, vadPadding: float = 1, vadPromptWindow: float = 1, **decodeOptions: dict):
        
        initial_prompt = decodeOptions.pop('initial_prompt', None)

        if ('task' in decodeOptions):
            task = decodeOptions.pop('task')

        # Callable for processing an audio file
        whisperCallable = model.create_callback(language, task, initial_prompt, **decodeOptions)

        # The results
        if (vad == 'silero-vad'):
            # Silero VAD where non-speech gaps are transcribed
            process_gaps = self._create_silero_config(NonSpeechStrategy.CREATE_SEGMENT, vadMergeWindow, vadMaxMergeSize, vadPadding, vadPromptWindow)
            result = self.process_vad(audio_path, whisperCallable, self.vad_model, process_gaps)
        elif (vad == 'silero-vad-skip-gaps'):
            # Silero VAD where non-speech gaps are simply ignored
            skip_gaps = self._create_silero_config(NonSpeechStrategy.SKIP, vadMergeWindow, vadMaxMergeSize, vadPadding, vadPromptWindow)
            result = self.process_vad(audio_path, whisperCallable, self.vad_model, skip_gaps)
        elif (vad == 'silero-vad-expand-into-gaps'):
            # Use Silero VAD where speech-segments are expanded into non-speech gaps
            expand_gaps = self._create_silero_config(NonSpeechStrategy.EXPAND_SEGMENT, vadMergeWindow, vadMaxMergeSize, vadPadding, vadPromptWindow)
            result = self.process_vad(audio_path, whisperCallable, self.vad_model, expand_gaps)
        elif (vad == 'periodic-vad'):
            # Very simple VAD - mark every 5 minutes as speech. This makes it less likely that Whisper enters an infinite loop, but
            # it may create a break in the middle of a sentence, causing some artifacts.
            periodic_vad = VadPeriodicTranscription()
            period_config = PeriodicTranscriptionConfig(periodic_duration=vadMaxMergeSize, max_prompt_window=vadPromptWindow)
            result = self.process_vad(audio_path, whisperCallable, periodic_vad, period_config)

        else:
            if (self._has_parallel_devices()):
                # Use a simple period transcription instead, as we need to use the parallel context
                periodic_vad = VadPeriodicTranscription()
                period_config = PeriodicTranscriptionConfig(periodic_duration=math.inf, max_prompt_window=1)

                result = self.process_vad(audio_path, whisperCallable, periodic_vad, period_config)
            else:
                # Default VAD
                result = whisperCallable(audio_path, 0, None, None)

        return result

    def process_vad(self, audio_path, whisperCallable, vadModel: AbstractTranscription, vadConfig: TranscriptionConfig):
        if (not self._has_parallel_devices()):
            # No parallel devices, so just run the VAD and Whisper in sequence
            return vadModel.transcribe(audio_path, whisperCallable, vadConfig)

        gpu_devices = self.parallel_device_list

        if (gpu_devices is None or len(gpu_devices) == 0):
            # No GPU devices specified, pass the current environment variable to the first GPU process. This may be NULL.
            gpu_devices = [os.environ.get("CUDA_VISIBLE_DEVICES", None)]

        # Create parallel context if needed
        if (self.gpu_parallel_context is None):
            # Create a context wih processes and automatically clear the pool after 1 hour of inactivity
            self.gpu_parallel_context = ParallelContext(num_processes=len(gpu_devices), auto_cleanup_timeout_seconds=self.vad_process_timeout)
        # We also need a CPU context for the VAD
        if (self.cpu_parallel_context is None):
            self.cpu_parallel_context = ParallelContext(num_processes=self.vad_cpu_cores, auto_cleanup_timeout_seconds=self.vad_process_timeout)

        parallel_vad = ParallelTranscription()
        return parallel_vad.transcribe_parallel(transcription=vadModel, audio=audio_path, whisperCallable=whisperCallable,  
                                                config=vadConfig, cpu_device_count=self.vad_cpu_cores, gpu_devices=gpu_devices, 
                                                cpu_parallel_context=self.cpu_parallel_context, gpu_parallel_context=self.gpu_parallel_context) 

    def _has_parallel_devices(self):
        return (self.parallel_device_list is not None and len(self.parallel_device_list) > 0) or self.vad_cpu_cores > 1

    def _concat_prompt(self, prompt1, prompt2):
        if (prompt1 is None):
            return prompt2
        elif (prompt2 is None):
            return prompt1
        else:
            return prompt1 + " " + prompt2

    def _create_silero_config(self, non_speech_strategy: NonSpeechStrategy, vadMergeWindow: float = 5, vadMaxMergeSize: float = 150, vadPadding: float = 1, vadPromptWindow: float = 1):
        # Use Silero VAD 
        if (self.vad_model is None):
            self.vad_model = VadSileroTranscription()

        config = TranscriptionConfig(non_speech_strategy = non_speech_strategy, 
                max_silent_period=vadMergeWindow, max_merge_size=vadMaxMergeSize, 
                segment_padding_left=vadPadding, segment_padding_right=vadPadding, 
                max_prompt_window=vadPromptWindow)

        return config

    def write_result(self, result: dict, source_name: str, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        text = result["text"]
        language = result["language"]
        languageMaxLineWidth = self.__get_max_line_width(language)

        print("Max line width " + str(languageMaxLineWidth))
        vtt = self.__get_subs(result["segments"], "vtt", languageMaxLineWidth)
        srt = self.__get_subs(result["segments"], "srt", languageMaxLineWidth)

        output_files = []
        output_files.append(self.__create_file(srt, output_dir, source_name + "-subs.srt"));
        output_files.append(self.__create_file(vtt, output_dir, source_name + "-subs.vtt"));
        output_files.append(self.__create_file(text, output_dir, source_name + "-transcript.txt"));

        return output_files, text, vtt

    def clear_cache(self):
        self.model_cache.clear()
        self.vad_model = None

    def __get_source(self, urlData, uploadFile, microphoneData):
        if urlData:
            # Download from YouTube
            source = download_url(urlData, self.inputAudioMaxDuration)[0]
        else:
            # File input
            source = uploadFile if uploadFile is not None else microphoneData

            if self.inputAudioMaxDuration > 0:
                # Calculate audio length
                audioDuration = ffmpeg.probe(source)["format"]["duration"]
            
                if float(audioDuration) > self.inputAudioMaxDuration:
                    raise ExceededMaximumDuration(videoDuration=audioDuration, maxDuration=self.inputAudioMaxDuration, message="Video is too long")

        file_path = pathlib.Path(source)
        sourceName = file_path.stem[:MAX_FILE_PREFIX_LENGTH] + file_path.suffix

        return source, sourceName

    def __get_max_line_width(self, language: str) -> int:
        if (language and language.lower() in ["japanese", "ja", "chinese", "zh"]):
            # Chinese characters and kana are wider, so limit line length to 40 characters
            return 40
        else:
            # TODO: Add more languages
            # 80 latin characters should fit on a 1080p/720p screen
            return 80

    def __get_subs(self, segments: Iterator[dict], format: str, maxLineWidth: int) -> str:
        segmentStream = StringIO()

        if format == 'vtt':
            write_vtt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
        elif format == 'srt':
            write_srt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
        else:
            raise Exception("Unknown format " + format)

        segmentStream.seek(0)
        return segmentStream.read()

    def __create_file(self, text: str, directory: str, fileName: str) -> str:
        # Write the text to a file
        with open(os.path.join(directory, fileName), 'w+', encoding="utf-8") as file:
            file.write(text)

        return file.name

    def close(self):
        self.clear_cache()

        if (self.gpu_parallel_context is not None):
            self.gpu_parallel_context.close()
        if (self.cpu_parallel_context is not None):
            self.cpu_parallel_context.close()


def create_ui(input_audio_max_duration, share=False, server_name: str = None, server_port: int = 7860, 
              default_model_name: str = "medium", default_vad: str = None, vad_parallel_devices: str = None, vad_process_timeout: float = None, vad_cpu_cores: int = 1, auto_parallel: bool = False):
    ui = WhisperTranscriber(input_audio_max_duration, vad_process_timeout, vad_cpu_cores)

    # Specify a list of devices to use for parallel processing
    ui.set_parallel_devices(vad_parallel_devices)
    ui.set_auto_parallel(auto_parallel)

    ui_description = "Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse " 
    ui_description += " audio and is also a multi-task model that can perform multilingual speech recognition "
    ui_description += " as well as speech translation and language identification. "

    ui_description += "\n\n\n\nFor longer audio files (>10 minutes) not in English, it is recommended that you select Silero VAD (Voice Activity Detector) in the VAD option."

    if input_audio_max_duration > 0:
        ui_description += "\n\n" + "Max audio file length: " + str(input_audio_max_duration) + " s"

    ui_article = "Read the [documentation here](https://huggingface.co/spaces/aadnk/whisper-webui/blob/main/docs/options.md)"

    demo = gr.Interface(fn=ui.transcribe_webui, description=ui_description, article=ui_article, inputs=[
        gr.Dropdown(choices=["tiny", "base", "small", "medium", "large"], value=default_model_name, label="Model"),
        gr.Dropdown(choices=sorted(LANGUAGES), label="Language"),
        gr.Text(label="URL (YouTube, etc.)"),
        gr.Audio(source="upload", type="filepath", label="Upload Audio"), 
        gr.Audio(source="microphone", type="filepath", label="Microphone Input"),
        gr.Dropdown(choices=["transcribe", "translate"], label="Task"),
        gr.Dropdown(choices=["none", "silero-vad", "silero-vad-skip-gaps", "silero-vad-expand-into-gaps", "periodic-vad"], value=default_vad, label="VAD"),
        gr.Number(label="VAD - Merge Window (s)", precision=0, value=5),
        gr.Number(label="VAD - Max Merge Size (s)", precision=0, value=30),
        gr.Number(label="VAD - Padding (s)", precision=None, value=1),
        gr.Number(label="VAD - Prompt Window (s)", precision=None, value=3)
    ], outputs=[
        gr.File(label="Download"),
        gr.Text(label="Transcription"), 
        gr.Text(label="Segments")
    ])

    demo.launch(share=share, server_name=server_name, server_port=server_port)
    
    # Clean up
    ui.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_audio_max_duration", type=int, default=DEFAULT_INPUT_AUDIO_MAX_DURATION, help="Maximum audio file length in seconds, or -1 for no limit.")
    parser.add_argument("--share", type=bool, default=False, help="True to share the app on HuggingFace.")
    parser.add_argument("--server_name", type=str, default=None, help="The host or IP to bind to. If None, bind to localhost.")
    parser.add_argument("--server_port", type=int, default=7860, help="The port to bind to.")
    parser.add_argument("--default_model_name", type=str, default="medium", help="The default model name.")
    parser.add_argument("--default_vad", type=str, default="silero-vad", help="The default VAD.")
    parser.add_argument("--vad_parallel_devices", type=str, default="", help="A commma delimited list of CUDA devices to use for parallel processing. If None, disable parallel processing.")
    parser.add_argument("--vad_cpu_cores", type=int, default=1, help="The number of CPU cores to use for VAD pre-processing.")
    parser.add_argument("--vad_process_timeout", type=float, default="1800", help="The number of seconds before inactivate processes are terminated. Use 0 to close processes immediately, or None for no timeout.")
    parser.add_argument("--auto_parallel", type=bool, default=False, help="True to use all available GPUs and CPU cores for processing. Use vad_cpu_cores/vad_parallel_devices to specify the number of CPU cores/GPUs to use.")

    args = parser.parse_args().__dict__
    create_ui(**args)