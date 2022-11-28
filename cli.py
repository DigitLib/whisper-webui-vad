import argparse
import os
import pathlib
from urllib.parse import urlparse
import warnings
import numpy as np

import whisper

import torch
from app import LANGUAGES, WhisperTranscriber
from src.download import download_url

from src.utils import optional_float, optional_int, str2bool
from src.whisperContainer import WhisperContainer


def cli():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large"], help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES), help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--vad", type=str, default="none", choices=["none", "silero-vad", "silero-vad-skip-gaps", "silero-vad-expand-into-gaps", "periodic-vad"], help="The voice activity detection algorithm to use")
    parser.add_argument("--vad_merge_window", type=optional_float, default=5, help="The window size (in seconds) to merge voice segments")
    parser.add_argument("--vad_max_merge_size", type=optional_float, default=30, help="The maximum size (in seconds) of a voice segment")
    parser.add_argument("--vad_padding", type=optional_float, default=1, help="The padding (in seconds) to add to each voice segment")
    parser.add_argument("--vad_prompt_window", type=optional_float, default=3, help="The window size of the prompt to pass to Whisper")
    parser.add_argument("--vad_cpu_cores", type=int, default=1, help="The number of CPU cores to use for VAD pre-processing.")
    parser.add_argument("--vad_parallel_devices", type=str, default="", help="A commma delimited list of CUDA devices to use for parallel processing. If None, disable parallel processing.")
    parser.add_argument("--auto_parallel", type=bool, default=False, help="True to use all available GPUs and CPU cores for processing. Use vad_cpu_cores/vad_parallel_devices to specify the number of CPU cores/GPUs to use.")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple lengt normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    model_dir: str = args.pop("model_dir")
    output_dir: str = args.pop("output_dir")
    device: str = args.pop("device")
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        warnings.warn(f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead.")
        args["language"] = "en"

    temperature = args.pop("temperature")
    temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    vad = args.pop("vad")
    vad_merge_window = args.pop("vad_merge_window")
    vad_max_merge_size = args.pop("vad_max_merge_size")
    vad_padding = args.pop("vad_padding")
    vad_prompt_window = args.pop("vad_prompt_window")
    vad_cpu_cores = args.pop("vad_cpu_cores")
    auto_parallel = args.pop("auto_parallel")

    model = WhisperContainer(model_name, device=device, download_root=model_dir)
    transcriber = WhisperTranscriber(delete_uploaded_files=False, vad_cpu_cores=vad_cpu_cores)
    transcriber.set_parallel_devices(args.pop("vad_parallel_devices"))
    transcriber.set_auto_parallel(auto_parallel)

    if (transcriber._has_parallel_devices()):
        print("Using parallel devices:", transcriber.parallel_device_list)

    for audio_path in args.pop("audio"):
        sources = []

        # Detect URL and download the audio
        if (uri_validator(audio_path)):
            # Download from YouTube/URL directly
            for source_path in  download_url(audio_path, maxDuration=-1, destinationDirectory=output_dir, playlistItems=None):
                source_name = os.path.basename(source_path)
                sources.append({ "path": source_path, "name": source_name })
        else:
            sources.append({ "path": audio_path, "name": os.path.basename(audio_path) })

        for source in sources:
            source_path = source["path"]
            source_name = source["name"]

            result = transcriber.transcribe_file(model, source_path, temperature=temperature, 
                                                vad=vad, vadMergeWindow=vad_merge_window, vadMaxMergeSize=vad_max_merge_size, 
                                                vadPadding=vad_padding, vadPromptWindow=vad_prompt_window, **args)
            
            transcriber.write_result(result, source_name, output_dir)

    transcriber.close()

def uri_validator(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False

if __name__ == '__main__':
    cli()