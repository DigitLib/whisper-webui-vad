# Options
To transcribe or translate an audio file, you can either copy an URL from a website (all [websites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md) 
supported by YT-DLP will work, including YouTube). Otherwise, upload an audio file (choose "All Files (*.*)" 
in the file selector to select any file type, including video files) or use the microphone.

For longer audio files (>10 minutes), it is recommended that you select Silero VAD (Voice Activity Detector) in the VAD option.

## Model
Select the model that Whisper will use to transcribe the audio:

| Size   | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|--------|------------|--------------------|--------------------|---------------|----------------|
| tiny   | 39 M       | tiny.en            | tiny               | ~1 GB         | ~32x           |
| base   | 74 M       | base.en            | base               | ~1 GB         | ~16x           |
| small  | 244 M      | small.en           | small              | ~2 GB         | ~6x            |
| medium | 769 M      | medium.en          | medium             | ~5 GB         | ~2x            |
| large  | 1550 M     | N/A                | large              | ~10 GB        | 1x             |

## Language

Select the language, or leave it empty for Whisper to automatically detect it. 

Note that if the selected language and the language in the audio differs, Whisper may start to translate the audio to the selected 
language. For instance, if the audio is in English but you select Japaneese, the model may translate the audio to Japanese.

## Inputs
The options "URL (YouTube, etc.)", "Upload Audio" or "Micriphone Input" allows you to send an audio input to the model.

Note that the UI will only process the first valid input - i.e. if you enter both an URL and upload an audio, it will only process 
the URL. 

## Task
Select the task - either "transcribe" to transcribe the audio to text, or "translate" to translate it to English.

## Vad
Using a VAD will improve the timing accuracy of each transcribed line, as well as prevent Whisper getting into an infinite
loop detecting the same sentence over and over again. The downside is that this may be at a cost to text accuracy, especially
with regards to unique words or names that appear in the audio. You can compensate for this by increasing the prompt window. 

Note that English is very well handled by Whisper, and it's less susceptible to issues surrounding bad timings and infinite loops. 
So you may only need to use a VAD for other languages, such as Japanese, or when the audio is very long.

* none
  * Run whisper on the entire audio input
* silero-vad
   * Use Silero VAD to detect sections that contain speech, and run Whisper on independently on each section. Whisper is also run 
     on the gaps between each speech section, by either expanding the section up to the max merge size, or running Whisper independently 
     on the non-speech section.
* silero-vad-expand-into-gaps
   * Use Silero VAD to detect sections that contain speech, and run Whisper on independently on each section. Each spech section will be expanded
     such that they cover any adjacent non-speech sections. For instance, if an audio file of one minute contains the speech sections 
     00:00 - 00:10 (A) and 00:30 - 00:40 (B), the first section (A) will be expanded to 00:00 - 00:30, and (B) will be expanded to 00:30 - 00:60.
* silero-vad-skip-gaps
   * As above, but sections that doesn't contain speech according to Silero will be skipped. This will be slightly faster, but 
     may cause dialogue to be skipped.
* periodic-vad
   * Create sections of speech every 'VAD - Max Merge Size' seconds. This is very fast and simple, but will potentially break 
     a sentence or word in two.

## VAD - Merge Window
If set, any adjacent speech sections that are at most this number of seconds apart will be automatically merged.

## VAD - Max Merge Size (s)
Disables merging of adjacent speech sections if they are this number of seconds long.

## VAD - Padding (s)
The number of seconds (floating point) to add to the beginning and end of each speech section. Setting this to a number
larger than zero ensures that Whisper is more likely to correctly transcribe a sentence in the beginning of 
a speech section. However, this also increases the probability of Whisper assigning the wrong timestamp 
to each transcribed line. The default value is 1 second.

## VAD - Prompt Window (s)
The text of a detected line will be included as a prompt to the next speech section, if the speech section starts at most this
number of seconds after the line has finished. For instance, if a line ends at 10:00, and the next speech section starts at
10:04, the line's text will be included if the prompt window is 4 seconds or more (10:04 - 10:00 = 4 seconds).

Note that detected lines in gaps between speech sections will not be included in the prompt 
(if silero-vad or silero-vad-expand-into-gaps) is used.