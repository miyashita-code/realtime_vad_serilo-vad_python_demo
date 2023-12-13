import argparse
import pyaudio
import torch
from queue import Queue
from sys import platform
import io
import wave
from datetime import datetime

# Constants for audio stream
FORMAT = pyaudio.paInt16
SAMPLE_RATE = 16000  # Samples per second
CHANNELS = 1
INPUT_DEVICE_INDEX = 2
CALLBACK_INTERVAL = 0.6  # 50 milliseconds
frames_per_buffer = int(SAMPLE_RATE * CALLBACK_INTERVAL)  # Number of frames per 50 ms

def main():
    global model, get_speech_timestamps, read_audio
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    print("Model loaded.\n")

    data_queue = Queue()
    look_for_audio_input()
    realtime_vad(data_queue)

def look_for_audio_input():
    pa = pyaudio.PyAudio()
    for i in range(pa.get_device_count()):
        print(pa.get_device_info_by_index(i))
        print()
    pa.terminate()

def callback(in_data, frame_count, time_info, status, data_queue):
    try:
        data_queue.put(in_data)
    except Exception as e:
        print(f"Callback error: {e}")
    finally:
        return (None, pyaudio.paContinue)

def realtime_vad(data_queue):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        rate=SAMPLE_RATE,
                        channels=CHANNELS,
                        input_device_index=INPUT_DEVICE_INDEX,
                        input=True,
                        frames_per_buffer=frames_per_buffer,
                        stream_callback=lambda in_data, frame_count, time_info, status: callback(in_data, frame_count, time_info, status, data_queue))

    stream.start_stream()

    while stream.is_active():
        if not data_queue.empty():
            data = data_queue.get()
            try:
                wav_data = io.BytesIO()
                with wave.open(wav_data, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(audio.get_sample_size(FORMAT))
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(data)
                wav_data.seek(0)

                wav = read_audio(wav_data, sampling_rate=SAMPLE_RATE)
                speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLE_RATE)

                if speech_timestamps:
                    print('Speech Detected!')
                else:
                    print('Silence Detected!')

            except Exception as e:
                print(f"Error processing audio: {e}")

    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == '__main__':
    main()
