# Real-Time Voice Activity Detection (VAD)

This Python script performs real-time voice activity detection using the Silero VAD model, with audio streaming via PyAudio.

## Dependencies
- Python 3.x
- PyAudio
- Torch
- Wave

## Setup
Install dependencies using:
```bash
pip install pyaudio torch wave

## Usage
1. Place the script in a directory.
2. Open a command line or terminal and navigate to the script's directory.
3. Run the script:
   ```bash
   python main.py

The script monitors audio in real-time and prints 'Speech Detected!' or 'Silence Detected!' based on voice activity.

## Configuration
Adjust parameters in the script for different audio settings:
- `SAMPLE_RATE`: Sample rate in Hz (default 16000).
- `CHANNELS`: Number of audio channels (default 1 for mono).
- `INPUT_DEVICE_INDEX`: Index of the audio input device.
- `CALLBACK_INTERVAL`: Callback frequency in seconds (default 0.6).

Note: Performance depends on system capabilities; a powerful hardware is recommended.

