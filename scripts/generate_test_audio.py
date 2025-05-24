#!/usr/bin/env python3
"""
Script to generate a test audio file for testing the ASR API.
This creates a simple sine wave audio file of specified duration.
"""

import argparse
import numpy as np
import wave
import struct

def generate_sine_wave(frequency, duration, sample_rate):
    """
    Generate a sine wave.
    
    Args:
        frequency: Frequency of the sine wave in Hz
        duration: Duration of the audio in seconds
        sample_rate: Sample rate in Hz
    
    Returns:
        numpy.ndarray: The generated sine wave
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    sine_wave = np.sin(2 * np.pi * frequency * t)
    
    # Normalize to -1 to 1
    sine_wave = sine_wave / np.max(np.abs(sine_wave))
    
    return sine_wave

def save_wav(audio, filename, sample_rate):
    """
    Save audio data as a WAV file.
    
    Args:
        audio: Audio data as a numpy array
        filename: Output filename
        sample_rate: Sample rate in Hz
    """
    # Scale to int16 range
    audio = audio * 32767
    audio = audio.astype(np.int16)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes (16 bits)
        wav_file.setframerate(sample_rate)
        
        # Write frames
        for sample in audio:
            wav_file.writeframes(struct.pack('h', sample))

def main():
    parser = argparse.ArgumentParser(description="Generate a test audio file")
    parser.add_argument("--frequency", type=float, default=440.0, 
                        help="Frequency of the sine wave in Hz")
    parser.add_argument("--duration", type=float, default=5.0, 
                        help="Duration of the audio in seconds")
    parser.add_argument("--sample_rate", type=int, default=16000, 
                        help="Sample rate in Hz")
    parser.add_argument("--output", type=str, default="test_audio.wav", 
                        help="Output filename")
    
    args = parser.parse_args()
    
    # Generate sine wave
    audio = generate_sine_wave(args.frequency, args.duration, args.sample_rate)
    
    # Save as WAV file
    save_wav(audio, args.output, args.sample_rate)
    
    print(f"Generated test audio file: {args.output}")
    print(f"Duration: {args.duration} seconds")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Frequency: {args.frequency} Hz")

if __name__ == "__main__":
    main() 