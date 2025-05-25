#!/usr/bin/env python3
"""
Generate a Hindi speech-like audio sample.
"""

import argparse
import os
import numpy as np
import soundfile as sf
from scipy import signal
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define Hindi phoneme parameters
HINDI_VOWELS = ["a", "aa", "i", "ii", "u", "uu", "e", "ai", "o", "au"]
HINDI_CONSONANTS = ["k", "kh", "g", "gh", "ch", "chh", "j", "jh", "t", "th", "d", "dh", "n", 
                    "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "sh", "s", "h"]

def generate_formant(frequency, sample_rate, duration, formant_hz=None):
    """
    Generate a formant (vowel-like sound) with the given frequency.
    
    Args:
        frequency: Base frequency
        sample_rate: Sample rate
        duration: Duration in seconds
        formant_hz: List of formant frequencies
        
    Returns:
        numpy.ndarray: Audio data
    """
    if formant_hz is None:
        # Default formants for a generic vowel
        formant_hz = [500, 1500, 2500]
    
    t = np.arange(0, duration, 1/sample_rate)
    signal_data = np.zeros_like(t)
    
    # Generate base frequency
    signal_data += 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Add formants
    for i, formant in enumerate(formant_hz):
        amplitude = 0.5 / (i + 1)  # Decreasing amplitude for higher formants
        signal_data += amplitude * np.sin(2 * np.pi * formant * t)
    
    # Envelope to smoothly fade in and out
    envelope = np.ones_like(t)
    fade_samples = int(0.02 * sample_rate)  # 20ms fade
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    return signal_data * envelope

def generate_consonant(sample_rate, duration):
    """
    Generate a consonant-like sound.
    
    Args:
        sample_rate: Sample rate
        duration: Duration in seconds
        
    Returns:
        numpy.ndarray: Audio data
    """
    t = np.arange(0, duration, 1/sample_rate)
    noise = np.random.normal(0, 0.5, len(t))
    
    # Filter to make it sound more like speech
    b, a = signal.butter(4, [500/(sample_rate/2), 4000/(sample_rate/2)], btype='band')
    filtered_noise = signal.lfilter(b, a, noise)
    
    # Envelope to shape the sound
    envelope = np.ones_like(t)
    fade_samples = int(0.01 * sample_rate)  # 10ms fade
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    return filtered_noise * envelope

def generate_speech_like_audio(duration, sample_rate):
    """
    Generate Hindi speech-like audio.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
        
    Returns:
        numpy.ndarray: Audio data
    """
    # Create empty audio array
    audio = np.array([])
    
    # Generate a sequence of syllables until we reach the desired duration
    current_duration = 0
    
    while current_duration < duration:
        # Choose a consonant and vowel
        if random.random() < 0.9:  # 90% chance for CV syllable
            # Generate consonant (shorter)
            consonant_duration = random.uniform(0.05, 0.1)
            consonant = generate_consonant(sample_rate, consonant_duration)
            
            # Generate vowel
            vowel_duration = random.uniform(0.1, 0.3)
            frequency = random.uniform(80, 180)  # Typical fundamental frequency range
            
            # Random formants for different vowel qualities
            f1 = random.uniform(300, 800)
            f2 = random.uniform(1000, 2500)
            f3 = random.uniform(2200, 3500)
            
            vowel = generate_formant(frequency, sample_rate, vowel_duration, [f1, f2, f3])
            
            # Combine consonant and vowel
            syllable = np.concatenate([consonant, vowel])
        else:
            # Just a vowel occasionally
            vowel_duration = random.uniform(0.2, 0.4)
            frequency = random.uniform(80, 180)
            
            f1 = random.uniform(300, 800)
            f2 = random.uniform(1000, 2500)
            f3 = random.uniform(2200, 3500)
            
            syllable = generate_formant(frequency, sample_rate, vowel_duration, [f1, f2, f3])
        
        # Add a short pause occasionally
        if random.random() < 0.3:
            pause_duration = random.uniform(0.05, 0.2)
            pause = np.zeros(int(pause_duration * sample_rate))
            syllable = np.concatenate([syllable, pause])
        
        # Add to the audio array
        audio = np.concatenate([audio, syllable]) if len(audio) > 0 else syllable
        current_duration = len(audio) / sample_rate
    
    # Trim to the exact duration
    audio = audio[:int(duration * sample_rate)]
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio

def main():
    parser = argparse.ArgumentParser(description="Generate a Hindi speech-like audio sample")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of the audio in seconds")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate of the audio")
    parser.add_argument("--output", type=str, default="hindi_speech_sample.wav", help="Output file path")
    
    args = parser.parse_args()
    
    # Generate speech-like audio
    logger.info(f"Generating Hindi speech-like audio with duration {args.duration}s and sample rate {args.sample_rate}Hz")
    audio = generate_speech_like_audio(args.duration, args.sample_rate)
    
    # Save to file
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    sf.write(args.output, audio, args.sample_rate)
    logger.info(f"Audio sample saved to {args.output}")

if __name__ == "__main__":
    main() 