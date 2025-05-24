#!/usr/bin/env python3
"""
Script to test the ASR API locally.
"""

import argparse
import requests
import os
import json
from pathlib import Path

def test_api(api_url, audio_file):
    """
    Test the ASR API by sending an audio file and printing the transcription.
    
    Args:
        api_url: URL of the API
        audio_file: Path to the audio file
    """
    print(f"Testing API at {api_url} with audio file {audio_file}")
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return
    
    # Send request
    try:
        with open(audio_file, "rb") as f:
            files = {"audio_file": (os.path.basename(audio_file), f, "audio/wav")}
            response = requests.post(f"{api_url}/transcribe", files=files)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print("\nTranscription:")
            print("-" * 50)
            print(result["text"])
            print("-" * 50)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test the ASR API")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000", 
                        help="URL of the API")
    parser.add_argument("--audio_file", type=str, required=True, 
                        help="Path to the audio file")
    
    args = parser.parse_args()
    
    test_api(args.api_url, args.audio_file)

if __name__ == "__main__":
    main() 