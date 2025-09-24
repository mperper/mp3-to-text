# Description
Super accurate mp3 to text, but with the large-v3 it takes awhile... to anyone who couldnt find music lyrics online but so direly want to know their musics lyrics lol

# Convert video to mp3
https://y2mate.nu/R2lu/

# Download whisper
pip3 install openai-whisper

# High-accuracy English vocals from bet.mp3
python3 index.py bet.mp3 \
  --model large-v3 --language en --temperature 0 --beam_size 5 --best_of 5 \
  --initial_prompt "Transcribe clean English vocals with proper punctuation." \
  --formats txt,srt,json


# Multiple files + per-word timestamps
python3 index.py vocals.wav chorus.wav --word_timestamps --output_dir out
