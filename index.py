#!/usr/bin/env python3
# transcribe_whisper.py (openai-whisper backend, no vad_filter)

import argparse
import os
import sys
import json
from datetime import datetime

try:
    import whisper
except ImportError:
    print("Error: The 'openai-whisper' package is not installed. Run: pip install openai-whisper", file=sys.stderr)
    sys.exit(1)

def save_outputs(base_out, result, formats, with_words):
    os.makedirs(os.path.dirname(base_out), exist_ok=True)

    if "txt" in formats:
        with open(base_out + ".txt", "w", encoding="utf-8") as f:
            f.write(result["text"].strip() + "\n")

    def secs_to_ts(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    if "srt" in formats:
        with open(base_out + ".srt", "w", encoding="utf-8") as f:
            for i, seg in enumerate(result["segments"], 1):
                start = secs_to_ts(seg["start"])
                end = secs_to_ts(seg["end"])
                lines = [seg["text"].strip()]
                if with_words and "words" in seg:
                    word_line = " ".join([w["word"] for w in seg["words"]])
                    if word_line:
                        lines.append(word_line.strip())
                f.write(f"{i}\n{start} --> {end}\n" + "\n".join(lines) + "\n\n")

    if "vtt" in formats:
        with open(base_out + ".vtt", "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for seg in result["segments"]:
                start = f"{int(seg['start']//3600):02d}:{int((seg['start']%3600)//60):02d}:{seg['start']%60:06.3f}"
                end = f"{int(seg['end']//3600):02d}:{int((seg['end']%3600)//60):02d}:{seg['end']%60:06.3f}"
                f.write(f"{start} --> {end}\n{seg['text'].strip()}\n\n")

    if "json" in formats:
        with open(base_out + ".json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with OpenAI Whisper (local).")
    parser.add_argument("inputs", nargs="+", help="Input audio file(s)")
    parser.add_argument("--model", default="large-v3", help="tiny/base/small/medium/large-v3 (default: large-v3)")
    parser.add_argument("--language", default="en", help="e.g., en, es, fr (default: en)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size (default: 5)")
    parser.add_argument("--best_of", type=int, default=5, help="Candidates when sampling (default: 5)")
    parser.add_argument("--initial_prompt", default="", help="Initial prompt to prime decoding.")
    parser.add_argument("--condition_on_previous_text", type=str, default="false",
                        help="true/false (default: false)")
    parser.add_argument("--word_timestamps", action="store_true", help="Per-word timestamps.")
    parser.add_argument("--device", default=None, help="cuda/cpu (auto if not set)")
    parser.add_argument("--output_dir", default="transcripts", help="Output dir (default: transcripts)")
    parser.add_argument("--formats", default="txt,srt", help="txt,srt,vtt,json (default: txt,srt)")

    # Extra accuracy-related thresholds supported by openai-whisper:
    parser.add_argument("--no_speech_threshold", type=float, default=0.6, help="Default 0.6")
    parser.add_argument("--logprob_threshold", type=float, default=-1.0, help="Default -1.0 (no cutoff)")
    parser.add_argument("--compression_ratio_threshold", type=float, default=2.4, help="Default 2.4")
    parser.add_argument("--patience", type=float, default=1.0, help="Beam search patience (default: 1.0)")
    parser.add_argument("--length_penalty", type=float, default=None, help="Optional length penalty")

    args = parser.parse_args()
    formats = {fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()}
    cond_prev = str(args.condition_on_previous_text).lower() in ("1", "true", "yes", "y")

    print(f"[{datetime.now().isoformat(timespec='seconds')}] Loading model: {args.model} ...")
    model = whisper.load_model(args.model, device=args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    for path in args.inputs:
        if not os.path.isfile(path):
            print(f"Skipping (not found): {path}", file=sys.stderr)
            continue

        print(f"[{datetime.now().isoformat(timespec='seconds')}] Transcribing: {path}")
        result = model.transcribe(
            path,
            language=args.language,
            temperature=args.temperature,
            beam_size=args.beam_size,
            best_of=args.best_of,
            initial_prompt=args.initial_prompt or None,
            condition_on_previous_text=cond_prev,
            word_timestamps=args.word_timestamps,
            patience=args.patience,
            length_penalty=args.length_penalty,
            compression_ratio_threshold=args.compression_ratio_threshold,
            logprob_threshold=args.logprob_threshold,
            no_speech_threshold=args.no_speech_threshold,
        )

        base = os.path.splitext(os.path.basename(path))[0]
        out_base = os.path.join(args.output_dir, base)
        save_outputs(out_base, result, formats, args.word_timestamps)
        print(f"   → Saved to: {out_base}.{{{','.join(sorted(formats))}}}")

if __name__ == "__main__":
    main()
