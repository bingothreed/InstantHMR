"""Play a clip's own audio track alongside the demo.

Rerun 0.31 has no audio archetype, so the sound cannot go into the recording: the demo extracts the
audio track to a temporary WAV and hands it to the system player while it runs.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import wave
from typing import Optional

SAMPLE_RATE = 16000


def _audio_stream(path: str):
    import av

    container = av.open(path)
    stream = next((s for s in container.streams if s.type == "audio"), None)
    return container, stream


def has_audio(path: str) -> bool:
    """True when the file carries at least one audio stream."""
    try:
        container, stream = _audio_stream(path)
        ok = stream is not None
        container.close()
        return ok
    except Exception:
        return False


def extract_to_wav(path: str, wav_path: str) -> Optional[float]:
    """Decode the first audio stream to 16 kHz mono s16 WAV; returns its duration in seconds."""
    import av

    container, stream = _audio_stream(path)
    if stream is None:
        container.close()
        return None
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    pcm = bytearray()
    try:
        for frame in container.decode(stream):
            for out in resampler.resample(frame) or ():
                pcm += out.to_ndarray().tobytes()
        for out in resampler.resample(None) or ():
            pcm += out.to_ndarray().tobytes()
    finally:
        container.close()
    if not pcm:
        return None
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(pcm))
    return len(pcm) / 2 / SAMPLE_RATE


class _Player:
    """Whatever started the sound; `stop()` silences it again."""

    def __init__(self, kind: str, handle=None, wav_path: Optional[str] = None) -> None:
        self.kind, self.handle, self.wav_path = kind, handle, wav_path

    def stop(self) -> None:
        try:
            if self.kind == "winsound":
                import winsound

                winsound.PlaySound(None, winsound.SND_PURGE)
            elif self.handle is not None:
                self.handle.terminate()
                self.handle.wait(timeout=2)
        except Exception:
            pass
        if self.wav_path and os.path.exists(self.wav_path):
            try:
                os.remove(self.wav_path)
            except OSError:
                pass


def play_video_audio(path: str) -> Optional[_Player]:
    """Start the clip's audio on the system player, or print why it could not and return None."""
    if not has_audio(path):
        print("[audio] no audio track in this file, nothing to play")
        return None
    wav = os.path.join(tempfile.gettempdir(), f"instanthmr_{os.getpid()}.wav")
    try:
        duration = extract_to_wav(path, wav)
    except Exception as e:
        print(f"[audio] could not decode the audio track: {e}")
        return None
    if duration is None:
        print("[audio] the audio track decoded to zero samples, nothing to play")
        return None
    print(f"[audio] playing {duration:.1f} s from the clip (system speakers, not inside the Rerun viewer)")

    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
        return _Player("winsound", wav_path=wav)
    for exe in ("afplay", "aplay", "paplay"):
        if shutil.which(exe):
            handle = subprocess.Popen([exe, wav], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return _Player("proc", handle=handle, wav_path=wav)
    print("[audio] no system audio player found (afplay / aplay / paplay)")
    os.remove(wav)
    return None
