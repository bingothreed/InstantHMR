"""A tiny always-on-top panel for the clip audio: switch the sound on or off while the demo runs.

Rerun 0.31 has no audio support, so the toggle cannot live in the viewer itself. This panel is pumped
from the demo's frame loop (`AudioControl.poll()`), which keeps it responsive without a second thread.
"""
from __future__ import annotations

from typing import Callable, Optional

from instanthmr.audio import AudioPlayer


class AudioControl:
    """Sound on/off, strict A/V sync on/off and a playhead readout for `player`."""

    def __init__(self, player: AudioPlayer, clock: Callable[[], float]) -> None:
        import tkinter as tk

        self._tk = tk
        self.player = player
        self._clock = clock
        self.closed = False

        self.root = tk.Tk()
        self.root.title("InstantHMR audio")
        self.root.resizable(False, False)
        try:
            self.root.attributes("-topmost", True)
        except Exception:
            pass
        body = tk.Frame(self.root, padx=10, pady=8)
        body.pack()

        self.sound = tk.BooleanVar(value=player.enabled)
        tk.Checkbutton(body, text="播放片段原声（与画面同步）", variable=self.sound,
                       command=self._apply_sound).grid(row=0, column=0, columnspan=2, sticky="w")
        self.sync = tk.BooleanVar(value=True)
        tk.Checkbutton(body, text="按片段时长节流（严格 A/V 对齐）", variable=self.sync).grid(
            row=1, column=0, columnspan=2, sticky="w")

        tk.Label(body, text="音量").grid(row=2, column=0, sticky="w")
        self.volume = tk.DoubleVar(value=player.volume * 100.0)
        tk.Scale(body, from_=0, to=100, orient="horizontal", length=170, variable=self.volume,
                 showvalue=False, command=lambda _v: self._apply_volume()).grid(row=2, column=1, sticky="we")

        self.status = tk.Label(body, text="", anchor="w", width=36, font=("Consolas", 9))
        self.status.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._refresh_status()
        self.root.update()

    # -- controls --------------------------------------------------------------------------------
    def _apply_sound(self) -> None:
        on = bool(self.sound.get())
        self.player.set_enabled(on)
        if on:
            self.player.seek(self._clock())
            print(f"[audio] sound on, realigned to {self.player.cursor:.1f} s")
        else:
            print("[audio] sound off")

    def _apply_volume(self) -> None:
        self.player.volume = float(self.volume.get()) / 100.0

    # -- per frame -------------------------------------------------------------------------------
    def poll(self) -> None:
        """Pump the window's events and refresh the readout; cheap enough to call every frame."""
        if self.closed:
            return
        try:
            self.root.update()
        except Exception:
            self.closed = True
            return
        self._refresh_status()

    def _refresh_status(self) -> None:
        state = "播放中" if self.player.enabled else "已静音"
        if self.player.enabled and self.player.at_end:
            state = "播放完毕"
        self.status.configure(
            text=f"{self.player.cursor:5.1f}s / {self.player.duration:5.1f}s   {state}")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.player.stop()
        try:
            self.root.destroy()
        except Exception:
            pass
