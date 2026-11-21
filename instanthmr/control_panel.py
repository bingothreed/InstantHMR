"""One small always-on-top window with the switches the Rerun viewer cannot host.

Rerun 0.31 has no audio archetype and no widgets, so the viewer can neither play the clip's sound nor
report a click back to us. This panel is pumped from the demo's frame loop (`ControlPanel.poll()`): the
audio switches drive `AudioPlayer` directly, and the 3D switches re-send the blueprint through
`RerunVisualizer.update_layout()`, which the viewer applies live.
"""
from __future__ import annotations

from typing import Callable, Optional

from instanthmr.audio import AudioPlayer
from instanthmr.visualizer import RerunVisualizer


class ControlPanel:
    """Audio on/off + strict A/V sync + volume, and what the 3D view draws."""

    def __init__(self, player: Optional[AudioPlayer] = None, viewer: Optional[RerunVisualizer] = None,
                 clock: Callable[[], float] = lambda: 0.0, has_audio: bool = True) -> None:
        import tkinter as tk

        self._tk = tk
        self.player, self.viewer = player, viewer
        self._clock = clock
        self.closed = False

        self.root = tk.Tk()
        self.root.title("InstantHMR 控制面板")
        self.root.resizable(False, False)
        try:
            self.root.attributes("-topmost", True)
        except Exception:
            pass
        body = tk.Frame(self.root, padx=10, pady=8)
        body.pack()
        row = 0

        self.sound = tk.BooleanVar(value=bool(player and player.enabled))
        self.sync = tk.BooleanVar(value=True)
        self.volume = tk.DoubleVar(value=player.volume * 100.0 if player else 100.0)

        if player is None:
            tk.Label(body, text="片段原声：该文件没有音轨" if has_audio is False else "片段原声：未启用",
                     anchor="w").grid(row=row, column=0, columnspan=2, sticky="w")
            row += 1
        else:
            tk.Checkbutton(body, text="播放片段原声（与画面同步）", variable=self.sound,
                           command=self._apply_sound).grid(row=row, column=0, columnspan=2, sticky="w")
            row += 1
            tk.Checkbutton(body, text="按片段时长节流（严格 A/V 对齐）", variable=self.sync).grid(
                row=row, column=0, columnspan=2, sticky="w")
            row += 1
            tk.Label(body, text="音量").grid(row=row, column=0, sticky="w")
            tk.Scale(body, from_=0, to=100, orient="horizontal", length=170, variable=self.volume,
                     showvalue=False, command=lambda _v: self._apply_volume()).grid(row=row, column=1, sticky="we")
            row += 1

        if viewer is not None:
            tk.Frame(body, height=1, bg="#888").grid(row=row, column=0, columnspan=2, sticky="we", pady=4)
            row += 1
            self.camera_in_3d = tk.BooleanVar(value=bool(viewer.camera_in_3d))
            tk.Checkbutton(body, text="3D 视图：显示相机与视频图像平面", variable=self.camera_in_3d,
                           command=lambda: viewer.update_layout(camera_in_3d=self.camera_in_3d.get())).grid(
                row=row, column=0, columnspan=2, sticky="w")
            row += 1
            self.floor = tk.BooleanVar(value=bool(viewer.floor))
            tk.Checkbutton(body, text="3D 视图：显示地面网格", variable=self.floor,
                           command=lambda: viewer.update_layout(floor=self.floor.get())).grid(
                row=row, column=0, columnspan=2, sticky="w")
            row += 1
            self.viewer_grid = tk.BooleanVar(value=bool(viewer.viewer_grid))
            tk.Checkbutton(body, text="3D 视图：显示 Rerun 内置网格", variable=self.viewer_grid,
                           command=lambda: viewer.update_layout(viewer_grid=self.viewer_grid.get())).grid(
                row=row, column=0, columnspan=2, sticky="w")
            row += 1

        self.status = tk.Label(body, text="", anchor="w", width=36, font=("Consolas", 9))
        self.status.grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._refresh_status()
        self.root.update()

    # -- audio -----------------------------------------------------------------------------------
    def _apply_sound(self) -> None:
        if self.player is None:
            return
        on = bool(self.sound.get())
        self.player.set_enabled(on)
        if on:
            self.player.seek(self._clock())
            print(f"[audio] sound on, realigned to {self.player.cursor:.1f} s")
        else:
            print("[audio] sound off")

    def _apply_volume(self) -> None:
        if self.player is not None:
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
        if self.player is None:
            self.status.configure(text="无音轨，可用上面 3D 开关")
            return
        state = "播放中" if self.player.enabled else "已静音"
        if self.player.enabled and self.player.at_end:
            state = "播放完毕"
        self.status.configure(text=f"{self.player.cursor:5.1f}s / {self.player.duration:5.1f}s   {state}")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.player is not None:
            self.player.stop()
        try:
            self.root.destroy()
        except Exception:
            pass
