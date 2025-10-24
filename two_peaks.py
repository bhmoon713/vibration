#!/usr/bin/env python3
"""
one_window.py — Live IMU plotter (single window) with top-2 FFT peak annotations
- Reads CSV from a serial port (first three numeric columns treated as X,Y,Z)
- Plots time-domain (3 axes) and combined FFT magnitude
- Annotates tallest and 2nd tallest FFT peaks in real time
- Start/Stop Save buttons to write CSV to disk

Example:
  python3 one_window.py --baud 115200 --port /dev/ttyACM0 --window 4.0 --fs 500

If you're unsure of the port, you can pass a glob:
  python3 one_window.py --port '/dev/ttyACM*'

CSV format assumed per line (flexible):
  [timestamp?,] ax, ay, az [, more...]
The script picks the first 3 numeric fields as X,Y,Z; timestamp is optional.
"""

import argparse
import glob
import os
import sys
import time
from collections import deque
from datetime import datetime
from threading import Lock, Thread

import numpy as np
import serial
import serial.tools.list_ports as list_ports
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# -------------------- Tunables --------------------
BAUD_DEFAULT = 115200
DEFAULT_PORT_GLOBS = ["/dev/ttyACM*", "/dev/ttyUSB*"]
WINDOW_SECONDS = 4.0            # seconds of data kept in the ring buffer
FS_HINT = 500.0                 # used for x-axis and FFT if no reliable timestamps (Hz)

FFT_MIN_HZ = 0.5                # ignore DC / very low frequencies
FFT_REL_MIN = 0.15              # show peaks >= this * global_max(|FFT|)
FFT_SUPPRESS_HALFSPAN = 3       # bins to suppress around tallest when searching 2nd peak

UI_UPDATE_HZ = 30               # UI refresh rate
SAVE_DIR = "./logs"             # where CSV is saved

# --------------------------------------------------


def _parse_numeric_triplet(raw_line: str):
    """
    Robustly parse a line for the first 3 numeric values.
    Returns (x,y,z) or None if fewer than 3 numeric tokens are found.
    """
    # Split by comma OR whitespace
    if "," in raw_line:
        toks = raw_line.strip().split(",")
    else:
        toks = raw_line.strip().split()

    nums = []
    for t in toks:
        try:
            v = float(t)
            nums.append(v)
        except ValueError:
            # skip non-numeric tokens
            continue
        if len(nums) >= 3:
            break

    if len(nums) >= 3:
        return nums[0], nums[1], nums[2]
    return None


class SerialReader(Thread):
    def __init__(self, port: str, baud: int, buf_seconds: float, fs_hint: float):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.fs_hint = fs_hint
        self.dt_hint = 1.0 / max(1e-9, fs_hint)

        nmax = int(max(100, buf_seconds * fs_hint))
        self.x = deque(maxlen=nmax)
        self.y = deque(maxlen=nmax)
        self.z = deque(maxlen=nmax)
        self.ts = deque(maxlen=nmax)

        self._lock = Lock()
        self._stop = False
        self._ser = None

        # saving
        os.makedirs(SAVE_DIR, exist_ok=True)
        self._saving = False
        self._save_fp = None

    def run(self):
        while not self._stop:
            try:
                if self._ser is None:
                    self._ser = serial.Serial(self.port, self.baud, timeout=1.0)
                    # small delay to settle
                    time.sleep(0.1)

                line = self._ser.readline().decode(errors="ignore")
                if not line:
                    continue

                parsed = _parse_numeric_triplet(line)
                if parsed is None:
                    continue
                ax, ay, az = parsed
                t = time.time()

                with self._lock:
                    self.x.append(ax)
                    self.y.append(ay)
                    self.z.append(az)
                    self.ts.append(t)

                    if self._saving and self._save_fp:
                        # CSV: epoch, ax, ay, az
                        self._save_fp.write(f"{t:.6f},{ax:.6f},{ay:.6f},{az:.6f}\n")

            except (serial.SerialException, OSError) as e:
                # Port hiccup—try to reopen after a short pause
                self._close_serial()
                time.sleep(0.5)
            except Exception:
                # Keep running even on a bad line
                pass

        self._close_serial()
        self._close_save()

    def _close_serial(self):
        try:
            if self._ser is not None:
                self._ser.close()
        finally:
            self._ser = None

    def stop(self):
        self._stop = True

    # --------- Export data safely ----------
    def snapshot(self):
        with self._lock:
            xs = np.array(self.x, dtype=float)
            ys = np.array(self.y, dtype=float)
            zs = np.array(self.z, dtype=float)
            ts = np.array(self.ts, dtype=float)
        return ts, xs, ys, zs

    # --------- Saving control ----------
    def start_save(self):
        with self._lock:
            if self._saving:
                return
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SAVE_DIR, f"imu_{ts}.csv")
            self._save_fp = open(path, "w", buffering=1)
            self._save_fp.write("epoch,ax,ay,az\n")
            self._saving = True
            print(f"[SAVE] Started: {path}")

    def stop_save(self):
        with self._lock:
            if not self._saving:
                return
            print("[SAVE] Stopped.")
            self._saving = False
            self._close_save()

    def _close_save(self):
        try:
            if self._save_fp:
                self._save_fp.flush()
                self._save_fp.close()
        finally:
            self._save_fp = None


def compute_fft_mag(sig: np.ndarray, fs: float):
    """
    Returns (freqs, mag) for rFFT of input sig.
    """
    if sig.size < 4:
        return np.array([]), np.array([])
    # detrend
    s = sig - np.mean(sig)
    n = s.size
    # apply Hann window
    w = np.hanning(n)
    sw = s * w
    # rFFT
    spec = np.fft.rfft(sw)
    mag = np.abs(spec) * 2.0 / np.sum(w)  # amplitude spectrum (roughly)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, mag


def estimate_fs(ts: np.ndarray, fs_hint: float):
    """
    Estimate sampling rate from timestamps; fall back to fs_hint if too small/invalid.
    """
    if ts.size >= 4:
        dt = np.diff(ts)
        dt = dt[(dt > 0) & (dt < 1.0)]  # ignore outliers
        if dt.size >= 3:
            med = float(np.median(dt))
            if 1e-5 < med < 1.0:
                return 1.0 / med
    return fs_hint


def find_top2_peaks(freqs: np.ndarray,
                    mag: np.ndarray,
                    min_hz: float = FFT_MIN_HZ,
                    rel_min: float = FFT_REL_MIN,
                    suppress_halfspan: int = FFT_SUPPRESS_HALFSPAN):
    """
    Return up to two (freq, mag) peaks >= min_hz and >= rel_min * global_max.
    Suppress a small neighborhood around the tallest to find the 2nd.
    """
    if freqs.size == 0 or mag.size != freqs.size:
        return []

    valid = freqs >= float(min_hz)
    if not np.any(valid):
        return []

    mv = mag.copy()
    global_max = float(np.max(mv))
    if global_max <= 0.0:
        return []

    peaks = []

    # Tallest peak among valid
    k1_local = int(np.argmax(mv[valid]))
    idx_valid = np.nonzero(valid)[0]
    i1 = idx_valid[k1_local]
    f1, m1 = float(freqs[i1]), float(mv[i1])
    if m1 >= rel_min * global_max:
        peaks.append((f1, m1))

    # Suppress neighborhood around tallest
    if suppress_halfspan > 0:
        lo = max(0, i1 - suppress_halfspan)
        hi = min(mv.size, i1 + suppress_halfspan + 1)
        mv[lo:hi] = 0.0

    # Second tallest
    if np.any(valid):
        k2_local = int(np.argmax(mv[valid]))
        i2 = idx_valid[k2_local]
        f2, m2 = float(freqs[i2]), float(mv[i2])
        if m2 >= rel_min * global_max and (len(peaks) == 0 or i2 != i1):
            peaks.append((f2, m2))

    return peaks


class OneWindowUI:
    def __init__(self, reader: SerialReader, window_s: float, fs_hint: float):
        self.reader = reader
        self.window_s = float(window_s)
        self.fs_hint = float(fs_hint)

        # --- Figure layout: 2 rows x 1 col: time (top), FFT (bottom)
        self.fig = plt.figure("IMU Live — One Window", figsize=(10, 7))
        gs = self.fig.add_gridspec(3, 3, height_ratios=[3, 3, 1])

        self.ax_time = self.fig.add_subplot(gs[0, :])
        self.ax_fft = self.fig.add_subplot(gs[1, :])

        # Buttons row
        ax_btn_start = self.fig.add_subplot(gs[2, 0])
        ax_btn_stop = self.fig.add_subplot(gs[2, 1])
        ax_btn_quit = self.fig.add_subplot(gs[2, 2])

        self.btn_start = Button(ax_btn_start, "Start Save")
        self.btn_stop = Button(ax_btn_stop, "Stop Save")
        self.btn_quit = Button(ax_btn_quit, "Quit")

        self.btn_start.on_clicked(self._on_start_save)
        self.btn_stop.on_clicked(self._on_stop_save)
        self.btn_quit.on_clicked(self._on_quit)

        # Lines for time domain
        (self.line_x,) = self.ax_time.plot([], [], label="X")
        (self.line_y,) = self.ax_time.plot([], [], label="Y")
        (self.line_z,) = self.ax_time.plot([], [], label="Z")
        self.ax_time.set_title("Time Domain (most recent window)")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("Value")
        self.ax_time.grid(True, alpha=0.3)
        self.ax_time.legend(loc="upper right")

        # FFT line
        (self.line_fft,) = self.ax_fft.plot([], [])
        # Tallest peak marker + label
        (self.peak1_marker,) = self.ax_fft.plot([], [], marker="o", linestyle="None", alpha=0.95, visible=False)
        self.peak1_text = self.ax_fft.text(0, 0, "", fontsize=9, ha="left", va="bottom", alpha=0.95, visible=False)
        # 2nd tallest peak marker + label
        (self.peak2_marker,) = self.ax_fft.plot([], [], marker="x", linestyle="None", alpha=0.95, visible=False)
        self.peak2_text = self.ax_fft.text(0, 0, "", fontsize=9, ha="left", va="bottom", alpha=0.95, visible=False)

        self.ax_fft.set_title("FFT Magnitude (|X|+|Y|+|Z| combined)")
        self.ax_fft.set_xlabel("Frequency (Hz)")
        self.ax_fft.set_ylabel("Amplitude")
        self.ax_fft.grid(True, alpha=0.3)

        self._running = True
        self._last_redraw = 0.0

    # -------- Buttons --------
    def _on_start_save(self, _evt):
        self.reader.start_save()

    def _on_stop_save(self, _evt):
        self.reader.stop_save()

    def _on_quit(self, _evt):
        self._running = False
        plt.close(self.fig)

    # -------- Update --------
    def update(self):
        """
        Manual UI loop using plt.pause(); call in a while loop.
        """
        now = time.time()
        if now - self._last_redraw < 1.0 / UI_UPDATE_HZ:
            return
        self._last_redraw = now

        ts, xs, ys, zs = self.reader.snapshot()
        if ts.size < 4:
            plt.pause(0.001)
            return

        # Time-window crop
        tmax = ts[-1]
        tmin = tmax - self.window_s
        m = ts >= tmin
        if not np.any(m):
            plt.pause(0.001)
            return

        t_win = ts[m]
        x_win = xs[m]
        y_win = ys[m]
        z_win = zs[m]

        # Normalize time to seconds from end
        t0 = t_win[0]
        t_rel = t_win - t0

        # ---- Time plot
        self.line_x.set_data(t_rel, x_win)
        self.line_y.set_data(t_rel, y_win)
        self.line_z.set_data(t_rel, z_win)

        # Set x-lim tightly to window
        self.ax_time.set_xlim(0, max(1e-3, t_rel[-1]))
        # Auto y-lim with small padding
        vmin = float(np.min([x_win.min(), y_win.min(), z_win.min()]))
        vmax = float(np.max([x_win.max(), y_win.max(), z_win.max()]))
        if np.isfinite(vmin) and np.isfinite(vmax):
            if vmax - vmin < 1e-12:
                pad = 1.0
            else:
                pad = 0.05 * (vmax - vmin)
            self.ax_time.set_ylim(vmin - pad, vmax + pad)

        # ---- FFT (combine magnitudes)
        fs = estimate_fs(t_win, fs_hint=FS_HINT)
        # resample to uniform spacing if timestamps are irregular—simple approach:
        # if spacing irregularity is small, FFT is still okay; otherwise fallback to hint.
        # Here, we'll just use the measured fs if reasonable, else hint.
        fxs, magx = compute_fft_mag(x_win, fs)
        fys, magy = compute_fft_mag(y_win, fs)
        fzs, magz = compute_fft_mag(z_win, fs)

        # All freq grids should be equal if sizes equal; use safest union
        # We will downselect to the shortest vector for simplicity
        lengths = [fxs.size, fys.size, fzs.size]
        L = min([l for l in lengths if l > 0] + [0])
        if L > 0:
            f = fxs[:L] if fxs.size >= L else (fys[:L] if fys.size >= L else fzs[:L])
            mx = magx[:L] if magx.size >= L else np.zeros(L)
            my = magy[:L] if magy.size >= L else np.zeros(L)
            mz = magz[:L] if magz.size >= L else np.zeros(L)
            mag = mx + my + mz
        else:
            f = np.array([])
            mag = np.array([])

        self.line_fft.set_data(f, mag)

        if f.size:
            self.ax_fft.set_xlim(0, max(5.0, f[-1]))
            # dynamic y
            mmax = float(np.max(mag)) if mag.size else 1.0
            self.ax_fft.set_ylim(0, mmax * 1.1 if mmax > 0 else 1.0)

            # ---- Annotate tallest & 2nd tallest peaks
            peaks = find_top2_peaks(f, mag, min_hz=FFT_MIN_HZ, rel_min=FFT_REL_MIN)
            # Hide all by default
            self.peak1_marker.set_visible(False)
            self.peak1_text.set_visible(False)
            self.peak2_marker.set_visible(False)
            self.peak2_text.set_visible(False)

            if len(peaks) >= 1:
                f1, m1 = peaks[0]
                self.peak1_marker.set_data([f1], [m1])
                self.peak1_marker.set_visible(True)
                self.peak1_text.set_position((f1, m1))
                self.peak1_text.set_text(f"{f1:.2f} Hz")
                self.peak1_text.set_visible(True)

            if len(peaks) >= 2:
                f2, m2 = peaks[1]
                self.peak2_marker.set_data([f2], [m2])
                self.peak2_marker.set_visible(True)
                self.peak2_text.set_position((f2, m2))
                self.peak2_text.set_text(f"{f2:.2f} Hz (2nd)")
                self.peak2_text.set_visible(True)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def pick_port_from_globs(globs):
    # Respect explicit string that isn't a glob
    if isinstance(globs, str) and ("*" not in globs and "?" not in globs):
        return globs

    # Expand
    candidates = []
    if isinstance(globs, str):
        glist = [globs]
    else:
        glist = list(globs)
    for g in glist:
        candidates.extend(sorted(glob.glob(g)))

    # Fall back to discovered ports if nothing matched
    if not candidates:
        ports = [p.device for p in list_ports.comports()]
        if ports:
            return ports[0]
        return None
    return candidates[0]


def main():
    ap = argparse.ArgumentParser(description="One-window IMU plotter with 2-peak FFT annotation + saving.")
    ap.add_argument("--port", type=str, default=None,
                    help=f"Serial port or glob. Defaults to search: {DEFAULT_PORT_GLOBS}")
    ap.add_argument("--baud", type=int, default=BAUD_DEFAULT)
    ap.add_argument("--window", type=float, default=WINDOW_SECONDS, help="Time window (seconds) to display.")
    ap.add_argument("--fs", type=float, default=FS_HINT, help="Sampling rate hint (Hz) if timestamps are irregular.")
    args = ap.parse_args()

    port = args.port or pick_port_from_globs(DEFAULT_PORT_GLOBS)
    if not port:
        print("ERROR: No serial port found. Try --port /dev/ttyACM0 (or a glob like '/dev/ttyACM*').")
        sys.exit(1)

    print(f"[INFO] Using port={port}, baud={args.baud}, window={args.window}s, fs_hint={args.fs}Hz")

    reader = SerialReader(port=port, baud=args.baud, buf_seconds=args.window, fs_hint=args.fs)
    reader.start()

    ui = OneWindowUI(reader, window_s=args.window, fs_hint=args.fs)
    try:
        while ui._running:
            ui.update()
    finally:
        reader.stop()
        reader.join(timeout=1.0)
        # ensure save file is closed even if user killed the window
        reader.stop_save()


if __name__ == "__main__":
    main()
