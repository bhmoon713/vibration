#!/usr/bin/env python3
# Live IMU plotter with parameterized groups/layout, color mapping, line styles,
# live FFT peak annotations, and CSV logging (Start/Stop buttons).
#
# Examples:
#   python3 imu_plot.py --groups acc --layout per-axis
#   python3 imu_plot.py --groups acc --groups gyr --layout combined --fft-ch az --fft-ch gz
#
import sys
import time
import glob
import argparse
import numpy as np
import serial
import threading
from threading import Lock, Event, Thread
from collections import deque
from queue import Queue, Full, Empty
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# -------------------- Defaults (can be overridden by CLI where noted) --------------------
DEFAULT_PORT_GLOBS = ["/dev/ttyACM*", "/dev/ttyUSB*"]
BAUD = 115200

# Rolling window for time plot (seconds)  [--window-s]
WINDOW_S = 2.0

# Unit scaling (set to 9.80665 if your accel is in g and you want m/s^2)
SCALE_ACC = 1.0
# If gyro is in deg/s and you want rad/s, set to np.pi/180
SCALE_GYR = 1.0

# FFT minimum samples to attempt an FFT
FFT_MIN_SAMPLES = 256

# Deque capacity (enough for a few seconds at 1 kHz without reallocation)
DEQUE_MAX = 100000

# Where to save CSV logs
LOG_DIR = "."

# Groups and channels
CHANNEL_GROUPS = {
    "acc": ["ax", "ay", "az"],
    "gyr": ["gx", "gy", "gz"],
}
TITLE_PREFIX = {"acc": "ACC", "gyr": "GYR"}

# 🎨 Color scheme per channel
COLOR_MAP = {
    "ax": "#FF3B30",  # red
    "ay": "#34C759",  # green
    "az": "#007AFF",  # blue
    "gx": "#FF9500",  # orange
    "gy": "#AF52DE",  # purple
    "gz": "#FFD60A",  # yellow
}

# --- Line style configuration ---
LINE_WIDTH_TIME = 1.8   # thickness for time-domain lines
LINE_WIDTH_FFT  = 1.0   # thickness for FFT lines
LINE_ALPHA_FFT  = 0.5   # transparency for FFT lines

# --- FFT peak annotation config ---
ANNOTATE_FFT_PEAKS = True   # turn on/off
FFT_PEAK_MIN_HZ    = 5.0    # ignore DC / very low freq
FFT_PEAK_MIN_REL   = 0.15   # show only if >= 15% of max magnitude

# CSV writer queue size (protect against bursts)
WRITER_QUEUE_MAX = 200000
# -----------------------------------------------------------------------------------------

def auto_port():
    cands = []
    for pattern in DEFAULT_PORT_GLOBS:
        cands += glob.glob(pattern)
    cands = sorted(set(cands))
    if not cands:
        raise RuntimeError(
            "No serial ports found. Plug in your board or specify a port:\n"
            "  python3 imu_plot.py /dev/ttyACM0"
        )
    print(f"[info] auto-selected port: {cands[0]}")
    return cands[0]


def power_of_two_le(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


class CSVWriterThread:
    """Background CSV writer with start/stop support."""
    def __init__(self, log_dir: str = LOG_DIR):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.q: Queue = Queue(maxsize=WRITER_QUEUE_MAX)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._file = None
        self._writer = None
        self._enabled = False
        self._rows_since_flush = 0
        self._flush_every = 1000  # flush every N rows to reduce data loss risk
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def start(self):
        if self._thread is not None:
            return
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            # push a sentinel to unblock
            try:
                self.q.put_nowait(None)
            except Full:
                pass
            self._thread.join(timeout=1.0)
            self._thread = None
        # close file if open
        self._close_file()

    def _open_new_file(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.log_dir, f"imu_{ts}.csv")
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["t_us", "ax", "ay", "az", "gx", "gy", "gz", "seq"])
        self._rows_since_flush = 0
        with self._lock:
            self._enabled = True
        return path

    def _close_file(self):
        with self._lock:
            self._enabled = False
        if self._file:
            try:
                self._file.flush()
            except Exception:
                pass
            try:
                self._file.close()
            except Exception:
                pass
        self._file = None
        self._writer = None
        self._rows_since_flush = 0

    def begin_logging(self) -> str:
        """Begin a new CSV file and return the path."""
        # if already logging, close current and open new
        self._close_file()
        return self._open_new_file()

    def end_logging(self):
        """Stop logging and close the file."""
        self._close_file()

    def enqueue(self, row):
        """Enqueue a row if logging is enabled. Row must be a sequence."""
        if not self.enabled:
            return
        try:
            self.q.put_nowait(row)
        except Full:
            # If overflow happens, silently drop newest row to avoid blocking serial read
            pass

    def _run(self):
        while not self._stop_event.is_set():
            try:
                item = self.q.get(timeout=0.2)
            except Empty:
                continue
            if item is None:  # sentinel
                break
            if self._writer is not None:
                try:
                    self._writer.writerow(item)
                    self._rows_since_flush += 1
                    if self._rows_since_flush >= self._flush_every:
                        self._file.flush()
                        self._rows_since_flush = 0
                except Exception:
                    # Ignore write errors to avoid crashing the UI
                    pass


class IMUPlotter:
    def __init__(self, port: str, baud: int, window_s: float,
                 groups: list[str], layout: str, fft_ch_by_group: dict[str, str]):
        self.port = port
        self.baud = baud
        self.window_s = float(window_s)
        self.groups = groups
        self.layout = layout  # "per-axis" or "combined"
        self.fft_ch_by_group = fft_ch_by_group  # used in combined layout

        # Shared buffers
        self.lock = Lock()
        self.buf_t  = deque(maxlen=DEQUE_MAX)  # seconds, start at 0
        self.buf_ax = deque(maxlen=DEQUE_MAX)
        self.buf_ay = deque(maxlen=DEQUE_MAX)
        self.buf_az = deque(maxlen=DEQUE_MAX)
        self.buf_gx = deque(maxlen=DEQUE_MAX)
        self.buf_gy = deque(maxlen=DEQUE_MAX)
        self.buf_gz = deque(maxlen=DEQUE_MAX)
        self.buf_seq = deque(maxlen=DEQUE_MAX)

        self.stop = False
        self.fs_est = 1000.0  # start assumption
        self._reader_th = None

        # CSV writer
        self.writer = CSVWriterThread(LOG_DIR)
        self.writer.start()

        # Controls UI elements (created in run_ui)
        self.ctrl_fig = None
        self.btn_start = None
        self.btn_stop = None
        self.ctrl_status_txt = None
        self._last_log_path = None

    # ---------- Thread-safe helpers ----------
    def _append_sample(self, t_s, ax, ay, az, gx, gy, gz, seq):
        with self.lock:
            self.buf_t.append(t_s)
            self.buf_ax.append(ax)
            self.buf_ay.append(ay)
            self.buf_az.append(az)
            self.buf_gx.append(gx)
            self.buf_gy.append(gy)
            self.buf_gz.append(gz)
            self.buf_seq.append(seq)

    def _snapshot(self):
        with self.lock:
            t = np.array(self.buf_t, dtype=float)
            snap = {
                "ax": np.array(self.buf_ax, dtype=float),
                "ay": np.array(self.buf_ay, dtype=float),
                "az": np.array(self.buf_az, dtype=float),
                "gx": np.array(self.buf_gx, dtype=float),
                "gy": np.array(self.buf_gy, dtype=float),
                "gz": np.array(self.buf_gz, dtype=float),
                "seq": np.array(self.buf_seq, dtype=float),
            }
        return t, snap

    # ---------- Reader thread ----------
    def start_reader(self):
        self._reader_th = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_th.start()

    def stop_reader(self):
        self.stop = True
        if self._reader_th is not None:
            self._reader_th.join(timeout=1.0)
        # stop CSV writer thread
        self.writer.stop()

    def _reader_loop(self):
        """Read CSV lines: t_us,ax,ay,az,gx,gy,gz[,seq]"""
        try:
            with serial.Serial(self.port, self.baud, timeout=1) as ser:
                # Try to skip header if present
                first = ser.readline().decode("utf-8", "ignore").strip()
                if not first.startswith("t_us"):
                    self._ingest_line(first)

                t0 = None
                ts_hist = deque(maxlen=4000)  # for fs estimation
                while not self.stop:
                    line = ser.readline().decode("utf-8", "ignore").strip()
                    if not line:
                        continue
                    self._ingest_line(line, t0_ref=[t0], ts_hist=ts_hist)
                    t0 = t0 if t0 is not None else (ts_hist[0] if ts_hist else None)
        except serial.SerialException as e:
            print(f"[error] Serial error: {e}")
        except Exception as e:
            print(f"[error] Reader exception: {e}")

    def _ingest_line(self, line, t0_ref=None, ts_hist=None):
        if not line or line.startswith("#"):
            return
        parts = line.split(",")
        if len(parts) < 7:
            return
        try:
            t_us = float(parts[0])
            ax = float(parts[1]) * SCALE_ACC
            ay = float(parts[2]) * SCALE_ACC
            az = float(parts[3]) * SCALE_ACC
            gx = float(parts[4]) * SCALE_GYR
            gy = float(parts[5]) * SCALE_GYR
            gz = float(parts[6]) * SCALE_GYR
            seq = float(parts[7]) if len(parts) >= 8 else np.nan
        except ValueError:
            return

        t_s = t_us * 1e-6
        if t0_ref is not None:
            if t0_ref[0] is None:
                t0_ref[0] = t_s
            t_rel = t_s - t0_ref[0]
        else:
            if len(self.buf_t) == 0:
                self._append_sample(0.0, ax, ay, az, gx, gy, gz, seq)
                # enqueue for CSV (use absolute t_us)
                if self.writer.enabled:
                    self.writer.enqueue((t_us, ax, ay, az, gx, gy, gz, seq))
                return
            t_rel = t_s - (self.buf_t[0])

        self._append_sample(t_rel, ax, ay, az, gx, gy, gz, seq)

        # enqueue for CSV (use absolute timestamp in microseconds)
        if self.writer.enabled:
            self.writer.enqueue((t_us, ax, ay, az, gx, gy, gz, seq))

        # Update fs estimate robustly
        if ts_hist is not None:
            ts_hist.append(t_s)
            if len(ts_hist) > 3:
                dt = np.diff(np.array(ts_hist))
                # robust range: 500..2500 Hz plausible sample intervals
                good = dt[(dt > 4e-4) & (dt < 2e-3)]
                if good.size > 5:
                    self.fs_est = 1.0 / float(np.median(good))

    # ---------- Control panel (buttons) ----------
    def _init_controls_window(self):
        self.ctrl_fig = plt.figure("IMU Controls", figsize=(4, 2.2))
        # layout: two buttons on top row, status text below
        ax_start = plt.axes([0.08, 0.58, 0.38, 0.32])
        ax_stop  = plt.axes([0.54, 0.58, 0.38, 0.32])
        self.btn_start = Button(ax_start, "Start Save")
        self.btn_stop  = Button(ax_stop, "Stop Save")

        # status text area
        ax_status = plt.axes([0.08, 0.12, 0.84, 0.32])
        ax_status.axis("off")
        self.ctrl_status_txt = ax_status.text(0, 0.5, "Status: Not saving",
                                              fontsize=10, va="center", ha="left")

        def on_start(_event):
            path = self.writer.begin_logging()
            self._last_log_path = path
            self.ctrl_status_txt.set_text(f"Status: Saving to\n{os.path.basename(path)}")

        def on_stop(_event):
            self.writer.end_logging()
            txt = "Status: Not saving"
            if self._last_log_path:
                txt += f"\nLast file: {os.path.basename(self._last_log_path)}"
            self.ctrl_status_txt.set_text(txt)

        self.btn_start.on_clicked(on_start)
        self.btn_stop.on_clicked(on_stop)
        self.ctrl_fig.canvas.manager.set_window_title("IMU Controls")

    # ---------- Plotting ----------
    def run_ui(self):
        # Create control panel window
        self._init_controls_window()

        if self.layout == "per-axis":
            self._run_ui_per_axis()
        else:
            self._run_ui_combined()

    def _run_ui_per_axis(self):
        # One window per channel across selected groups
        fig_axes = {}
        for grp in self.groups:
            for ch in CHANNEL_GROUPS[grp]:
                fig = plt.figure(f"{TITLE_PREFIX[grp]} {ch} – Time & FFT")
                ax_time = plt.subplot(2, 1, 1)
                ax_fft  = plt.subplot(2, 1, 2)

                c = COLOR_MAP.get(ch, "black")
                line_time, = ax_time.plot([], [], label=ch, color=c, linewidth=LINE_WIDTH_TIME)
                line_fft,  = ax_fft.plot([], [], color=c, alpha=LINE_ALPHA_FFT, linewidth=LINE_WIDTH_FFT)

                # Peak marker & label (start hidden)
                peak_marker, = ax_fft.plot([], [], marker='o', linestyle='None', color=c, alpha=0.9, visible=False)
                peak_text = ax_fft.text(0, 0, "", fontsize=9, color=c, alpha=0.9,
                                        ha='left', va='bottom', visible=False)

                ax_time.set_xlabel("Time (s)")
                ax_time.set_ylabel("Signal")
                ax_time.legend(loc="upper right")
                ax_time.grid(True)

                ax_fft.set_xlabel("Frequency (Hz)")
                ax_fft.set_ylabel("|FFT|")
                ax_fft.grid(True)

                fig_axes[ch] = {
                    "ax_time": ax_time, "ax_fft": ax_fft,
                    "line_time": line_time, "line_fft": line_fft,
                    "peak_marker": peak_marker, "peak_text": peak_text
                }

        try:
            while True:
                time.sleep(0.02)  # ~50 FPS
                t, snap = self._snapshot()
                if t.size < 10:
                    plt.pause(0.001); continue

                # Align to shortest length across all buffered channels
                minlen = min(
                    t.size, *(snap[k].size for k in ["ax", "ay", "az", "gx", "gy", "gz"])
                )
                if minlen < 10:
                    plt.pause(0.001); continue

                t = t[-minlen:]
                snap = {k: v[-minlen:] for k, v in snap.items()}

                # Relative, windowed time mask
                t_rel = t - t[-1]
                tmask = t_rel >= -self.window_s
                if np.count_nonzero(tmask) < 10:
                    plt.pause(0.001); continue

                fs = max(1.0, self.fs_est)

                # Update each accel/gyro channel window
                for grp in self.groups:
                    for ch in CHANNEL_GROUPS[grp]:
                        axes = fig_axes[ch]
                        y = snap[ch][tmask]

                        # Time-domain
                        axes["line_time"].set_data(t_rel[tmask], y)
                        axes["ax_time"].set_xlim(-self.window_s, 0)
                        if y.size > 0:
                            y_min, y_max = float(np.min(y)), float(np.max(y))
                            pad = 0.1 * (y_max - y_min + 1e-9)
                            axes["ax_time"].set_ylim(y_min - pad, y_max + pad)

                        # FFT
                        if y.size >= FFT_MIN_SAMPLES:
                            L = power_of_two_le(y.size)
                            sig = y[-L:] - float(np.mean(y[-L:]))
                            win = np.hanning(L)
                            Y = np.fft.rfft(sig * win)
                            freqs = np.fft.rfftfreq(L, d=1.0 / fs)
                            mag = np.abs(Y) / (L / 2.0)

                            axes["line_fft"].set_data(freqs, mag)
                            axes["ax_fft"].set_xlim(0, fs / 2.0)
                            axes["ax_fft"].set_ylim(0, max(1e-12, float(np.max(mag))) * 1.1)

                            # Peak annotation
                            if ANNOTATE_FFT_PEAKS:
                                valid = freqs >= FFT_PEAK_MIN_HZ
                                if np.any(valid):
                                    mag_valid = mag[valid]
                                    if mag_valid.size:
                                        k = int(np.argmax(mag_valid))
                                        idx = np.nonzero(valid)[0][k]
                                        fpk, mpk = float(freqs[idx]), float(mag[idx])
                                        if mpk >= FFT_PEAK_MIN_REL * float(np.max(mag)):
                                            axes["peak_marker"].set_data([fpk], [mpk])
                                            axes["peak_marker"].set_visible(True)
                                            axes["peak_text"].set_position((fpk, mpk))
                                            axes["peak_text"].set_text(f"{fpk:.1f} Hz")
                                            axes["peak_text"].set_visible(True)
                                        else:
                                            axes["peak_marker"].set_visible(False)
                                            axes["peak_text"].set_visible(False)
                                else:
                                    axes["peak_marker"].set_visible(False)
                                    axes["peak_text"].set_visible(False)

                plt.pause(0.001)
        except KeyboardInterrupt:
            pass

    def _run_ui_combined(self):
        # One window per group (time plot has 3 lines; single FFT per group)
        fig_groups = {}
        for grp in self.groups:
            chans = CHANNEL_GROUPS[grp]
            fig = plt.figure(f"{TITLE_PREFIX[grp]} – Time (all) & FFT")
            ax_time = plt.subplot(2, 1, 1)
            ax_fft  = plt.subplot(2, 1, 2)

            lines_time = {
                ch: ax_time.plot([], [], label=ch,
                                 color=COLOR_MAP.get(ch, "black"),
                                 linewidth=LINE_WIDTH_TIME)[0]
                for ch in chans
            }
            fft_ch = self.fft_ch_by_group.get(grp, chans[0])
            fft_color = COLOR_MAP.get(fft_ch, "gray")
            line_fft, = ax_fft.plot([], [], color=fft_color,
                                    alpha=LINE_ALPHA_FFT, linewidth=LINE_WIDTH_FFT)

            # Peak marker & label (start hidden)
            peak_marker, = ax_fft.plot([], [], marker='o', linestyle='None',
                                       color=fft_color, alpha=0.9, visible=False)
            peak_text = ax_fft.text(0, 0, "", fontsize=9, color=fft_color,
                                    alpha=0.9, ha='left', va='bottom', visible=False)

            ax_time.set_xlabel("Time (s)")
            ax_time.set_ylabel("Signal")
            ax_time.legend(loc="upper right")
            ax_time.grid(True)

            ax_fft.set_xlabel("Frequency (Hz)")
            ax_fft.set_ylabel("|FFT|")
            ax_fft.grid(True)

            fig_groups[grp] = {
                "ax_time": ax_time, "ax_fft": ax_fft,
                "lines_time": lines_time, "line_fft": line_fft,
                "peak_marker": peak_marker, "peak_text": peak_text
            }

        try:
            while True:
                time.sleep(0.02)
                t, snap = self._snapshot()
                if t.size < 10:
                    plt.pause(0.001); continue

                minlen = min(
                    t.size, *(snap[k].size for k in ["ax", "ay", "az", "gx", "gy", "gz"])
                )
                if minlen < 10:
                    plt.pause(0.001); continue

                t = t[-minlen:]
                snap = {k: v[-minlen:] for k, v in snap.items()}

                t_rel = t - t[-1]
                tmask = t_rel >= -self.window_s
                if np.count_nonzero(tmask) < 10:
                    plt.pause(0.001); continue

                fs = max(1.0, self.fs_est)

                for grp in self.groups:
                    chans = CHANNEL_GROUPS[grp]
                    axes = fig_groups[grp]

                    # Time (3 lines)
                    yall = []
                    for ch in chans:
                        y = snap[ch][tmask]
                        axes["lines_time"][ch].set_data(t_rel[tmask], y)
                        yall.append(y)
                    axes["ax_time"].set_xlim(-self.window_s, 0)
                    if any(len(y) > 0 for y in yall):
                        ycat = np.hstack([y for y in yall if y.size > 0])
                    else:
                        ycat = np.array([0.0])
                    y_min, y_max = float(np.min(ycat)), float(np.max(ycat))
                    pad = 0.1 * (y_max - y_min + 1e-9)
                    axes["ax_time"].set_ylim(y_min - pad, y_max + pad)

                    # FFT on selected channel for this group
                    fft_ch = self.fft_ch_by_group.get(grp, chans[0])
                    yf = snap[fft_ch][tmask]
                    if yf.size >= FFT_MIN_SAMPLES:
                        L = power_of_two_le(yf.size)
                        sig = yf[-L:] - float(np.mean(yf[-L:]))
                        win = np.hanning(L)
                        Y = np.fft.rfft(sig * win)
                        freqs = np.fft.rfftfreq(L, d=1.0 / fs)
                        mag = np.abs(Y) / (L / 2.0)

                        axes["line_fft"].set_data(freqs, mag)
                        axes["ax_fft"].set_xlim(0, fs / 2.0)
                        axes["ax_fft"].set_ylim(0, max(1e-12, float(np.max(mag))) * 1.1)

                        # Peak annotation
                        if ANNOTATE_FFT_PEAKS:
                            valid = freqs >= FFT_PEAK_MIN_HZ
                            if np.any(valid):
                                mag_valid = mag[valid]
                                if mag_valid.size:
                                    k = int(np.argmax(mag_valid))
                                    idx = np.nonzero(valid)[0][k]
                                    fpk, mpk = float(freqs[idx]), float(mag[idx])
                                    if mpk >= FFT_PEAK_MIN_REL * float(np.max(mag)):
                                        axes["peak_marker"].set_data([fpk], [mpk])
                                        axes["peak_marker"].set_visible(True)
                                        axes["peak_text"].set_position((fpk, mpk))
                                        axes["peak_text"].set_text(f"{fpk:.1f} Hz")
                                        axes["peak_text"].set_visible(True)
                                    else:
                                        axes["peak_marker"].set_visible(False)
                                        axes["peak_text"].set_visible(False)

                plt.pause(0.001)
        except KeyboardInterrupt:
            pass


# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Live IMU plotter (parameterized)")
    p.add_argument("port", nargs="?", help="Serial port (auto if omitted)")
    p.add_argument("--baud", type=int, default=BAUD, help=f"Baud rate (default {BAUD})")
    p.add_argument("--window-s", type=float, default=WINDOW_S, help="Time window seconds (default 2.0)")
    p.add_argument("--groups", action="append", choices=list(CHANNEL_GROUPS.keys()),
                   help="Signal groups to show: acc / gyr. Repeat for multiple.")
    p.add_argument("--layout", choices=["per-axis", "combined"], default="per-axis",
                   help="Window layout (default: per-axis).")
    # For combined layout, you can pass one or more --fft-ch in the same order as --groups.
    p.add_argument("--fft-ch", action="append",
                   help="FFT channel per group (combined layout). Repeat to match --groups order.")
    args = p.parse_args()

    # Defaults
    if not args.groups:
        args.groups = ["acc"]

    # Build fft channel map (combined only)
    fft_map = {}
    if args.layout == "combined":
        user_fft = args.fft_ch or []
        for i, grp in enumerate(args.groups):
            if i < len(user_fft) and user_fft[i] in CHANNEL_GROUPS[grp]:
                fft_map[grp] = user_fft[i]
            else:
                fft_map[grp] = CHANNEL_GROUPS[grp][0]
    return args, fft_map


def main():
    args, fft_map = parse_args()
    port = args.port if args.port else auto_port()
    print(f"[info] connecting to {port} @ {args.baud} baud")
    print(f"[info] groups={args.groups}, layout={args.layout}, fft_map={fft_map}")

    app = IMUPlotter(
        port=port,
        baud=args.baud,
        window_s=args.window_s,
        groups=args.groups,
        layout=args.layout,
        fft_ch_by_group=fft_map,
    )
    app.start_reader()
    try:
        app.run_ui()
    finally:
        app.stop_reader()
        print("[info] exiting cleanly")


if __name__ == "__main__":
    main()
