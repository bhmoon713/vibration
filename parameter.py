#!/usr/bin/env python3
# Live IMU plotter with parameterized windows/groups/layout.
# - Groups: acc(ax,ay,az), gyr(gx,gy,gz)
# - Layouts:
#     * per-axis  : one window per channel, each with its own FFT
#     * combined  : one window per group, 3 lines in time plot, one FFT per group
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
from threading import Lock
from collections import deque
import matplotlib.pyplot as plt

# -------------------- Config (defaults; can be overridden by CLI) --------------------
DEFAULT_PORT_GLOBS = ["/dev/ttyACM*", "/dev/ttyUSB*"]
BAUD = 115200

# Rolling window for time plot (seconds)
WINDOW_S = 2.0

# Unit scaling (set to 9.80665 if your accel is in g and you want m/s^2)
SCALE_ACC = 1.0
# If gyro is in deg/s and you want rad/s, set to np.pi/180
SCALE_GYR = 1.0

# FFT minimum samples
FFT_MIN_SAMPLES = 256

# Deque capacity (enough for a few seconds at 1 kHz without reallocation)
DEQUE_MAX = 100000
# ------------------------------------------------------------------------------------

CHANNEL_GROUPS = {
    "acc": ["ax", "ay", "az"],
    "gyr": ["gx", "gy", "gz"],
}

TITLE_PREFIX = {
    "acc": "ACC",
    "gyr": "GYR",
}

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
    return 1 << ((n.bit_length() - 1))


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
        self.buf_t = deque(maxlen=DEQUE_MAX)   # seconds, start at 0
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
            self._reader_th.join(timeout=1)

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
                return
            t_rel = t_s - (self.buf_t[0])

        self._append_sample(t_rel, ax, ay, az, gx, gy, gz, seq)

        # Update fs estimate robustly
        if ts_hist is not None:
            ts_hist.append(t_s)
            if len(ts_hist) > 3:
                dt = np.diff(np.array(ts_hist))
                good = dt[(dt > 4e-4) & (dt < 2e-3)]  # 500..2500 Hz plausible
                if good.size > 5:
                    self.fs_est = 1.0 / float(np.median(good))

    # ---------- Plotting ----------
    def run_ui(self):
        if self.layout == "per-axis":
            self._run_ui_per_axis()
        else:
            self._run_ui_combined()

    def _run_ui_per_axis(self):
        # Create one window per channel across selected groups
        fig_axes = {}
        for grp in self.groups:
            for ch in CHANNEL_GROUPS[grp]:
                fig = plt.figure(f"{TITLE_PREFIX[grp]} {ch} – Time & FFT")
                ax_time = plt.subplot(2, 1, 1)
                ax_fft  = plt.subplot(2, 1, 2)

                line_time, = ax_time.plot([], [], label=ch)
                ax_time.set_xlabel("Time (s)")
                ax_time.set_ylabel("Signal")
                ax_time.legend(loc="upper right")
                ax_time.grid(True)

                line_fft, = ax_fft.plot([], [])
                ax_fft.set_xlabel("Frequency (Hz)")
                ax_fft.set_ylabel("|FFT|")
                ax_fft.grid(True)

                fig_axes[ch] = {
                    "ax_time": ax_time,
                    "ax_fft": ax_fft,
                    "line_time": line_time,
                    "line_fft": line_fft,
                }

        try:
            while True:
                time.sleep(0.02)
                t, snap = self._snapshot()
                if t.size < 10:
                    plt.pause(0.001); continue

                minlen = min(
                    t.size, *(snap[k].size for k in ["ax","ay","az","gx","gy","gz"])
                )
                if minlen < 10:
                    plt.pause(0.001); continue

                t = t[-minlen:]
                for k in list(snap.keys()):
                    snap[k] = snap[k][-minlen:]

                t_rel = t - t[-1]
                tmask = t_rel >= -self.window_s
                if np.count_nonzero(tmask) < 10:
                    plt.pause(0.001); continue

                fs = max(1.0, self.fs_est)

                for grp in self.groups:
                    for ch in CHANNEL_GROUPS[grp]:
                        axes = fig_axes[ch]
                        y = snap[ch][tmask]

                        # time
                        axes["line_time"].set_data(t_rel[tmask], y)
                        axes["ax_time"].set_xlim(-self.window_s, 0)
                        if y.size > 0:
                            y_min, y_max = float(np.min(y)), float(np.max(y))
                            pad = 0.1 * (y_max - y_min + 1e-9)
                            axes["ax_time"].set_ylim(y_min - pad, y_max + pad)

                        # fft (channel's own)
                        if y.size >= FFT_MIN_SAMPLES:
                            L = power_of_two_le(y.size)
                            sig = y[-L:] - float(np.mean(y[-L:]))
                            win = np.hanning(L)
                            Y = np.fft.rfft(sig * win)
                            freqs = np.fft.rfftfreq(L, d=1.0 / fs)
                            mag = np.abs(Y) / (L / 2.0)

                            axes["line_fft"].set_data(freqs, mag)
                            axes["ax_fft"].set_xlim(0, fs/2.0)
                            axes["ax_fft"].set_ylim(0, max(1e-12, float(np.max(mag))) * 1.1)

                plt.pause(0.001)
        except KeyboardInterrupt:
            pass

    def _run_ui_combined(self):
        # Create one window per group (time plot has 3 lines; single FFT per group)
        fig_groups = {}
        for grp in self.groups:
            chans = CHANNEL_GROUPS[grp]
            fig = plt.figure(f"{TITLE_PREFIX[grp]} – Time (all) & FFT({self.fft_ch_by_group.get(grp, chans[0])})")
            ax_time = plt.subplot(2, 1, 1)
            ax_fft  = plt.subplot(2, 1, 2)

            lines_time = {ch: ax_time.plot([], [], label=ch)[0] for ch in chans}
            ax_time.set_xlabel("Time (s)")
            ax_time.set_ylabel("Signal")
            ax_time.legend(loc="upper right")
            ax_time.grid(True)

            line_fft, = ax_fft.plot([], [])
            ax_fft.set_xlabel("Frequency (Hz)")
            ax_fft.set_ylabel("|FFT|")
            ax_fft.grid(True)

            fig_groups[grp] = {
                "ax_time": ax_time,
                "ax_fft": ax_fft,
                "lines_time": lines_time,
                "line_fft": line_fft,
            }

        try:
            while True:
                time.sleep(0.02)
                t, snap = self._snapshot()
                if t.size < 10:
                    plt.pause(0.001); continue

                minlen = min(
                    t.size, *(snap[k].size for k in ["ax","ay","az","gx","gy","gz"])
                )
                if minlen < 10:
                    plt.pause(0.001); continue

                t = t[-minlen:]
                for k in list(snap.keys()):
                    snap[k] = snap[k][-minlen:]

                t_rel = t - t[-1]
                tmask = t_rel >= -self.window_s
                if np.count_nonzero(tmask) < 10:
                    plt.pause(0.001); continue

                fs = max(1.0, self.fs_est)

                for grp in self.groups:
                    chans = CHANNEL_GROUPS[grp]
                    axes = fig_groups[grp]

                    # time lines
                    yall = []
                    for ch in chans:
                        y = snap[ch][tmask]
                        axes["lines_time"][ch].set_data(t_rel[tmask], y)
                        yall.append(y)
                    axes["ax_time"].set_xlim(-self.window_s, 0)
                    ycat = np.hstack([y for y in yall if y.size > 0]) if any(len(y)>0 for y in yall) else np.array([0.0])
                    y_min, y_max = float(np.min(ycat)), float(np.max(ycat))
                    pad = 0.1 * (y_max - y_min + 1e-9)
                    axes["ax_time"].set_ylim(y_min - pad, y_max + pad)

                    # fft on selected channel for this group
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
                        axes["ax_fft"].set_xlim(0, fs/2.0)
                        axes["ax_fft"].set_ylim(0, max(1e-12, float(np.max(mag))) * 1.1)

                plt.pause(0.001)
        except KeyboardInterrupt:
            pass


def parse_args():
    p = argparse.ArgumentParser(description="Live IMU plotter (parameterized groups/layout)")
    p.add_argument("port", nargs="?", help="Serial port (auto-detect if omitted)")
    p.add_argument("--baud", type=int, default=BAUD, help=f"Baud rate (default {BAUD})")
    p.add_argument("--window-s", type=float, default=WINDOW_S, help="Time window seconds (default 2.0)")
    p.add_argument("--groups", action="append", choices=list(CHANNEL_GROUPS.keys()),
                   help="Signal groups to show (acc/gyr). Repeat for multiple.")
    p.add_argument("--layout", choices=["per-axis","combined"], default="per-axis",
                   help="Window layout: per-axis or combined (default: per-axis)")
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
        # assign per group, prefer user-provided; else default to first channel in group
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
