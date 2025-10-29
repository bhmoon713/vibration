# Linux
# python3 imu_binary_plot.py /dev/ttyACM0 --save run1.csv

# Windows (pick your COM port)
# python imu_binary_plot.py COM3 --save run1.csv

#!/usr/bin/env python3
import sys, os, glob, time, struct, threading
import numpy as np
import serial
from collections import deque
from threading import Lock
import matplotlib.pyplot as plt

# ---- serial & frame config ----
BAUD = 2000000
FRAME = struct.Struct("<IffffffI")  # t_us, ax,ay,az,gx,gy,gz,seq (32 bytes)
WINDOW_S = 2.0
TIME_CHANNELS = ["ax","ay","az"]
FFT_CH = "az"
FFT_MIN = 256
BUF_MAX = 100000

def auto_port():
    if os.name == "nt":
        return "COM3"  # change if needed
    cands = sorted(set(glob.glob("/dev/ttyACM*")+glob.glob("/dev/ttyUSB*")))
    if not cands:
        raise RuntimeError("No serial ports found. Specify one, e.g. /dev/ttyACM0 or COM3")
    return cands[0]

class BinReader:
    def __init__(self, port, baud, save_path=None):
        self.port = port; self.baud = baud
        self.save_path = save_path
        self.lock = Lock()
        self.stop = False
        # ring buffers
        self.t = deque(maxlen=BUF_MAX)
        self.ax = deque(maxlen=BUF_MAX); self.ay = deque(maxlen=BUF_MAX); self.az = deque(maxlen=BUF_MAX)
        self.gx = deque(maxlen=BUF_MAX); self.gy = deque(maxlen=BUF_MAX); self.gz = deque(maxlen=BUF_MAX)
        self.seq = deque(maxlen=BUF_MAX)
        self.fs = 1000.0

    def start(self):
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def stop_join(self):
        self.stop = True
        if hasattr(self, "th"):
            self.th.join(timeout=1)

    def _loop(self):
        out = None
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            out = open(self.save_path, "w", buffering=1)
            out.write("t_us,ax,ay,az,gx,gy,gz,seq\n")

        try:
            with serial.Serial(self.port, self.baud, timeout=0.1) as ser:
                ser.reset_input_buffer()
                buf = bytearray()
                t0 = None
                ts_hist = deque(maxlen=4000)
                while not self.stop:
                    chunk = ser.read(4096)
                    if chunk:
                        buf += chunk
                    sz = FRAME.size
                    n = len(buf) // sz
                    if n == 0:
                        continue
                    for i in range(n):
                        raw = bytes(buf[i*sz:(i+1)*sz])
                        t_us, ax, ay, az, gx, gy, gz, seq = FRAME.unpack(raw)
                        t_s = t_us * 1e-6
                        if t0 is None: t0 = t_s
                        tr = t_s - t0
                        with self.lock:
                            self.t.append(tr)
                            self.ax.append(ax); self.ay.append(ay); self.az.append(az)
                            self.gx.append(gx); self.gy.append(gy); self.gz.append(gz)
                            self.seq.append(seq)
                        if out:
                            out.write(f"{t_us},{ax:.6f},{ay:.6f},{az:.6f},{gx:.6f},{gy:.6f},{gz:.6f},{seq}\n")
                        ts_hist.append(t_s)
                        if len(ts_hist) > 5:
                            dt = np.diff(np.array(ts_hist))
                            good = dt[(dt>4e-4)&(dt<2e-3)]
                            if good.size > 5:
                                self.fs = 1.0/np.median(good)
                    del buf[:n*sz]
        finally:
            if out: out.close()

    def snap(self):
        with self.lock:
            t  = np.array(self.t,  dtype=float)
            ax = np.array(self.ax, dtype=float); ay = np.array(self.ay, dtype=float); az = np.array(self.az, dtype=float)
            gx = np.array(self.gx, dtype=float); gy = np.array(self.gy, dtype=float); gz = np.array(self.gz, dtype=float)
        return t, {"ax":ax,"ay":ay,"az":az,"gx":gx,"gy":gy,"gz":gz}

def pow2_le(n): return 1<<(n.bit_length()-1) if n>0 else 1

def main():
    # args: [port] [--save file.csv]
    port = None; save = None
    args = sys.argv[1:]
    if args and not args[0].startswith("-"): port = args.pop(0)
    while args:
        a = args.pop(0)
        if a == "--save" and args: save = args.pop(0)
    port = port or auto_port()
    print(f"[info] opening {port} @ {BAUD}, binary 32B frames. save={save or 'no'}")

    r = BinReader(port, BAUD, save_path=save)
    r.start()

    plt.figure("IMU (binary) – Time & FFT")
    ax_time = plt.subplot(2,1,1); ax_fft = plt.subplot(2,1,2)
    lines = {ch: ax_time.plot([],[], label=ch)[0] for ch in TIME_CHANNELS}
    ax_time.legend(); ax_time.grid(True)
    ax_time.set_xlabel("Time (s)"); ax_time.set_ylabel("Signal")
    lfft, = ax_fft.plot([],[]); ax_fft.grid(True)
    ax_fft.set_xlabel("Frequency (Hz)"); ax_fft.set_ylabel("|FFT|")

    try:
        while True:
            time.sleep(0.02)
            t, s = r.snap()
            if t.size < 10:
                plt.pause(0.001); continue
            t_rel = t - t.max()
            mask = t_rel >= -WINDOW_S

            # time plot
            yall = []
            for ch in TIME_CHANNELS:
                y = s[ch][mask]
                lines[ch].set_data(t_rel[mask], y)
                yall.append(y)
            if yall:
                yy = np.hstack(yall)
                ax_time.set_xlim(-WINDOW_S, 0)
                pad = 0.1*(yy.max()-yy.min()+1e-9)
                ax_time.set_ylim(yy.min()-pad, yy.max()+pad)

            # FFT
            yv = s[FFT_CH][mask]
            if yv.size >= FFT_MIN:
                L = pow2_le(yv.size)
                sig = yv[-L:] - float(np.mean(yv[-L:]))
                win = np.hanning(L)
                Y = np.fft.rfft(sig*win)
                fs = max(1.0, r.fs)
                freqs = np.fft.rfftfreq(L, d=1.0/fs)
                mag = np.abs(Y)/(L/2.0)
                lfft.set_data(freqs, mag)
                ax_fft.set_xlim(0, fs/2.0)
                ax_fft.set_ylim(0, max(1e-12, float(np.max(mag)))*1.1)

            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
      r.stop_join()
      print("\n[info] bye")

if __name__ == "__main__":
    main()
