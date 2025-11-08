from queue import Queue, Full, Empty
from scipy.signal import spectrogram
from PyQt5.QtCore import QSize, Qt
import matplotlib.pyplot as plt
from PyQt5.QtGui import QColor
from collections import deque
from PyQt5.QtWidgets import (
    QGraphicsDropShadowEffect,    
    QStackedLayout,
    QApplication, 
    QMainWindow,
    QPushButton, 
    QVBoxLayout, 
    QHBoxLayout,
    QSizePolicy,
    QWidget,
    QLabel
)
from PyQt5 import QtCore
import numpy as np
import threading
import pyaudio
import sys
import math
import time


CHANNELS = 1                                    # mono
RATE = 48000                                    # sample per sekund (r)
FFT_SIZE = 2048                                 # størrelse på FFT-vindu (N)
HOP_SIZE = 256                                  # hop størrelse
CHUNK = HOP_SIZE                                # antall prøver per buffer
MAX_FREQ = 8000                                 # maksimal frekvens å analysere
MIN_FREQ = 20                                   # minimal frekvens å analysere
INT16_MAX = 32767                               # maksimal verdi for int16
NOISE = 0.004 * INT16_MAX                       # initial støyterskel
ALPHA = 0.99                                    # glatt faktor
NOISE_MULTIPLIER = 2                            # justerbar multiplikator for støyterskel
MINIMUM_GUI_SIZE = (1500, 1000)                 # fast størrelse på GUI
FONT_SIZE = 10                                  # skriftstørrelse for labels
EXCLUSION_BINS = 3                              # 2–4 er bra for Hann-vindu (undertrykk nabo-binner)
PADDING_FACTOR = 4                              # zero-padding faktor for FFT
STORE_LAST_SECONDS = 5                          # lagre siste N sekunder for plotting
DELAY = 0.01                                    # forsinkelse i sekunder for registering av noter


class AudioRecorderProducer(threading.Thread):
    def __init__(self, queue, chunk=CHUNK, channels=CHANNELS, rate=RATE):
        super().__init__(daemon=True)
        self.queue = queue
        self.chunk = chunk
        self.channels = channels
        self.rate = rate
        self._stop_producer = False
        self._pause = False
        self._wake_event = threading.Event()
        self._pa = None
        self._stream = None

    # --- PyAudio callback: blir kalt i PortAudio-tråd ---
    def _cb(self, in_data, frame_count, time_info, status):
        # Ikke blokker her! (callback må være superrask)
        if not self._pause and not self._stop_producer:
            arr = np.frombuffer(in_data, dtype=np.int16).copy()  # kopier ut av PA-buffer
            try:
                self.queue.put_nowait(arr)
            except Full:
                # dropp eldste for å holde lav latens
                try:
                    _ = self.queue.get_nowait()
                except Empty:
                    pass
                try:
                    self.queue.put_nowait(arr)
                except Full:
                    pass
        # fortsett streamen
        return (None, pyaudio.paContinue)

    def run(self):
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,   # du kan sette = HOP_SIZE for litt lavere latens
            stream_callback=self._cb
        )
        self._stream.start_stream()

        try:
            # hold tråden i live til vi skal stoppe
            while not self._stop_producer:
                if self._pause:
                    # sov litt mens vi er pauset (callback kjører fortsatt, men dropper data)
                    self._wake_event.wait(timeout=0.05)
                else:
                    time.sleep(0.05)
        finally:
            # rydd opp pent
            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                except Exception:
                    pass
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                self._pa = None

            # signaliser til consumer at vi er ferdige
            try:
                self.queue.put_nowait(None)
            except Full:
                pass

    def start(self):
        self._stop_producer = False
        return super().start()

    def pause(self):
        self._pause = True
        # ikke clear() her—callback sjekker bare flagget

    def unpause(self):
        self._pause = False
        self._wake_event.set()

    def stop(self):
        self._stop_producer = True
        self._wake_event.set()

class AnalyzerSignals(QtCore.QObject):
    peaks = QtCore.pyqtSignal(list) 
    highlight = QtCore.pyqtSignal(str)  
              
class AudioAnalyzerConsumer(threading.Thread):

    def __init__(self, queue, my_window=None):
        super().__init__(daemon=True)
        self.queue = queue
        self.chuncks = deque()
        self.total = 0
        self.winbuff = np.empty(FFT_SIZE, dtype=np.float32)
        self.window = np.hanning(FFT_SIZE).astype(np.float32)
        self.win_rms2 = np.mean(self.window**2)
        self.max_k = np.floor(MAX_FREQ / (RATE / FFT_SIZE)).astype(int)
        self.min_k = np.ceil(MIN_FREQ / (RATE / FFT_SIZE)).astype(int)
        self.noise = float(NOISE**2)  # initial støyterskel i effekt
        self.alpha = ALPHA
        self.noise_multiplier = NOISE_MULTIPLIER
        self.last_note = None
        self.last_print = 0 
        self.my_window = my_window
        self._stop_consumer = False
        self._pause = False
        self._wake_event = threading.Event()
        self.mags = np.zeros(self.max_k + 1, dtype=np.float32)
        self.last_wind_data = None
        self.M = FFT_SIZE * PADDING_FACTOR
        self.last_time_data = None 
        self.last_s_data = deque(maxlen=STORE_LAST_SECONDS * RATE // HOP_SIZE)

    def run(self):               
        
        while not self._stop_consumer:
            if self._pause:
                self._wake_event.clear()
                self._wake_event.wait()  # vent til vi blir vekket

            item = self.queue.get()
            if item is None:
                break # no more data to process

            self.append_chunk(item.astype(np.float32))

            while self.total >= FFT_SIZE:
                data = self.build_window()
                data_windowed = data * self.window
                self.consume_left(HOP_SIZE)
                
                rms = float(np.mean(data_windowed**2) / self.win_rms2)

                if rms < (self.noise_multiplier**2) * self.noise:
                    self.noise = self.alpha * self.noise + (1 - self.alpha) * rms

                RMS_THRESHOLD =  (self.noise_multiplier**2) * self.noise 

                if rms < RMS_THRESHOLD:
                    continue # skip lav effekts rammer 

                freq_domain = np.fft.rfft(data_windowed, n=FFT_SIZE)
                self.mags = np.abs(freq_domain)
                mags = self.mags[:self.max_k + 1]

                kmax = int(min(self.max_k, len(mags) - 1))
                if kmax <= self.min_k + 1:
                    continue # ikke interessant

                k_top10 = self.pick_peaks_nms(mags, self.min_k, kmax, K=10, exclusion=EXCLUSION_BINS)
                if k_top10.size == 0:
                    continue

                # Kvadratisk interpolasjon for frekvensestimat
                delta_k_top_10 = np.array([self.quad_interpolate(mags, k) for k in k_top10], dtype=np.float32)
                freq = delta_k_top_10 * (RATE / FFT_SIZE)

                order = np.argsort(mags[k_top10])[::-1]
                k_top10 = k_top10[order]
                freq = freq[order]

                n = min(len(self.my_window.labels), len(freq), len(k_top10))
                items = [(float(freq[i]), float(mags[k_top10[i]])) for i in range(n)]

                now = time.time()
                if not hasattr(self, "ui_last_emit"):
                    self.ui_last_emit = 0.0
                if not hasattr(self, "ui_last_highlight_emit"):
                    self.ui_last_highlight_emit = 0.0

                if now - self.ui_last_emit >= 0.05:
                    self.ui_last_emit = now
                    self.my_window.signals.peaks.emit(items)

                note_to_emit = ""
                if len(freq) > 0:
                    note_name = self.freq_to_note(freq[0])[0]
                    if isinstance(note_name, str):
                        note_to_emit = note_name

                if now - self.ui_last_highlight_emit >= DELAY:
                    self.ui_last_highlight_emit = now
                    self.my_window.signals.highlight.emit(note_to_emit)

                # --- lagre siste rammer til plotting ---
                self.last_wind_data = data_windowed.copy()
                self.last_time_data = data.copy()
                self.last_s_data.append(data[:HOP_SIZE].copy())

    def freq_to_note(self, freq, a4=440.0, prefer_sharps=True):
        note_names_sharp = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        note_names_flat  = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
        if not math.isfinite(freq) or freq <= 0:
            return float('nan'), float('nan'), float('nan'), float('nan')

        # MIDI note nummer
        n = 69 + 12 * math.log2(freq / a4)
        n_round = int(round(n))                     # nærmeste MIDI-note
        cents = 100.0 * (n - n_round)               # avvik i cents

        names = note_names_sharp if prefer_sharps else note_names_flat
        note_name = f"{names[n_round % 12]}{(n_round // 12) - 1}"

        # Ideell frekvens for denne noten
        note_freq = a4 * (2 ** ((n_round - 69) / 12))
        error_hz = freq - note_freq

        return note_name, cents, note_freq, error_hz
    
    def pick_peaks_nms(self, mags, k_min, k_max, K=10, exclusion=EXCLUSION_BINS):
        """Velg maks K topper uten nære duplikater (NMS i bin-rom)."""
        region = mags[k_min:k_max]
        if region.size == 0:
            return np.array([], dtype=int)
        # hent mange kandidater, sorter sterkest først
        cand_rel = np.argpartition(region, -K*8)[-K*8:]
        cand = cand_rel + k_min
        cand = cand[np.argsort(mags[cand])[::-1]]

        selected = []
        for k in cand:
            if all(abs(k - s) > exclusion for s in selected):
                selected.append(k)
                if len(selected) == K:
                    break
        return np.array(selected, dtype=int)
    
    def append_chunk(self, chunk):
        self.chuncks.append(chunk)
        self.total += len(chunk)
        
    def build_window(self):
        filled = 0
        for c in self.chuncks:
            take = min(len(c), FFT_SIZE - filled)
            self.winbuff[filled:filled+take] = c[:take]
            filled += take
            if filled == FFT_SIZE:
                break
        
        return self.winbuff.copy()
    
    def consume_left(self, n: int):
        while n > 0 and self.chuncks:
            c = self.chuncks[0]
            if len(c) <= n:
                n -= len(c)
                self.total -= len(c)
                self.chuncks.popleft()
            else:
                self.chuncks[0] = c[n:]
                self.total -= n
                n = 0
        
    def quad_interpolate(self, mags, k):
        if k <= 0 or k >= len(mags) - 1:
            return 0  # Kan ikke interpolere ved kantene
        m_b = mags[k - 1]
        m_m = mags[k]
        m_n = mags[k + 1]
        denominator = (m_b - 2 * m_m + m_n)
        if denominator == 0:
            return 0  # Unngå deling på null
            
        delta = 0.5 * (m_b - m_n) / denominator
        return k + delta

    def pause(self):
        self._pause = True

    def unpause(self):
        self._pause = False
        self._wake_event.set()

    def stop(self):
        self._stop_consumer = True

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pitch Perfect - Audio Visualizer")
        self.setStyleSheet("background-color: black;")  # Mørk bakgrunnsfarge
        self.current_note = None
        
        self.signals = AnalyzerSignals()
        self.signals.peaks.connect(self.on_peaks)
        self.signals.highlight.connect(self.on_highlight)

        self.plotButton = QPushButton("Plot Frequency Spectrum of the last FFT Frame")
        self.plotButton2 = QPushButton(f"Plot Spectrogram of the last {STORE_LAST_SECONDS}s sampled data")
        self.plotButton3 = QPushButton("Plot Time Domain of the last FFT Frame (Non-Windowed)")
        self.plotButton4 = QPushButton("Plot Time Domain of the last FFT Frame (Windowed)")
        self.button_unpause = QPushButton("Start Audio Processing")
        self.button_pause = QPushButton("Stop Audio Processing")
        self.plotButton5 = QPushButton(f"Plot Last {STORE_LAST_SECONDS}s Time Domain (non-windowed)")
        self.plotButton6 = QPushButton(f"Plot Last {STORE_LAST_SECONDS}s Time Domain (windowed)")
        self.plotButton7 = QPushButton(f"Plot Last {STORE_LAST_SECONDS}s FFT (windowed)")

        self.button_unpause.clicked.connect(self.unpause_audio_processing)
        self.button_pause.clicked.connect(self.pause_audio_processing)    
        self.plotButton.clicked.connect(self.plotLastFFT)
        self.plotButton2.clicked.connect(self.plotLastSpectrogramSeconds)
        self.plotButton3.clicked.connect(self.plotLastTimeDomainNonWindowed)
        self.plotButton4.clicked.connect(self.plotLastTimeDomainWindowed)
        self.plotButton5.clicked.connect(self.plotLastTimeDomainNonWindowedSeconds)
        self.plotButton6.clicked.connect(self.plotLastTimeDomainWindowedSeconds)
        self.plotButton7.clicked.connect(self.plotLastFFTWindowedSeconds)
        self.figs = {}

        buttons = [
            self.button_unpause,
            self.button_pause,
            self.plotButton,
            self.plotButton3,
            self.plotButton4
        ]

        buttons5s = [
            self.plotButton2,
            self.plotButton5,
            self.plotButton6,
            self.plotButton7
        ]

        self.button_unpause.setEnabled(True)
        self.button_pause.setEnabled(False)
        self.plotButton.setEnabled(False)
        self.plotButton2.setEnabled(False)
        self.plotButton3.setEnabled(False)
        self.plotButton4.setEnabled(False)
        self.plotButton5.setEnabled(False)
        self.plotButton6.setEnabled(False)
        self.plotButton7.setEnabled(False)

        button_size = QSize(100, 50)
        for button in buttons:
            shadow_effect = QGraphicsDropShadowEffect()
            shadow_effect.setBlurRadius(15.0)
            shadow_effect.setColor(QColor(255, 255, 255, 140))
            shadow_effect.setOffset(5.0, 5.0)
            button.setMinimumSize(button_size)
            button.setGraphicsEffect(shadow_effect)
            button.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    text-align: left; 
                    font-weight: bold;
                    padding: 10px; 
                    color: black;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #6a097d;
                    color: white;
                }
                
                QPushButton:disabled {
                    background-color: rgba(255, 255, 255, 130);
                    color: gray;
                }
            """)

        for i, button in enumerate(buttons5s):  
            shadow_effect = QGraphicsDropShadowEffect()
            shadow_effect.setBlurRadius(15.0)
            shadow_effect.setColor(QColor(255, 255, 255, 140))
            shadow_effect.setOffset(5.0, 5.0)
            button.setMinimumSize(QSize(300, 30))
            button.setGraphicsEffect(shadow_effect)
            button.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    text-align: center; 
                    font-weight: bold;
                    padding: 10px; 
                    color: black;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #6a097d;
                    color: white;
                }
                
                QPushButton:disabled {
                    background-color: rgba(255, 255, 255, 130);
                    color: gray;
                }
            """)

        self.labels = [QLabel(f"{i}: N/A") for i in range(1, 11)]
        font = self.labels[0].font()
        font.setPointSize(FONT_SIZE)

        layoutH1 = QHBoxLayout()
        for i in range(len(self.labels)//2):
            label = self.labels[i]
            label.setStyleSheet("color: white;")
            label.setFont(font)
            layoutH1.addWidget(label)

        layoutH2 = QHBoxLayout()
        for i in range(len(self.labels)//2, len(self.labels)):
            label = self.labels[i]
            label.setStyleSheet("color: white;")
            label.setFont(font)
            layoutH2.addWidget(label)


        self.whitekeys = []
        for i in range(52):
            key = QWidget()
            key.setStyleSheet("background-color: white; border: 1px solid black;")
            key.setFixedSize(QSize(20, 200))
            self.whitekeys.append(key)

        self.blackkeys = []
        for i in range(36):
            key = QWidget()
            key.setStyleSheet("background-color: black; border: 1px solid black;")
            key.setFixedSize(QSize(15, 120))
            self.blackkeys.append(key)

        # Build mapping from musical note names (e.g., "A4", "C#5") to the corresponding QWidget key
        # The 88-key piano range is MIDI 21 (A0) to 108 (C8). We'll construct names using sharps.
        def midi_to_name(m):
            pcs = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            pc = pcs[m % 12]
            octv = m // 12 - 1
            return f"{pc}{octv}"

        note_names = [midi_to_name(m) for m in range(21, 109)]  # A0..C8
        note_to_widget = {}
        w_i = 0  # hvit index
        b_i = 0  # svart index
        for name in note_names:
            if "#" in name:
                note_to_widget[name] = self.blackkeys[b_i]
                b_i += 1
            else:
                note_to_widget[name] = self.whitekeys[w_i]
                w_i += 1

        self.flat_to_sharp = {"Db":"C#","Eb":"D#","Gb":"F#","Ab":"G#","Bb":"A#"}
        for name in list(note_to_widget.keys()):
            if "#" in name:
                base = name[:-1]  # f.eks. A#
                octv = name[-1]
                for fl, sh in self.flat_to_sharp.items():
                    if base == sh:
                        note_to_widget[f"{fl}{octv}"] = note_to_widget[name]

        self.desiredbox = note_to_widget

        # default
        self._default_white_style = "background-color: white; border: 1px solid black;"
        self._default_black_style = "background-color: black; border: 1px solid black;"

        # bunn lag hvite taster
        layout_white = QHBoxLayout()
        layout_white.setContentsMargins(0, 0, 0, 0)
        layout_white.setSpacing(0)
        for key in self.whitekeys:
            layout_white.addWidget(key, alignment=Qt.AlignLeft | Qt.AlignTop)
        layout_white.addStretch(1)

        white_layer = QWidget()
        white_layer.setLayout(layout_white)

    # tip lag svart lag over hvit
        black_layer = QWidget()
        black_layer.setAttribute(Qt.WA_StyledBackground, True)
        black_layer.setStyleSheet("background: transparent;")

        white_w = 20
        black_w = 15

        whites_in_order = [n for n in note_names if "#" not in n]
        # svart existerer etter hvit hvis hvit ikke er E eller B
        has_black_after_white_flags = [w[0] not in ("E", "B") for w in whites_in_order[:-1]]

        positions = [i for i, flag in enumerate(has_black_after_white_flags) if flag]
        positions = positions[:len(self.blackkeys)]

        for key, i_between in zip(self.blackkeys, positions):
            key.setParent(black_layer)
            x = int((i_between + 1) * white_w - black_w / 2)
            key.move(x, 0)

        stacked = QStackedLayout()
        stacked.setStackingMode(QStackedLayout.StackAll)
        stacked.addWidget(white_layer)
        stacked.addWidget(black_layer)
        stacked.setCurrentWidget(black_layer) 

        center = QWidget()
        center.setLayout(stacked)
        center.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        layoutH3 = QHBoxLayout()
        layoutH3.addWidget(center, alignment=Qt.AlignCenter)

        self.NoteLabel = QLabel("N/A")
        notefont = self.NoteLabel.font()
        notefont.setPointSize(30)
        self.NoteLabel.setStyleSheet("color: white;")
        self.NoteLabel.setFont(notefont)
        self.NoteLabel.setAlignment(Qt.AlignCenter)
        
        
        layoutV1_1 = QVBoxLayout()
        layoutV1_1.addWidget(self.button_unpause)
        layoutV1_1.addWidget(self.button_pause)
        
        layoutH1_1 = QHBoxLayout()
        layoutH1_1.addWidget(self.plotButton)
        layoutH1_1.addWidget(self.plotButton3)
        layoutH1_1.addWidget(self.plotButton4)

        centerV1 = QWidget()
        centerV1.setLayout(layoutV1_1)
        centerV1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        layoutV1 = QHBoxLayout()
        layoutV1.addWidget(centerV1, alignment=Qt.AlignCenter)


        layoutV1 = QVBoxLayout()
        layoutV1.addWidget(centerV1, alignment=Qt.AlignCenter)

        layoutButton2 = QVBoxLayout()
        layoutButton2.addWidget(self.plotButton2)
        layoutButton2.addWidget(self.plotButton5)
        layoutButton2.addWidget(self.plotButton6)
        layoutButton2.addWidget(self.plotButton7)

        layoutButton2.setContentsMargins(0, 0, 0, 20)
        container = QWidget()
        layoutV = QVBoxLayout(container)
        layoutV.addLayout(layoutH1)
        layoutV.addLayout(layoutH2)
        layoutV.addWidget(self.NoteLabel)
        layoutV.addLayout(layoutH3)
        layoutV.addLayout(layoutV1)
        layoutV.addLayout(layoutH1_1)
        layoutV.addLayout(layoutButton2)

        container.setLayout(layoutV)
        self.setCentralWidget(container)
        self.setMinimumSize(QSize(*MINIMUM_GUI_SIZE))
        
        self.init_audio_processing()
    
    def init_audio_processing(self):
        print("Audio processing started...")
        self.queue = Queue(maxsize=31)
        self.producer = AudioRecorderProducer(self.queue)
        self.consumer = AudioAnalyzerConsumer(self.queue, my_window=self)
        self.producer.start()
        self.consumer.start()
        self.pause_audio_processing()
        self.plotButton.setEnabled(False)
        self.plotButton2.setEnabled(False)
        self.plotButton3.setEnabled(False)
        self.plotButton4.setEnabled(False)
        self.plotButton5.setEnabled(False)
        self.plotButton6.setEnabled(False)
        self.plotButton7.setEnabled(False)

    def set_note_color(self, note: str, color: str):
            n = note.strip().upper().replace('B', 'B') 
            for fl, sh in self.flat_to_sharp.items():
                n = n.replace(fl, sh)
            w = self.desiredbox.get(n)
            if not w:
                return
            w.setStyleSheet(f"background-color: {color}; border: 1px solid black;")
    
    def start_audio_processing(self):
        self.button_start.setEnabled(False)
        self.button_stop.setEnabled(True)
        self.producer.start()
        self.consumer.start()

    def pause_audio_processing(self):
        self.button_unpause.setEnabled(True)
        self.button_pause.setEnabled(False)
        self.plotButton.setEnabled(True)
        self.plotButton2.setEnabled(True)
        self.plotButton3.setEnabled(True)
        self.plotButton4.setEnabled(True)
        self.plotButton5.setEnabled(True)
        self.plotButton6.setEnabled(True)
        self.plotButton7.setEnabled(True)

        if hasattr(self, 'producer'):
            self.producer.pause()
        if hasattr(self, 'consumer'):
            self.consumer.pause()

    def unpause_audio_processing(self):
        self.button_unpause.setEnabled(False)
        self.button_pause.setEnabled(True)
        
        for fig in self.figs.values():
            plt.close(fig)
            
        self.plotButton.setEnabled(False)
        self.plotButton2.setEnabled(False)
        self.plotButton3.setEnabled(False)
        self.plotButton4.setEnabled(False)
        self.plotButton5.setEnabled(False)
        self.plotButton6.setEnabled(False)
        self.plotButton7.setEnabled(False)

        if hasattr(self, 'producer'):
            self.producer.unpause()
        if hasattr(self, 'consumer'):
            self.consumer.unpause()

    def stop_audio_processing(self):
        self.button_unpause.setEnabled(True)
        self.button_pause.setEnabled(False)
        for fig in self.figs.values():
            plt.close(fig)
        if hasattr(self, 'producer'):
            self.producer.stop()
        if hasattr(self, 'consumer'):
            self.consumer.stop()

    def closeEvent(self, event):
        self.stop_audio_processing()
        event.accept()
        
    def plotLastFFT(self):
        self.figs["fft"] = plt.figure("Frequency Spectrum", figsize=(10, 6))
        def on_close(event):
            self.plotButton.setEnabled(True)

        self.figs["fft"].canvas.mpl_connect('close_event', on_close)

        ax = self.figs["fft"].add_subplot(1, 1, 1)

        self.plotButton.setEnabled(False)
        fft = np.fft.rfft(self.consumer.last_wind_data, n=self.consumer.M)
        freqs_full = np.fft.rfftfreq(self.consumer.M, d=1.0 / RATE)
        fft_mags = np.abs(fft)

        min_k_plot = int(np.searchsorted(freqs_full, MIN_FREQ, side='left'))
        k_max_plot = int(np.searchsorted(freqs_full, MAX_FREQ, side='right')) - 1
        k_max_plot = max(min_k_plot, min(k_max_plot, len(freqs_full) - 1, len(fft_mags) - 1))
        
        if k_max_plot < 1:
            return  # nothing meaningful to plot

        freqs = freqs_full[min_k_plot: k_max_plot + 1]
        plot_mags = fft_mags[min_k_plot: k_max_plot + 1]

        ax.plot(freqs, plot_mags)
        ax.set_title("Frequency Spectrum of Last FFT Frame")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.grid(False)
        plt.show(block=False)

    def plotLastSpectrogramSeconds(self):
        if len(self.consumer.last_s_data) > 0:
            x = np.concatenate(list(self.consumer.last_s_data))
            target = STORE_LAST_SECONDS * RATE
            if x.size > target:
                x = x[-target:]
        else:
            x = self.consumer.last_time_data
            if x is None:
                return

        self.figs["spectrogram"] = plt.figure("Spectrogram", figsize=(10, 6))
        fig = self.figs["spectrogram"]
        def on_close(event):
            self.plotButton2.setEnabled(True)

        fig.canvas.mpl_connect('close_event', on_close)
        ax = fig.add_subplot(1, 1, 1)
        
        self.plotButton2.setEnabled(False)
        f, t, S = spectrogram(
            x,
            fs=RATE,
            window='hann',
            nperseg=FFT_SIZE,
            noverlap=FFT_SIZE - HOP_SIZE,
            nfft=len(x),
            mode='magnitude',         # => 20*log10
            detrend=False
        )
        # Klipp til frekvensområde
        S_db = 20.0 * np.log10(np.maximum(S, 1e-20))
        sel = (f >= MIN_FREQ) & (f <= MAX_FREQ)
        f = f[sel]; S_db = S_db[sel, :]

        # Hvis bare 1 kolonne: repliker og bytt shading
        if S_db.shape[1] == 1:
            T = len(x) / RATE
            S_db = np.repeat(S_db, 2, axis=1)
            t = np.array([0, T])
            shading = 'nearest'
        else:
            shading = 'auto'  

        pc = ax.pcolormesh(t, f, S_db, shading=shading)
        fig.colorbar(pc, ax=ax, label="dB (rel.)")
        ax.set_title(f"Spectrogram (last <{STORE_LAST_SECONDS}s sampled data)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(MIN_FREQ, MAX_FREQ)
        ax.grid(False)
        plt.show(block=False)

    def plotLastTimeDomainNonWindowed(self):
        self.figs["time_domain_non_windowed"] = plt.figure("Time Domain (Non-Windowed)", figsize=(10, 6))
        def on_close(event):
            self.plotButton3.setEnabled(True)

        self.figs["time_domain_non_windowed"].canvas.mpl_connect('close_event', on_close)

        ax = self.figs["time_domain_non_windowed"].add_subplot(1, 1, 1)

        self.plotButton3.setEnabled(False)
        t = np.arange(len(self.consumer.last_time_data)) / RATE
        ax.plot(t, self.consumer.last_time_data, color="green")
        ax.set_title("Time Domain of Last FFT Frame (Non-Windowed)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(False)
        plt.show(block=False)
        
    def plotLastTimeDomainWindowed(self):

        self.figs["time_domain_windowed"] = plt.figure("Time Domain (Windowed)", figsize=(10, 6))
        def on_close(event):
            self.plotButton4.setEnabled(True)
        self.figs["time_domain_windowed"].canvas.mpl_connect('close_event', on_close)

        ax = self.figs["time_domain_windowed"].add_subplot(1, 1, 1)

        self.plotButton4.setEnabled(False)
        t = np.arange(len(self.consumer.last_wind_data)) / RATE
        ax.plot(t, self.consumer.last_wind_data, color="red")
        ax.set_title("Time Domain of Last FFT Frame (Windowed)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(False)
        plt.show(block=False)

    def plotLastTimeDomainNonWindowedSeconds(self):
        if len(self.consumer.last_s_data) > 0:
            x = np.concatenate(list(self.consumer.last_s_data))
            target = STORE_LAST_SECONDS * RATE
            if x.size > target:
                x = x[-target:]
        else:
            x = self.consumer.last_time_data
            if x is None:
                return
        
        self.figs["time_domain_non_windowed_5s"] = plt.figure("Time Domain (Non-Windowed, 5s)", figsize=(10, 6))
        def on_close(event):
            self.plotButton5.setEnabled(True)
        self.figs["time_domain_non_windowed_5s"].canvas.mpl_connect('close_event', on_close)

        self.plotButton5.setEnabled(False)
        ax = self.figs["time_domain_non_windowed_5s"].add_subplot(1, 1, 1)

        t = np.arange(len(x)) / RATE
        ax.plot(t, x, color="green")
        ax.set_title(f"Time Domain of Last <{STORE_LAST_SECONDS}s (Non-Windowed)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(False)
        plt.show(block=False)

    def plotLastTimeDomainWindowedSeconds(self):
        if len(self.consumer.last_s_data) > 0:
            x = np.concatenate(list(self.consumer.last_s_data))
            target = STORE_LAST_SECONDS * RATE
            if x.size > target:
                x = x[-target:]
        else:
            x = self.consumer.last_time_data
            if x is None:
                return
        
        x_windowed = x * np.hanning(len(x))

        self.figs["time_domain_windowed_5s"] = plt.figure("Time Domain (Windowed, 5s)", figsize=(10, 6))
        def on_close(event):
            self.plotButton6.setEnabled(True)
        self.figs["time_domain_windowed_5s"].canvas.mpl_connect('close_event', on_close)

        self.plotButton6.setEnabled(False)
        ax = self.figs["time_domain_windowed_5s"].add_subplot(1, 1, 1)
        t = np.arange(len(x_windowed)) / RATE
        ax.plot(t, x_windowed, color="red")
        ax.set_title(f"Time Domain of last <{STORE_LAST_SECONDS}s (Windowed)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(False)
        plt.show(block=False)

    def plotLastFFTWindowedSeconds(self):
        if len(self.consumer.last_s_data) > 0:
            x = np.concatenate(list(self.consumer.last_s_data))
            target = STORE_LAST_SECONDS * RATE
            if x.size > target:
                x = x[-target:]
        else:
            x = self.consumer.last_time_data
            if x is None:
                return

        x_windowed = x * np.hanning(len(x))

        self.figs["fft_windowed_5s"] = plt.figure("FFT (Windowed, 5s)", figsize=(10, 6))
        def on_close(event):
            self.plotButton7.setEnabled(True)
        self.figs["fft_windowed_5s"].canvas.mpl_connect('close_event', on_close)

        self.plotButton7.setEnabled(False)
        ax = self.figs["fft_windowed_5s"].add_subplot(1, 1, 1)
        f = np.fft.rfftfreq(len(x_windowed), d=1/RATE)
        X = np.fft.rfft(x_windowed)
        ax.plot(f, np.abs(X), color="blue")
        ax.set_title(f"FFT of last <{STORE_LAST_SECONDS}s (Windowed)")
        ax.set_xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        ax.grid(False)
        plt.show(block=False)

    @QtCore.pyqtSlot(list)
    def on_peaks(self, items):
        n = min(len(self.labels), len(items))
        for i in range(n):
            f_i, mag_i = items[i]
            name, cents, note_f, err = self.consumer.freq_to_note(f_i)
            self.labels[i].setText(
                f"Top {i + 1}:\n\t Note: {name}  \n\t Cents: {cents:.2f} "
                f"\n\t Error: {err:.2f} Hz \n\t Ideell Freq: {note_f:.2f} Hz "
                f"\n\t Actual Freq: {f_i:.2f} Hz \n\t Magnitude: {mag_i:.2f}"
            )
        for j in range(n, len(self.labels)):
            self.labels[j].setText(f"{j + 1}: N/A")

    @QtCore.pyqtSlot(str)
    def on_highlight(self, note):
        if self.current_note:
            w_prev = self.desiredbox.get(self.current_note)
            if w_prev:
                if "#" in self.current_note:
                    w_prev.setStyleSheet(self._default_black_style)
                else:
                    w_prev.setStyleSheet(self._default_white_style)

        self.current_note = note if note else None
        self.NoteLabel.setText(note if note else "N/A")

        # Highlight ny note
        if note:
            w = self.desiredbox.get(note)
            if w:
                w.setStyleSheet("background-color: red; border: 1px solid black;")


def main():
    app = QApplication(sys.argv)
    my_window = MyWindow() 
    my_window.show()
    app.exec()

if __name__ == "__main__":
    main()