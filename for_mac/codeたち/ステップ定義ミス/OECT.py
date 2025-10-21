import threading
import time
from queue import Queue, Empty
from dataclasses import dataclass
import pyvisa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod
from enum import Enum
from collections import deque
import datetime

filename = "test"

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y.%m.%d-%H%M%S")
formatted_time_and_name = formatted_time + filename

# ============================================
# 測定モード定義
# ============================================
class MeasurementMode(Enum):
    """測定モード"""
    LINEAR = "linear"
    OUTPUT = "output"
    SATURATION = "saturation"

# ============================================
# リアルタイムプロッター追加
# ============================================
class RealtimePlotter:
    """リアルタイムデータプロット"""
    
    def __init__(self, mode: MeasurementMode, max_points: int = 1000):
        self.mode = mode
        self.max_points = max_points
        
        self.times = deque(maxlen=max_points)
        self.vref_data = deque(maxlen=max_points)
        self.isd_data = deque(maxlen=max_points)
        self.vg_data = deque(maxlen=max_points)
        
        self.lock = threading.Lock()
        
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 8))
        self.fig.suptitle(f'Real-time Measurement - {mode.value.upper()} Mode')
        
        self.axes[0].set_ylabel('Vref (mV)')
        self.axes[0].grid(True)
        self.line_vref, = self.axes[0].plot([], [], 'b-', linewidth=1.5)
        
        self.axes[1].set_ylabel('Drain Current (A)')
        self.axes[1].grid(True)
        self.axes[1].set_yscale('log')
        self.line_isd, = self.axes[1].plot([], [], 'r-', linewidth=1.5)
        
        self.axes[2].set_xlabel('Time (s)')
        self.axes[2].set_ylabel('Gate Voltage (V)')
        self.axes[2].grid(True)
        self.line_vg, = self.axes[2].plot([], [], 'g-', linewidth=1.5)
        
        plt.tight_layout()
        
        self.ani = FuncAnimation(
            self.fig, 
            self._update_plot, 
            interval=100,
            blit=False,
            cache_frame_data=False
        )
        
        plt.ion()
        plt.show(block=False)
    
    def update(self, time_val: float, vref: float, isd: float, vg: float):
        """データ更新(スレッドセーフ)"""
        with self.lock:
            self.times.append(time_val)
            self.vref_data.append(vref * 1000)
            self.isd_data.append(abs(isd) if isd != 0 else 1e-12)
            self.vg_data.append(vg)
    
    def _update_plot(self, frame):
        """プロット更新(アニメーションコールバック)"""
        with self.lock:
            if len(self.times) == 0:
                return
            
            times_list = list(self.times)
            vref_list = list(self.vref_data)
            isd_list = list(self.isd_data)
            vg_list = list(self.vg_data)
        
        self.line_vref.set_data(times_list, vref_list)
        self.axes[0].relim()
        self.axes[0].autoscale_view()
        
        self.line_isd.set_data(times_list, isd_list)
        self.axes[1].relim()
        self.axes[1].autoscale_view()
        
        self.line_vg.set_data(times_list, vg_list)
        self.axes[2].relim()
        self.axes[2].autoscale_view()
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def close(self):
        """グラフを閉じる"""
        if self.ani:
            self.ani.event_source.stop()
        plt.close(self.fig)

# ============================================
# 測定パラメータデータクラス
# ============================================
@dataclass
class SaturationParams:
    """Saturationモードのパラメータ"""
    vsd_sat: float
    cycle_num: int
    vg_sat_initial: float
    vref_sat_initial: float
    vref_sat_max_change: float
    vg_sat_step: float
    sat_wait_time: float
    data_interval: float = 1.0

# ============================================
# 抽象基底クラス(接続リトライ機能付き)
# ============================================
class InstrumentBase(ABC):
    """測定機器の基底クラス(接続リトライ機能付き)"""
    
    def __init__(self, rm: pyvisa.ResourceManager, ip: str, timeout: int = 5000, retries: int = 3):
        self.rm = rm
        self.ip = ip
        self.timeout = timeout
        self.retries = retries
        self.inst = None
        self.instrument_name = self.__class__.__name__
        self.lock = threading.Lock()  # 機器操作用ロック
    
    def connect(self):
        """機器に接続(リトライ機能付き)"""
        resource = f"TCPIP0::{self.ip}::inst0::INSTR"
        
        for attempt in range(self.retries):
            try:
                if self.inst:
                    try:
                        self.inst.close()
                    except Exception:
                        pass
                    self.inst = None
                
                print(f"[{self.instrument_name}] Connecting to {self.ip} (Attempt {attempt + 1}/{self.retries})...")
                self.inst = self.rm.open_resource(resource)
                self.inst.timeout = self.timeout
                self.inst.read_termination = "\n"
                self.inst.write_termination = "\n"
                
                time.sleep(0.5)
                
                self.inst.write("*CLS")
                time.sleep(0.2)
                idn = self.inst.query("*IDN?")
                print(f"[{self.instrument_name}] Connected successfully: {idn.strip()}")
                return True
                
            except Exception as e:
                print(f"[{self.instrument_name}] Connection attempt {attempt + 1} failed: {e}")
                
                if self.inst:
                    try:
                        self.inst.close()
                    except Exception:
                        pass
                    self.inst = None
                
                if attempt < self.retries - 1:
                    print(f"[{self.instrument_name}] Waiting 3 seconds before retry...")
                    time.sleep(3)
                else:
                    print(f"[{self.instrument_name}] Failed to connect after {self.retries} attempts.")
                    raise RuntimeError(f"Could not connect to {self.instrument_name} at {self.ip}")
        
        return False
    
    @abstractmethod
    def configure(self):
        """機器の設定(サブクラスで実装)"""
        pass
    
    def close(self):
        """接続を閉じる"""
        if self.inst is not None:
            try:
                print(f"[{self.instrument_name}] Closing connection...")
                self.inst.close()
                print(f"[{self.instrument_name}] Connection closed.")
            except Exception as e:
                print(f"[{self.instrument_name}] Error closing connection: {e}")
            finally:
                self.inst = None

# ============================================
# DMM6500クラス (Vref測定用)
# ============================================
class DMM6500(InstrumentBase):
    """KEITHLEY DMM6500 Digital Multimeter制御クラス"""
    
    def __init__(self, rm: pyvisa.ResourceManager, ip: str, v_range: float = 1.0,
                 nplc: float = 10.0, autozero: str = "ON", timeout: int = 5000, retries: int = 3):
        super().__init__(rm, ip, timeout, retries)
        self.v_range = v_range
        self.nplc = nplc
        self.autozero = autozero
    
    def configure(self):
        if self.inst is None:
            raise RuntimeError("Not connected. Call connect() first.")
        
        print(f"[{self.instrument_name}] Configuring...")
        self.inst.write("*RST")
        self.inst.write("*CLS")
        self.inst.write('SENS:FUNC "VOLT:DC"')
        self.inst.write(f"SENS:VOLT:RANG {self.v_range}")
        self.inst.write(f"SENS:VOLT:NPLC {self.nplc}")
        autozero_cmd = "SENS:VOLT:AZER ON" if self.autozero.upper() == "ON" else "SENS:VOLT:AZER OFF"
        self.inst.write(autozero_cmd)
        self.inst.write(f"SENS:VOLT:INP 10E6")
        self.inst.write("SENS:VOLT:AVER:STAT OFF")
        
        test_reading = self.read_single()
        print(f"[{self.instrument_name}] Configuration complete. Test reading: {test_reading:.6f} V")
    
    def read_single(self) -> float:
        """単一電圧測定(スレッドセーフ)"""
        if self.inst is None:
            raise RuntimeError("Not connected.")
        
        with self.lock:
            result = self.inst.query("READ?")
            return float(result.strip())

# ============================================
# Keithley2450 ドレイン用クラス
# ============================================
class Keithley2450Drain(InstrumentBase):
    """Keithley 2450 SourceMeter (Drain用)"""
    
    def __init__(self, rm: pyvisa.ResourceManager, ip: str, source_voltage: float,
                 compliance_current: float = 0.001, meas_range: float = None,
                 nplc: float = 10.0, autozero: str = "ON", timeout: int = 5000, retries: int = 3):
        super().__init__(rm, ip, timeout, retries)
        self.source_voltage = source_voltage
        self.compliance_current = compliance_current
        self.meas_range = meas_range
        self.nplc = nplc
        self.autozero = autozero
    
    def configure(self):
        if self.inst is None:
            raise RuntimeError("Not connected. Call connect() first.")
        
        print(f"[{self.instrument_name}] Configuring...")
        self.inst.write("*RST")
        self.inst.write("*CLS")
        self.inst.write("SOUR:FUNC VOLT")
        self.inst.write(f"SOUR:VOLT {self.source_voltage}")
        self.inst.write(f"SOUR:VOLT:ILIM {self.compliance_current}")
        self.inst.write("SENS:FUNC 'CURR'")
        
        if self.meas_range is not None:
            self.inst.write(f"SENS:CURR:RANG {self.meas_range}")
        else:
            self.inst.write("SENS:CURR:RANG:AUTO ON")
        
        self.inst.write(f"SENS:CURR:NPLC {self.nplc}")
        self.inst.write(f"SYST:AZER {self.autozero}")
        self.inst.write("OUTP ON")
        
        test_current = self.read_current()
        print(f"[{self.instrument_name}] Configuration complete. Output ON. Test reading: {test_current:.6E} A")
    
    def set_voltage(self, voltage: float):
        """電圧設定(スレッドセーフ)"""
        if self.inst is None:
            raise RuntimeError("Not connected.")
        
        with self.lock:
            self.inst.write(f"SOUR:VOLT {voltage}")
            self.source_voltage = voltage
    
    def read_current(self) -> float:
        """電流読取(スレッドセーフ)"""
        if self.inst is None:
            raise RuntimeError("Not connected.")
        
        with self.lock:
            result = self.inst.query("READ?")
            return float(result.strip().split(",")[0])
    
    def close(self):
        if self.inst is not None:
            try:
                with self.lock:
                    print(f"[{self.instrument_name}] Turning output OFF...")
                    self.inst.write("OUTP OFF")
            except Exception as e:
                print(f"[{self.instrument_name}] Error turning output OFF: {e}")
        super().close()

# ============================================
# Keithley2450 ゲート用クラス
# ============================================
class Keithley2450Gate(InstrumentBase):
    """Keithley 2450 SourceMeter (Gate用)"""
    
    def __init__(self, rm: pyvisa.ResourceManager, ip: str, compliance_current: float,
                 meas_range: float = None, nplc: float = 1.0, autozero: str = "ON",
                 timeout: int = 5000, retries: int = 3):
        super().__init__(rm, ip, timeout, retries)
        self.compliance_current = compliance_current
        self.meas_range = meas_range
        self.nplc = nplc
        self.autozero = autozero
        self.current_voltage = 0.0
    
    def configure(self):
        if self.inst is None:
            raise RuntimeError("Not connected. Call connect() first.")
        
        print(f"[{self.instrument_name}] Configuring...")
        self.inst.write("*RST")
        self.inst.write("*CLS")
        self.inst.write("SOUR:FUNC VOLT")
        self.inst.write(f"SOUR:VOLT:ILIM {self.compliance_current}")
        self.inst.write("SENS:FUNC 'CURR'")
        
        if self.meas_range is not None:
            self.inst.write(f"SENS:CURR:RANG {self.meas_range}")
        else:
            self.inst.write("SENS:CURR:RANG:AUTO ON")
        
        self.inst.write(f"SENS:CURR:NPLC {self.nplc}")
        self.inst.write(f"SYST:AZER {self.autozero}")
        self.inst.write("OUTP ON")
        
        test_current = self.read_current()
        print(f"[{self.instrument_name}] Configuration complete. Output ON. Test reading: {test_current:.6E} A")
    
    def set_voltage(self, voltage: float):
        """電圧設定(スレッドセーフ)"""
        if self.inst is None:
            raise RuntimeError("Not connected.")
        
        with self.lock:
            self.inst.write(f"SOUR:VOLT:LEV {voltage}")
            self.current_voltage = voltage
    
    def read_current(self) -> float:
        """電流読取(スレッドセーフ)"""
        if self.inst is None:
            raise RuntimeError("Not connected.")
        
        with self.lock:
            result = self.inst.query("READ?")
            return float(result.strip().split(",")[0])
    
    def close(self):
        if self.inst is not None:
            try:
                with self.lock:
                    print(f"[{self.instrument_name}] Turning output OFF...")
                    self.inst.write("OUTP OFF")
            except Exception as e:
                print(f"[{self.instrument_name}] Error turning output OFF: {e}")
        super().close()

# ============================================
# 測定制御クラス(3スレッド構成 + リアルタイムプロット)
# ============================================
class SharedState:
    """スレッド間で共有する状態"""
    def __init__(self):
        self.current_vg = 0.0
        self.current_vsd = 0.0
        self.current_cycle = 1
        self.latest_vref = 0.0
        self.vref_history = deque(maxlen=100)
        self.lock = threading.Lock()
    
    def update_voltage(self, vg: float):
        with self.lock:
            self.current_vg = vg
    
    def update_cycle(self, cycle: int):
        with self.lock:
            self.current_cycle = cycle
    
    def update_vref(self, vref: float):
        with self.lock:
            self.latest_vref = vref
            self.vref_history.append(vref)
    
    def get_state(self):
        with self.lock:
            return self.current_vg, self.current_vsd, self.current_cycle
    
    def get_latest_vref(self):
        with self.lock:
            return self.latest_vref
    
    def get_recent_vrefs(self, n: int):
        """最新n個のVrefを取得"""
        with self.lock:
            history_list = list(self.vref_history)
            return history_list[-n:] if len(history_list) >= n else history_list

class MeasurementController:
    """測定制御クラス(3スレッド構成 + リアルタイムプロット)"""
    
    def __init__(self, dmm, drain_smu, gate_smu, filename: str = f"{formatted_time_and_name}=sat.txt",
                 save_columns: List[str] = ['Time', "GateI", "GateV", "DrainI", "Vref"],
                 enable_plot: bool = True):
        self.dmm = dmm
        self.drain_smu = drain_smu
        self.gate_smu = gate_smu
        self.plotter = None
        self.enable_plot = enable_plot
        self.measurement_data = []
        
        self.measurement_thread = None
        self.writer_thread = None
        self.measuring = False
        self.writing = False
        
        self.data_queue = Queue()
        self.shared_state = SharedState()
        
        self.filename = filename
        self.file_handle = None
        
        self.available_columns = ['Time', "GateI", "GateV", "DrainI", "DrainV", "Vref"]
        if save_columns is None:
            self.save_columns = self.available_columns.copy()
        else:
            invalid_cols = [col for col in save_columns if col not in self.available_columns]
            if invalid_cols:
                raise ValueError(f"Invalid column names: {invalid_cols}. Available: {self.available_columns}")
            self.save_columns = save_columns
    
    def _get_column_value(self, data_point: Dict, col_name: str):
        col_map = {
            'Time': data_point['time'],
            'Cycle': data_point['cycle'],
            'Vref': data_point['vref'],
            'GateV': data_point['vg'],
            'DrainI': data_point['isd'],
            'GateI': data_point['ig'],
            'DrainV': data_point['vsd']
        }
        return col_map[col_name]
    
    def _write_data_line(self, data_point: Dict):
        if self.file_handle is None:
            return
        
        try:
            values = []
            for col in self.save_columns:
                val = self._get_column_value(data_point, col)
                if col == 'Cycle':
                    values.append(f"{val}")
                else:
                    values.append(f"{val:.7E}")
            
            self.file_handle.write("\t".join(values) + "\n")
            self.file_handle.flush()
        except Exception as e:
            print(f"Error writing data: {e}")
    
    def _measurement_loop(self, t_start: float, data_interval: float):
        """
        測定スレッド: 一定間隔で測定を実行
        【重要】メインスレッドからの電圧設定や他の操作に影響されない
        
        【測定値の説明】
        - vref: DMM6500から読み取った基準電圧 (V) → SharedStateで常に最新に保持
        - isd: Keithley2450(Drain)から読み取った電流 (A)
        - ig: Keithley2450(Gate)から読み取った電流 (A)
        - vg: SharedStateから取得したゲート電圧 (V) → メインスレッドが設定
        - vsd: SharedStateから取得したドレイン電圧 (V) → 初期値として設定
        - cycle: 現在のサイクル番号 (1, 2, 3, ...)
        
        【データの流れ】
        測定値 → キューに挿入 → WriterThread → ファイル出力 & プロット
        """
        next_measure_time = time.time()
        measurement_count = 0
        
        print(f"[MeasurementThread] Starting continuous measurement at {data_interval}s intervals")
        
        while self.measuring:
            current_time = time.time()
            
            if current_time >= next_measure_time:
                t_measure = current_time - t_start
                measurement_count += 1
                
                try:
                    # 【1】機器から測定値を読取 (スレッドセーフ)
                    vref = self.dmm.read_single()           # DMM: 基準電圧 (V)
                    isd = self.drain_smu.read_current()     # Drain SMU: ドレイン電流 (A)
                    ig = self.gate_smu.read_current()       # Gate SMU: ゲート電流 (A)
                    
                    # 【2】SharedStateから制御値を読取
                    vg, vsd, cycle = self.shared_state.get_state()
                    # vg: メインスレッドが設定したゲート電圧
                    # vsd: ドレイン電圧（通常は固定値）
                    # cycle: 現在のサイクル番号
                    
                    # 【3】Vrefを共有状態に保存（メインスレッドの到達判定用）
                    self.shared_state.update_vref(vref)
                    
                    # 【4】データセットを構築
                    data_point = {
                        'time': t_measure,          # 測定開始からの経過時間 (s)
                        'cycle': cycle,             # サイクル番号
                        'vref': vref,               # 基準電圧 (V)
                        'vg': vg,                   # ゲート電圧 (V)
                        'isd': isd,                 # ドレイン電流 (A)
                        'ig': ig,                   # ゲート電流 (A)
                        'vsd': vsd                  # ドレイン電圧 (V)
                    }
                    
                    # 【5】キューに追加（WriterThreadが処理）
                    self.data_queue.put(data_point)
                    
                    print(f"  [Meas#{measurement_count:04d} t={t_measure:7.1f}s cyc={cycle}] "
                          f"Vref={vref*1000:8.3f}mV, Isd={isd:12.6E}A, Vg={vg:8.4f}V, Ig={ig:12.6E}A")
                    
                except Exception as e:
                    print(f"[MeasurementThread] Measurement error: {e}")
                
                # 次の測定時刻をスケジュール
                next_measure_time += data_interval
            else:
                # 次の測定時刻まで小刻みで待機(無駄なCPU使用を防ぐ)
                sleep_time = next_measure_time - current_time
                time.sleep(min(0.01, sleep_time))
    
    def _writer_loop(self):
        """
        ライタースレッド: キューからデータを取り出してファイルに書き込み
        """
        while self.writing:
            try:
                data_point = self.data_queue.get(timeout=0.5)
                
                self.measurement_data.append(data_point)
                self._write_data_line(data_point)
                
                if self.plotter is not None:
                    self.plotter.update(
                        data_point['time'],
                        data_point['vref'],
                        data_point['isd'],
                        data_point['vg']
                    )
                
                self.data_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                print(f"[WriterThread] Writer error: {e}")
    
    def _check_vref_reached(self, target_vref: float, wait_time: float, data_interval: float,
                            tolerance_percent: float = 0.5) -> Tuple[bool, float, List[float]]:
        """
        wait_time以内でVrefが目標値に到達したか判定
        平均値がtarget_vrefの±tolerance_percent以内なら True
        測定スレッドが既に取得したデータを使用(DMMへの直接アクセスなし)
        
        Args:
            target_vref: 目標Vref値 (V)
            wait_time: 待機時間 (s)
            data_interval: データ取得間隔 (s)
            tolerance_percent: 許容誤差（パーセント）デフォルト 0.5%
        
        Returns:
            (到達判定, 平均Vref, サンプルリスト)
        """
        num_samples = max(int(wait_time / data_interval), 1)
        
        # 許容誤差範囲を計算
        tolerance = abs(target_vref * tolerance_percent / 100.0)
        vref_min = target_vref - tolerance
        vref_max = target_vref + tolerance
        
        print(f"    Target Vref: {target_vref:.6f} V (tolerance: ±{tolerance_percent}% = ±{tolerance:.6f} V)")
        print(f"    Waiting for {num_samples} samples over {wait_time} s...")
        
        start_check_time = time.time()
        vref_samples = []
        
        while time.time() - start_check_time < wait_time:
            time.sleep(data_interval)
            
            vref = self.shared_state.get_latest_vref()
            if vref != 0.0:
                vref_samples.append(vref)
                in_range = "✓" if vref_min <= vref <= vref_max else "✗"
                print(f"    Sample {len(vref_samples):2d}/{num_samples}: Vref = {vref:.6f} V {in_range}")
            
            if len(vref_samples) >= num_samples:
                break
        
        if len(vref_samples) == 0:
            print(f"    ✗ No samples collected")
            return False, 0.0, []
        
        # 平均値を計算
        vref_avg = sum(vref_samples) / len(vref_samples)
        
        # 平均値が許容範囲内か判定
        avg_in_range = vref_min <= vref_avg <= vref_max
        
        # 各サンプルが許容範囲内の割合
        within_range = sum(1 for v in vref_samples if vref_min <= v <= vref_max)
        percentage = within_range / len(vref_samples) * 100
        
        print(f"    Average Vref: {vref_avg:.6f} V {'✓' if avg_in_range else '✗'}")
        print(f"    {within_range}/{len(vref_samples)} samples within range ({percentage:.1f}%)")
        
        # 平均値が目標値の±tolerance_percent以内なら True
        return avg_in_range, vref_avg, vref_samples
    
    def measure_saturation(self, params: SaturationParams):
        print("="*70)
        print("Starting SATURATION mode measurement (3-thread + Real-time Plot)")
        print(f"Data acquisition interval: {params.data_interval} s")
        print(f"Output file: {self.filename}")
        print("="*70)
        
        try:
            self.file_handle = open(self.filename, "w")
            header = "\t".join(self.save_columns) + "\n"
            self.file_handle.write(header)
            self.file_handle.flush()
            print(f"Data file '{self.filename}' created successfully.")
            print(f"Saving columns: {', '.join(self.save_columns)}")
        except Exception as e:
            print(f"Error opening file: {e}")
            raise
        
        if self.enable_plot:
            try:
                self.plotter = RealtimePlotter(MeasurementMode.SATURATION)
                print("Real-time plotter initialized successfully.")
            except Exception as e:
                print(f"Warning: Could not initialize plotter: {e}")
                self.plotter = None
        
        self.drain_smu.set_voltage(params.vsd_sat)
        self.shared_state.current_vsd = params.vsd_sat
        print(f"Drain voltage set to {params.vsd_sat} V")
        
        t_start = time.time()
        
        # 【修正】測定スレッド起動
        self.measuring = True
        self.measurement_thread = threading.Thread(
            target=self._measurement_loop,
            args=(t_start, params.data_interval),
            daemon=True,
            name="MeasurementThread"
        )
        self.measurement_thread.start()
        print("[MeasurementThread] Started - continuous acquisition at regular intervals")
        
        # 【修正】ライタースレッド起動
        self.writing = True
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="WriterThread"
        )
        self.writer_thread.start()
        print("[WriterThread] Started")
        
        # メインスレッドは電圧制御とVref到達判定のみを担当
        try:
            for cycle in range(params.cycle_num):
                self.shared_state.update_cycle(cycle + 1)
                print(f"\n{'='*70}")
                print(f"Cycle {cycle + 1}/{params.cycle_num}")
                print(f"{'='*70}")
                
                # ステップ方向の決定
                if params.vref_sat_max_change > params.vref_sat_initial:
                    vg_step = params.vg_sat_step
                else:
                    vg_step = -params.vg_sat_step
                
                # --- Forward sweep: Vref_initial → Vref_max_change ---
                print(f"\n--- Forward sweep (Vref: {params.vref_sat_initial:.6f} → {params.vref_sat_max_change:.6f} V) ---")
                vg_current = params.vg_sat_initial
                step = 0
                
                while True:
                    step += 1
                    self.gate_smu.set_voltage(vg_current)
                    self.shared_state.update_voltage(vg_current)
                    
                    print(f"\n  Forward Step {step}: Vg = {vg_current:.6f} V")
                    
                    reached, vref_avg, samples = self._check_vref_reached(
                        params.vref_sat_max_change,
                        params.sat_wait_time,
                        params.data_interval
                    )
                    
                    if reached:
                        print(f"  ✓ Vref reached target {params.vref_sat_max_change:.6f} V")
                        break
                    else:
                        print(f"  → Vref not yet at target, stepping voltage...")
                        vg_current += vg_step
                
                # --- Backward sweep: Vref_max_change → Vref_initial ---
                print(f"\n--- Backward sweep (Vref: {params.vref_sat_max_change:.6f} → {params.vref_sat_initial:.6f} V) ---")
                vg_step = -vg_step
                step = 0
                
                while True:
                    step += 1
                    vg_current += vg_step
                    self.gate_smu.set_voltage(vg_current)
                    self.shared_state.update_voltage(vg_current)
                    
                    print(f"\n  Backward Step {step}: Vg = {vg_current:.6f} V")
                    
                    reached, vref_avg, samples = self._check_vref_reached(
                        params.vref_sat_initial,
                        params.sat_wait_time,
                        params.data_interval
                    )
                    
                    if reached:
                        print(f"  ✓ Vref returned to initial {params.vref_sat_initial:.6f} V")
                        break
                    else:
                        print(f"  → Vref not yet at initial, stepping voltage...")
                
                print(f"\n✓ Cycle {cycle + 1} completed")
        
        finally:
            print("\n[MainThread] Stopping measurement thread...")
            self.measuring = False
            if self.measurement_thread is not None:
                self.measurement_thread.join(timeout=3.0)
            print("[MeasurementThread] Stopped")
            
            print("[MainThread] Waiting for queue to empty...")
            try:
                self.data_queue.join()
            except:
                pass
            
            print("[MainThread] Stopping writer thread...")
            self.writing = False
            if self.writer_thread is not None:
                self.writer_thread.join(timeout=3.0)
            print("[WriterThread] Stopped")
            
            if self.file_handle is not None:
                try:
                    self.file_handle.close()
                    print(f"\nData file '{self.filename}' closed.")
                except Exception as e:
                    print(f"Error closing file: {e}")
        
        print("\n" + "="*70)
        print("SATURATION mode measurement completed.")
        print(f"Total data points: {len(self.measurement_data)}")
        print(f"Data saved to: {self.filename}")
        print("="*70)
    
    def save_data(self, filename: str = None):
        if filename is None:
            filename = self.filename
        
        if not self.measurement_data:
            print("No data to save.")
            return
        
        print(f"\nData was saved in real-time to: {filename}")
        print(f"Total data points saved: {len(self.measurement_data)}")
    
    def cleanup(self):
        self.measuring = False
        self.writing = False
        if self.plotter is not None:
            try:
                self.plotter.close()
            except:
                pass
        if self.file_handle is not None:
            try:
                self.file_handle.close()
            except:
                pass

# ============================================
# 使用例:接続確認付きSaturationモード測定
# ============================================
if __name__ == "__main__":
    rm = pyvisa.ResourceManager()
    
    dmm = DMM6500(rm=rm, 
                  ip="192.168.1.103", 
                  v_range=1.0, 
                  nplc=10, 
                  autozero="ON", 
                  timeout=5000, 
                  retries=3
                  )
    
    drain_smu = Keithley2450Drain(rm=rm, 
                                  ip="192.168.1.102", 
                                  source_voltage=0.01, 
                                  compliance_current=0.1,
                                   meas_range=None, 
                                   nplc=1.0, 
                                   autozero="ON", 
                                   timeout=5000, 
                                   retries=3)
    
    gate_smu = Keithley2450Gate(rm=rm, 
                                ip="192.168.1.101", 
                                compliance_current=0.001, 
                                meas_range=None,
                                 nplc=1.0, 
                                 autozero="ON", 
                                 timeout=5000, 
                                 retries=3)
    
    sat_params = SaturationParams(
        vsd_sat=-0.01,#ソースドレイン電圧[V]
        cycle_num=1,#サイクル数
        vg_sat_initial=0,#ゲート初期電圧[V]
        vref_sat_initial=-0.04, #Vref初期値[V]
        vref_sat_max_change=-0.8, #Vref最大変化値[V]
        vg_sat_step=0.05, #ゲート電圧ステップ[V]
        sat_wait_time=30, #Vref到達待機時間[s]
        data_interval=1.0 #データ取得間隔[s]
    )
    
    controller = None
    try:
        print("\n" + "="*70)
        print("CONNECTING TO INSTRUMENTS")
        print("="*70 + "\n")
        
        dmm.connect()
        dmm.configure()
        
        drain_smu.connect()
        drain_smu.configure()
        
        gate_smu.connect()
        gate_smu.configure()
        
        print("\n" + "="*70)
        print("ALL INSTRUMENTS READY")
        print("="*70 + "\n")
        
        controller = MeasurementController(dmm, drain_smu, gate_smu, enable_plot=True)
        controller.measure_saturation(sat_params)
        
        controller.save_data("measurement_saturation.txt")
        
        input("\nPress Enter to close the graphs and exit...")
        
    except Exception as e:
        print(f"\n!!! ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n" + "="*70)
        print("CLEANUP")
        print("="*70)
        if controller:
            controller.cleanup()
        dmm.close()
        drain_smu.close()
        gate_smu.close()
        print("All connections closed.")
