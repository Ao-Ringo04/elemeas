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

# ============================================
# 測定モード定義
# ============================================
class MeasurementMode(Enum):
    """測定モード"""
    LINEAR = "linear"
    OUTPUT = "output"
    SATURATION = "saturation"

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

#@dataclass
#class LinearParams:
#    """Linearモードのパラメータ（例）"""
#    vds_start: float
#    vds_stop: float
#    vgs_list: List[float]
#    points: int
#    interval: float

#@dataclass
#class OutputParams:
#    """Outputモードのパラメータ（例）"""
#    vgs_start: float
#    vgs_stop: float
#    vds_fixed: float
#    points: int
#    interval: float

# ============================================
# 抽象基底クラス（接続リトライ機能付き）
# ============================================
class InstrumentBase(ABC):
    """測定機器の基底クラス（接続リトライ機能付き）"""
    
    def __init__(self, rm: pyvisa.ResourceManager, ip: str, timeout: int = 5000, retries: int = 3):
        self.rm = rm
        self.ip = ip
        self.timeout = timeout
        self.retries = retries
        self.inst = None
        self.instrument_name = self.__class__.__name__
    
    def connect(self):
        """機器に接続（リトライ機能付き）"""
        resource = f"TCPIP0::{self.ip}::inst0::INSTR"
        
        for attempt in range(self.retries):
            try:
                # 既存の接続があれば強制的にクローズ
                if self.inst:
                    try:
                        self.inst.close()
                    except Exception:
                        pass
                    self.inst = None
                
                # ResourceManagerの古い接続をクリア
                try:
                    # 既存のリソースを検索してクローズ
                    resources = self.rm.list_resources()
                    if resource in resources:
                        try:
                            temp_inst = self.rm.open_resource(resource)
                            temp_inst.close()
                        except Exception:
                            pass
                except Exception:
                    pass
                
                # 新規接続
                print(f"[{self.instrument_name}] Connecting to {self.ip} (Attempt {attempt + 1}/{self.retries})...")
                self.inst = self.rm.open_resource(resource)
                self.inst.timeout = self.timeout
                self.inst.read_termination = "\n"
                self.inst.write_termination = "\n"
                
                # 短い待機時間を追加
                time.sleep(0.5)
                
                # 接続確認（IDクエリ）
                self.inst.write("*CLS")  # まずクリア
                time.sleep(0.2)
                idn = self.inst.query("*IDN?")
                print(f"[{self.instrument_name}] Connected successfully: {idn.strip()}")
                return True
                
            except Exception as e:
                print(f"[{self.instrument_name}] Connection attempt {attempt + 1} failed: {e}")
                
                # インスタンスをクリーンアップ
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
    
    def verify_connection(self) -> bool:
        """接続状態を確認"""
        if self.inst is None:
            return False
        try:
            self.inst.query("*IDN?")
            return True
        except Exception:
            return False
    
    @abstractmethod
    def configure(self):
        """機器の設定（サブクラスで実装）"""
        #機器へ送信するコマンドを実装
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
    
    def __init__(
        self,
        rm: pyvisa.ResourceManager,
        ip: str,
        v_range: float = 1.0,
        nplc: float = 10.0,
        autozero: str = "ON",
        timeout: int = 5000,
        retries: int = 3
    ):
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
        
        # DCV測定を選択
        self.inst.write("SENS:FUNC \"VOLT:DC\"")
        
        # レンジとNPLC
        self.inst.write(f"SENS:VOLT:RANG {self.v_range}")
        self.inst.write(f"SENS:VOLT:NPLC {self.nplc}")
        
        # オートゼロ
        self.inst.write("SENS:VOLT:AZER ON" if self.autozero.upper()=="ON" else "SENS:VOLT:AZER OFF")
        
        # 入力インピーダンス
        self.inst.write(f"SENS:VOLT:INP 10E6")  # 10MΩ
        
        # 平均化オフ
        self.inst.write("SENS:VOLT:AVER:STAT OFF")
        
        # 設定確認
        test_reading = self.read_single()
        print(f"[{self.instrument_name}] Configuration complete. Test reading: {test_reading:.6f} V")
    
    def read_single(self) -> float:
        """単一電圧測定"""
        if self.inst is None:
            raise RuntimeError("Not connected.")
        result = self.inst.query("READ?")
        return float(result.strip())

# ============================================
# Keithley2450 ドレイン用クラス
# ============================================
class Keithley2450Drain(InstrumentBase):
    """Keithley 2450 SourceMeter (Drain用)"""
    
    def __init__(
        self,
        rm: pyvisa.ResourceManager,
        ip: str,
        source_voltage: float,
        compliance_current: float = 0.001,
        meas_range: float = None,
        nplc: float = 10.0,
        autozero: str = "ON",
        timeout: int = 5000,
        retries: int = 3
    ):
        super().__init__(rm, ip, timeout, retries)
        self.source_voltage = source_voltage
        self.compliance_current = compliance_current
        self.meas_range = meas_range
        self.nplc = nplc
        self.autozero = autozero
    
    def configure(self):
        """電圧ソース・電流測定の設定"""
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
            self.inst.write("SENS:CURR:RANG:AUTO ON") #設定しなかった場合オートレンジ
        self.inst.write(f"SENS:CURR:NPLC {self.nplc}")
        self.inst.write(f"SYST:AZER {self.autozero}")
        
        self.inst.write("OUTP ON")
        
        # 設定確認
        test_current = self.read_current()
        print(f"[{self.instrument_name}] Configuration complete. Output ON. Test reading: {test_current:.6E} A")
    
    def set_voltage(self, voltage: float):
        """ソース電圧を設定"""
        if self.inst is None:
            raise RuntimeError("Not connected.")
        self.inst.write(f"SOUR:VOLT: {voltage}")
        self.source_voltage = voltage
    
    def read_current(self) -> float:
        """電流測定"""
        if self.inst is None:
            raise RuntimeError("Not connected.")
        result = self.inst.query("READ?")
        return float(result.strip().split(",")[0])
    
    def read_source_voltage(self) -> float:
        """ソース電圧測定"""
        if self.inst is None:
            raise RuntimeError("Not connected.")
        voltage = self.inst.query("SOUR:VOLT:LEV?")  # ソースとして設定した電圧を取得するコマンド
        return float(result.strip().split(",")[0])
    
    def close(self):
        """接続を閉じる"""
        if self.inst is not None:
            try:
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
    
    def __init__(
        self,
        rm: pyvisa.ResourceManager,
        ip: str,
        compliance_current: float,
        meas_range: float = None,
        nplc: float = 1.0,
        autozero: str = "ON",
        timeout: int = 5000,
        retries: int = 3
    ):
        super().__init__(rm, ip, timeout, retries)
        self.compliance_current = compliance_current
        self.meas_range = meas_range
        self.nplc = nplc
        self.autozero = autozero
        self.current_voltage = 0.0
    
    def configure(self):
        """電圧ソース・電流測定の設定"""
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
        
        # 設定確認
        test_current = self.read_current()
        print(f"[{self.instrument_name}] Configuration complete. Output ON. Test reading: {test_current:.6E} A")
    
    def set_voltage(self, voltage: float):
        """ゲート電圧を設定"""
        if self.inst is None:
            raise RuntimeError("Not connected.")
        self.inst.write(f"SOUR:VOLT:LEV {voltage}")
        self.current_voltage = voltage
    
    def read_current(self) -> float:
        """電流測定"""
        if self.inst is None:
            raise RuntimeError("Not connected.")
        result = self.inst.query("READ?")
        return float(result.strip().split(",")[0])

    def read_source_voltage(self) -> float:
        """ソース電圧測定"""
        if self.inst is None:
            raise RuntimeError("Not connected.")
        voltage = self.inst.query("SOUR:VOLT:LEV?")  # ソースとして設定した電圧を取得するコマンド
        return float(result.strip().split(",")[0])
    
    def close(self):
        """接続を閉じる"""
        if self.inst is not None:
            try:
                print(f"[{self.instrument_name}] Turning output OFF...")
                self.inst.write("OUTP OFF")
            except Exception as e:
                print(f"[{self.instrument_name}] Error turning output OFF: {e}")
        super().close()


# ============================================
# 測定制御クラス（3スレッド構成）
# ============================================

class SharedState:
    """スレッド間で共有する状態"""
    def __init__(self):
        self.current_vg = 0.0
        self.current_vsd = 0.0
        self.current_cycle = 1
        self.lock = threading.Lock()
    
    def update_voltage(self, vg: float):
        with self.lock:
            self.current_vg = vg
    
    def update_cycle(self, cycle: int):
        with self.lock:
            self.current_cycle = cycle
    
    def get_state(self):
        with self.lock:
            return self.current_vg, self.current_vsd, self.current_cycle


class MeasurementController:
    """測定制御クラス（3スレッド構成）"""
    
    def __init__(
        self,
        dmm,  # DMM6500
        drain_smu,  # Keithley2450Drain
        gate_smu,  # Keithley2450Gate
        filename: str = "measurement_saturation.txt",
        save_columns: List[str] = ['Time', "GateI", "GateV", "DrainI", "Vref"]
    ):
        self.dmm = dmm
        self.drain_smu = drain_smu
        self.gate_smu = gate_smu
        self.plotter = None  # Optional[RealtimePlotter]
        self.measurement_data = []
        
        # スレッド管理
        self.measurement_thread = None
        self.writer_thread = None
        self.measuring = False
        self.writing = False
        
        # データキュー
        self.data_queue = Queue()
        
        # 共有状態
        self.shared_state = SharedState()
        
        # ファイル設定
        self.filename = filename
        self.file_handle = None
        
        # カラム設定
        self.available_columns = ['Time', "GateI", "GateV", "DrainI", "DrainV", "Vref"]
        if save_columns is None:
            self.save_columns = self.available_columns.copy()
        else:
            invalid_cols = [col for col in save_columns if col not in self.available_columns]
            if invalid_cols:
                raise ValueError(f"Invalid column names: {invalid_cols}. Available: {self.available_columns}")
            self.save_columns = save_columns
    
    def _get_column_value(self, data_point: Dict, col_name: str):
        """カラム名からデータ値を取得"""
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
        """選択されたカラムのみファイルに書き込み"""
        if self.file_handle is None:
            return
        
        values = []
        for col in self.save_columns:
            val = self._get_column_value(data_point, col)
            if col == 'Cycle':
                values.append(f"{val}")
            else:
                values.append(f"{val:.7E}")
        
        self.file_handle.write("\t".join(values) + "\n")
        self.file_handle.flush()
    
    def _measurement_loop(self, t_start: float, data_interval: float):
        """測定スレッド: 一定間隔でデータを取得してQueueに追加"""
        next_measure_time = time.time()
        
        while self.measuring:
            current_time = time.time()
            
            if current_time >= next_measure_time:
                t_measure = current_time - t_start
                
                try:
                    # 全データ測定（既存のクラスメソッドを使用）
                    vref = self.dmm.read_single()
                    isd = self.drain_smu.read_current()
                    ig = self.gate_smu.read_current()
                    
                    # 現在の状態を取得
                    vg, vsd, cycle = self.shared_state.get_state()
                    
                    # データポイント作成
                    data_point = {
                        'time': t_measure,
                        'cycle': cycle,
                        'vref': vref,
                        'vg': vg,
                        'isd': isd,
                        'ig': ig,
                        'vsd': vsd
                    }
                    
                    # Queueに追加
                    self.data_queue.put(data_point)
                    
                    # コンソール出力
                    print(f"  [t={t_measure:.1f}s] Vref={vref*1000:.3f}mV, Isd={isd:.6E}A, Vg={vg:.4f}V")
                    
                except Exception as e:
                    print(f"Measurement error: {e}")
                
                # 次の測定時刻を設定
                next_measure_time += data_interval
            
            # 短時間スリープ
            time.sleep(0.01)
    
    def _writer_loop(self):
        """書き込みスレッド: Queueからデータを取得してファイル書き込み＆グラフ更新"""
        while self.writing:
            try:
                # Queueからデータ取得（タイムアウト付き）
                data_point = self.data_queue.get(timeout=0.5)
                
                # メモリに保存
                self.measurement_data.append(data_point)
                
                # ファイルに書き込み
                self._write_data_line(data_point)
                
                # グラフ更新
                if self.plotter is not None:
                    self.plotter.update(
                        data_point['time'],
                        data_point['vref'],
                        data_point['isd'],
                        data_point['vg']
                    )
                
                # Queueのタスク完了を通知
                self.data_queue.task_done()
                
            except Empty:
                # タイムアウト時は継続
                continue
            except Exception as e:
                print(f"Writer error: {e}")
    
    def measure_saturation(self, params):  # params: SaturationParams
        """Saturationモード測定（3スレッド構成）"""
        print("="*60)
        print("Starting SATURATION mode measurement (3-thread)")
        print(f"Data acquisition interval: {params.data_interval} s")
        print(f"Output file: {self.filename}")
        print("="*60)
        
        # ファイルを開く
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
        
        # リアルタイムプロット初期化（必要に応じて）
        # from your_plotting_module import RealtimePlotter, MeasurementMode
        # self.plotter = RealtimePlotter(MeasurementMode.SATURATION)
        
        # ドレイン電圧設定（既存のメソッドを使用）
        self.drain_smu.set_voltage(params.vsd_sat)
        self.shared_state.current_vsd = params.vsd_sat
        print(f"Drain voltage set to {params.vsd_sat} V")
        
        # 測定開始時刻
        t_start = time.time()
        
        # 測定スレッド開始
        self.measuring = True
        self.measurement_thread = threading.Thread(
            target=self._measurement_loop,
            args=(t_start, params.data_interval),
            daemon=True
        )
        self.measurement_thread.start()
        print("[MeasurementThread] Started")
        
        # 書き込みスレッド開始
        self.writing = True
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True
        )
        self.writer_thread.start()
        print("[WriterThread] Started")
        
        try:
            # メインスレッド: 電圧制御のみ
            for cycle in range(params.cycle_num):
                self.shared_state.update_cycle(cycle + 1)
                print(f"\n{'='*60}")
                print(f"Cycle {cycle + 1}/{params.cycle_num}")
                print(f"{'='*60}")
                
                # Vrefステップ計算
                if params.vref_sat_max_change > params.vref_sat_initial:
                    vref_step = params.vg_sat_step
                    num_steps = int((params.vref_sat_max_change - params.vref_sat_initial) / vref_step) + 1
                else:
                    vref_step = -params.vg_sat_step
                    num_steps = int((params.vref_sat_initial - params.vref_sat_max_change) / params.vg_sat_step) + 1
                
                # Forward sweep
                print(f"\n--- Forward sweep (Initial → Max) ---")
                for step in range(num_steps):
                    vg_current = params.vg_sat_initial + step * vref_step
                    self.gate_smu.set_voltage(vg_current)
                    self.shared_state.update_voltage(vg_current)
                    print(f"\nStep {step+1}/{num_steps}: Vg = {vg_current:.4f} V")
                    print(f"  Waiting {params.sat_wait_time} s for equilibration...")
                    time.sleep(params.sat_wait_time)
                
                # Backward sweep
                print(f"\n--- Backward sweep (Max → Initial) ---")
                for step in range(num_steps - 2, -1, -1):
                    vg_current = params.vg_sat_initial + step * vref_step
                    self.gate_smu.set_voltage(vg_current)
                    self.shared_state.update_voltage(vg_current)
                    print(f"\nStep {num_steps-step}/{num_steps}: Vg = {vg_current:.4f} V")
                    print(f"  Waiting {params.sat_wait_time} s for equilibration...")
                    time.sleep(params.sat_wait_time)
        
        finally:
            # スレッド停止
            print("\n[MainThread] Stopping measurement thread...")
            self.measuring = False
            if self.measurement_thread is not None:
                self.measurement_thread.join(timeout=2.0)
            print("[MeasurementThread] Stopped")
            
            print("[MainThread] Waiting for queue to empty...")
            # Queueの残りデータを処理
            self.data_queue.join()
            
            print("[MainThread] Stopping writer thread...")
            self.writing = False
            if self.writer_thread is not None:
                self.writer_thread.join(timeout=2.0)
            print("[WriterThread] Stopped")
            
            # ファイルを閉じる
            if self.file_handle is not None:
                try:
                    self.file_handle.close()
                    print(f"\nData file '{self.filename}' closed.")
                except Exception as e:
                    print(f"Error closing file: {e}")
        
        print("\n" + "="*60)
        print("SATURATION mode measurement completed.")
        print(f"Total data points: {len(self.measurement_data)}")
        print(f"Data saved to: {self.filename}")
        print("="*60)
    
    def save_data(self, filename: str = None):
        """データ保存（既にリアルタイムで保存されているので確認用）"""
        if filename is None:
            filename = self.filename
        
        if not self.measurement_data:
            print("No data to save.")
            return
        
        print(f"\nData was saved in real-time to: {filename}")
        print(f"Total data points saved: {len(self.measurement_data)}")
        
        # バックアップファイルを作成
        backup_filename = filename.replace(".txt", "_backup.txt")
        try:
            with open(backup_filename, "w") as f:
                header = "\t".join(self.save_columns) + "\n"
                f.write(header)
                for data in self.measurement_data:
                    values = []
                    for col in self.save_columns:
                        val = self._get_column_value(data, col)
                        if col == 'Cycle':
                            values.append(f"{val}")
                        else:
                            values.append(f"{val:.7E}")
                    f.write("\t".join(values) + "\n")
            print(f"Backup file '{backup_filename}' created successfully.")
        except Exception as e:
            print(f"Error creating backup file: {e}")
    
    def cleanup(self):
        """クリーンアップ（互換性のため）"""
        self.measuring = False
        self.writing = False
        if self.file_handle is not None:
            try:
                self.file_handle.close()
            except:
                pass


# ============================================
# 使用例：接続確認付きSaturationモード測定
# ============================================
if __name__ == "__main__":
    rm = pyvisa.ResourceManager()
    
    # 機器インスタンス作成（リトライ回数3回、タイムアウト5秒）
    dmm = DMM6500(
        rm=rm,
        ip="192.168.1.103",
        v_range=1.0,
        nplc=10,
        autozero="ON",
        timeout=5000,
        retries=3
    )
    
    drain_smu = Keithley2450Drain(
        rm=rm,
        ip="192.168.1.102",
        source_voltage=0.01,
        compliance_current=0.1,
        meas_range= None,
        nplc=1.0,
        autozero="ON",
        timeout=5000,
        retries=3
    )
    
    gate_smu = Keithley2450Gate(
        rm=rm,
        ip="192.168.1.101",
        compliance_current=0.001,
        meas_range= None,
        nplc=1.0,
        autozero="ON",
        timeout=5000,
        retries=3
    )
    
    sat_params = SaturationParams(
        vsd_sat=0.01,
        cycle_num=1,
        vg_sat_initial=0.08,
        vref_sat_initial=0.08,
        vref_sat_max_change=-0.8,
        vg_sat_step=0.05,
        sat_wait_time=30,
        data_interval=1.0
    )
    
    try:
        # 接続と設定（自動リトライ）
        print("\n" + "="*60)
        print("CONNECTING TO INSTRUMENTS")
        print("="*60 + "\n")
        
        dmm.connect()
        dmm.configure()
        
        drain_smu.connect()
        drain_smu.configure()
        
        gate_smu.connect()
        gate_smu.configure()
        
        print("\n" + "="*60)
        print("ALL INSTRUMENTS READY")
        print("="*60 + "\n")
        
        # 測定実行
        controller = MeasurementController(dmm, drain_smu, gate_smu)
        controller.measure_saturation(sat_params)
        
        # データ保存
        controller.save_data("measurement_saturation.txt")
        
        # グラフ表示継続
        input("\nPress Enter to close the graphs and exit...")
        
    except Exception as e:
        print(f"\n!!! ERROR: {e}")
        
    finally:
        # クリーンアップ
        print("\n" + "="*60)
        print("CLEANUP")
        print("="*60)
        controller.cleanup()
        dmm.close()
        drain_smu.close()
        gate_smu.close()
        print("All connections closed.")