"""Time processors
"""
import typing

import numpy as np

import lps_utils.quantities as lps_qty
#import lps_sp.signal as lps_signal
import lps_ml.core as ml_core
import scipy.signal as sp_signal
import logging

logger = logging.getLogger(__name__)


class Resampler(ml_core.AudioPipeline):
    """ AudioPipeline to change sample frequency. """

    def time_to_samples(t, fs_hz):
        return int(t.value * fs_hz)


    def __init__(self,
                 fs_out: lps_qty.Frequency):
        super().__init__()
        self.fs_out = fs_out

    def process(self, fs: lps_qty.Frequency, data: np.array) \
            -> typing.Tuple[lps_qty.Frequency, np.array]:
        
        # DEBUG
        #print(f"DEBUG Resampler: entrada - fs={fs}, data shape={data.shape}")
        logger.debug("Resampler: entrada - fs=%s, data shape=%s", fs, data.shape)

        # Extraia valores numéricos
        try:
            fs_hz = fs.value
            fs_out_hz = self.fs_out.value
        except AttributeError:
            try:
                fs_hz = fs.get_hz()
                fs_out_hz = self.fs_out.get_hz()
            except AttributeError:
                fs_hz = float(fs)
                fs_out_hz = float(self.fs_out)
        
        #print(f"DEBUG Resampler: fs_hz={fs_hz}, fs_out_hz={fs_out_hz}")
        logger.debug(
            "Resampler: fs_hz=%s, fs_out_hz=%s",
            fs_hz,
            fs_out_hz
        )
    


        # Calcule fator de decimação
        decimation_factor = int(fs_hz / fs_out_hz)
        
        #print(f"DEBUG Resampler: decimation_factor={decimation_factor}")
        logger.debug(
            "Resampler: decimation_factor=%d",
            decimation_factor
        )

        if decimation_factor > 1:
            decimated_signal = sp_signal.decimate(data, decimation_factor)
        else:
            decimated_signal = data
        
        #print(f"DEBUG Resampler: saída - data shape={decimated_signal.shape}")
        logger.debug(
            "Resampler: saída - data shape=%s",
            decimated_signal.shape
        )

        return self.fs_out, decimated_signal


class CPADetector(ml_core.AudioPipeline):
    """ AudioPipeline that detects the highest energy point (CPA) and cuts a centered window."""

    def __init__(self,
                 analysis_window: lps_qty.Time,
                 crop_window: lps_qty.Time):
        super().__init__()
        self.analysis_window = analysis_window
        self.crop_window = crop_window

    def process(self,
                fs: lps_qty.Frequency,
                data: np.ndarray) -> typing.Tuple[lps_qty.Frequency, np.ndarray]:

        if data.ndim > 2:
            raise ValueError(f"Input signal must have at 1 dimension, received: {data.ndim}D.")

        if data.ndim == 2:
            if 1 in data.shape:
                data = data.squeeze()
            else:
                raise ValueError(f"Input signal must have at 1 dimension, received: {data.ndim}D.")

        elif data.ndim < 1:
            raise ValueError(f"Input signal must have at 1 dimension, received: {data.ndim}D.")
        
        # Extraia valor numérico de fs
        try:
            fs_hz = fs.value
        except AttributeError:
            try:
                fs_hz = fs.get_hz()
            except AttributeError:
                fs_hz = float(fs)

        n_samples = len(data)
        #n_analysis = int(self.analysis_window * fs_hz)
        #n_analysis = int(self.analysis_window.to("s").value * fs_hz)
        #print(type(self.analysis_window), self.analysis_window)
        #print(type(fs_hz), fs_hz)

        logger.debug(
            "CPADetector: analysis_window type=%s, value=%s",
            type(self.analysis_window),
            self.analysis_window
        )

        logger.debug(
            "CPADetector: fs_hz type=%s, value=%s",
            type(fs_hz),
            fs_hz
        )



        #n_analysis = int(self.analysis_window.value * fs_hz)
        #n_crop = int(self.crop_window * fs_hz)

        #analysis_window E crop_window sao Time
        #analysis_s = self.analysis_window.value
        analysis_s = self.analysis_window.get_s()

        crop_s = self.crop_window.get_s()


        n_analysis = int(analysis_s * fs_hz)
        n_crop = int(crop_s * fs_hz)



        if n_crop > n_samples:
            raise ValueError("Crop window is larger than the input signal.")

        if n_analysis > n_samples:
            raise ValueError("Analysis window is larger than the input signal.")

        step = n_analysis // 4
        energies = []
        starts = []

        for start in range(0, n_samples - n_analysis + 1, step):
            window = data[start:start + n_analysis]
            energy = np.sum(window ** 2)
            energies.append(energy)
            starts.append(start)

        max_idx = int(np.argmax(energies))
        cpa_start = starts[max_idx]
        cpa_center = cpa_start + n_analysis // 2

        half_crop = n_crop // 2
        crop_start = max(0, cpa_center - half_crop)
        crop_end = min(n_samples, crop_start + n_crop)

        if crop_end - crop_start < n_crop:
            crop_start = max(0, crop_end - n_crop)

        cropped_signal = data[crop_start:crop_end]

        return fs, cropped_signal

class TimeProcessor(ml_core.AudioProcessor):
    """ Simple time processor for resampling, pipelined, and sliding windowing. """

    def __init__(self,
                 duration: lps_qty.Time,
                 overlap: lps_qty.Time,
                 fs_out: lps_qty.Frequency,
                 pipelines: typing.Union[ml_core.AudioPipeline,
                                         typing.List[ml_core.AudioPipeline]] = None):
        super().__init__()
        self.duration = duration
        self.overlap = overlap

        if pipelines is None:
            self.pipelines = []
        elif isinstance(pipelines, ml_core.AudioPipeline):
            self.pipelines = [pipelines]
        else:
            self.pipelines = pipelines

        if fs_out is not None:
            self.pipelines.insert(0, Resampler(fs_out=fs_out))

    def process(self, fs: lps_qty.Frequency, data: np.array) -> typing.List[np.array]:
        """Process audio data through pipelines and create windows."""
        
        # Processar todos os pipelines primeiro
        for pipeline in self.pipelines:
            fs, data = pipeline.process(fs=fs, data=data)
        
        # DEBUG
        #print(f"DEBUG TimeProcessor: fs após pipelines = {fs}, type = {type(fs)}")
        logger.debug(
            "TimeProcessor: fs após pipelines = %s (type=%s)",
            fs,
            type(fs)
        )

        
        print(f"DEBUG TimeProcessor: data shape = {data.shape if hasattr(data, 'shape') else 'no shape'}")
        
        # Converter fs para valor numérico
        try:
            fs_hz = fs.value
        except AttributeError:
            try:
                fs_hz = fs.get_hz()
            except AttributeError:
                fs_hz = float(fs)
        
        # Calcular tamanhos das janelas
        #window_size = int(self.duration * fs_hz)
        window_size = int(self.duration.get_s() * fs_hz)

        #overlap_size = int(self.overlap * fs_hz)
        overlap_size = int(self.overlap.get_s() * fs_hz)

        step = int(window_size - overlap_size)
        
        #print(f"DEBUG TimeProcessor: window_size = {window_size}, overlap_size = {overlap_size}, step = {step}")
        logger.debug(
            "TimeProcessor: window_size=%d, overlap_size=%d, step=%d",
            window_size,
            overlap_size,
            step
        )

        
        print(f"DEBUG TimeProcessor: len(data) = {len(data)}")
        
        # Criar janelas
        windows = []
        for start in range(0, len(data) - window_size + 1, step):
            windows.append(data[start:start + window_size])
        
        print(f"DEBUG TimeProcessor: criadas {len(windows)} janelas")
        
        if len(windows) == 0:
            print("WARNING: Nenhuma janela criada!")
            # Retornar pelo menos uma janela com zeros se necessário
            windows = [np.zeros(window_size)]
        
        return windows
    

    # obs: foram corrigidos para .value analysis_window, crop_window, duration, overlap