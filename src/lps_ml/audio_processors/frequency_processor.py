
import typing
import numpy as np

import lps_utils.quantities as lps_qty
import lps_ml.core as ml_core

from lps_ml.audio_processors.time_processors import Resampler
from lps_sp.acoustical.analysis import SpectralAnalysis, Parameters




class FrequencyProcessor(ml_core.AudioProcessor):
    """Processor that converts time windows into spectral representations."""

    def __init__(self,
                 duration: lps_qty.Time,
                 overlap: lps_qty.Time,
                 fs_out: lps_qty.Frequency,
                 spectral: SpectralAnalysis = SpectralAnalysis.LOFAR,
                 params: Parameters = Parameters(),
                 pipelines: typing.Union[
                     ml_core.AudioPipeline,
                     typing.List[ml_core.AudioPipeline]] = None):

        super().__init__()

        self.duration = duration
        self.overlap = overlap
        self.spectral = spectral
        self.params = params

        if pipelines is None:
            self.pipelines = []
        elif isinstance(pipelines, ml_core.AudioPipeline):
            self.pipelines = [pipelines]
        else:
            self.pipelines = pipelines

        if fs_out is not None:
            self.pipelines.insert(0, Resampler(fs_out=fs_out))

# EVAL
# Ha dois codigos abaixo que fazem praticamente a mesma coisa
# Duas funcoes para processamento
# Tentativa para reduzir tempo de processamento

    # def process(self,
    #             fs: lps_qty.Frequency,
    #             data: np.array) -> np.array:

    #     # aplica pipelines
    #     #for pipeline in self.pipelines:
    #      #   fs, data = pipeline.process(fs=fs, data=data)

    #     #fs_value = fs.get()

    #     #window_size = int(self.duration.get() * fs_value)
    #     #overlap_size = int(self.overlap.get() * fs_value)

    #     #step = window_size - overlap_size

    #     for pipeline in self.pipelines:
    #         fs, data = pipeline.process(fs=fs, data=data)

    #     fs_value = fs.get(lps_qty.lps_unity.Frequency.HZ) #Saber se isso e mesmo necessario (uso esta correto)

    #     window_size = int(self.duration.get_s() * fs_value)
    #     overlap_size = int(self.overlap.get_s() * fs_value)

    #     step = window_size - overlap_size
        
    #     features = []

    #     for start in range(0, len(data) - window_size + 1, step):

    #         window = data[start:start + window_size]

    #         spectrum, freqs, times = self.spectral.apply(
    #             #data=window, muito tempo de processamento
    #             data=data,
    #             fs=fs_value,
    #             params=self.params
    #         )

    #         features.append(spectrum)

    #     return np.array(features)

# Achei que a funcao abaixo foi um pouco mais rapida, mas nao tenho certeza (pedi para o chat uma funcao
#  menos custosa em tempo e ele me deu essa)
# Ha duas variaveis que nao foram usadas e essa funcao sera mudada ainda com o intuito de reduzir tempo e retirar
#  essas vars nao utilizadas

#TODO

    def process(self, fs, data):

        for pipeline in self.pipelines:
            fs, data = pipeline.process(fs=fs, data=data)

        fs_value = fs.get(lps_qty.lps_unity.Frequency.HZ)

        spectrum, freqs, times = self.spectral.apply(
            data=data,
            fs=fs_value,
            params=self.params
        )
#get_hz
        # time_step = times[1] - times[0]
        if len(times) < 2:
            return []

        time_step = times[1] - times[0]

        window_frames = int(self.duration.get_s() / time_step)
        #step_frames = int((self.duration.get_s() - self.overlap.get_s()) / time_step)
        step_frames = max(1, int((self.duration.get_s() - self.overlap.get_s()) / time_step))
        fragments = []

        #for start in range(0, spectrum.shape[1] - window_frames):
         #   fragment = spectrum[:, start:start + window_frames]
          #  fragments.append(fragment)
        #return fragments
        for start in range(0, spectrum.shape[1] - window_frames + 1, step_frames):
            fragment = spectrum[:, start:start + window_frames]
            fragments.append(fragment)

        return np.array(fragments)