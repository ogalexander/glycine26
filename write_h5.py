from typing import Tuple
import argparse
import os
import sys
from pathlib import Path

# Make the legacy beamtime decoder importable on both the local workstation
# (current repo layout) and the FLASH cluster (2026 beamtime path).
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "analysis" / "scripts"))
sys.path.insert(
    0,
    str(_REPO_ROOT / "11022188" / "processed" / "analysis_tools" / "decoding_script"),
)
sys.path.append('/asap3/flash/gpfs/fl24/2026/data/11022188/processed/analysis_tools/decoding_script')

import re
import glob
import fnmatch

from tqdm import tqdm
import numpy as np
import pandas as pd

from beamtime_scripts_2021.util import write_pos_data_to_file
from beamtime_scripts_2021.MCS6A_decoding import decoding, metadata_decoding
from beamtime_scripts_2021.util import write_tof_data_to_file

import h5py as h5

import config  # path resolution; the only place hardcoded paths live
class DataChunk():
    def __init__(self, chunk_size, train_length, max_ecounts=50, max_icounts=120):
        self.chunk_size = chunk_size
        self.train_length = train_length
        self.max_ecounts = max_ecounts
        self.max_icounts = max_icounts
        self.n_e_overflows = 0
        self.n_i_overflows = 0
        self.reset()
        self.count = 0

    def add_row(self, is_data, tID, gmd, mpe, hor_pos, ver_pos, z , z_std, tofs_e, tofs_i, between_tdc_files):
        self.is_datas[self.idx] = is_data
        self.tIDs[self.idx] = tID
        self.gmds[self.idx] = gmd
        self.mpes[self.idx] = mpe
        self.hor_poss[self.idx] = hor_pos
        self.ver_poss[self.idx] = ver_pos
        self.zs[self.idx] = z
        self.z_stds[self.idx] = z_std
        self.between_tdc_filess[self.idx] = between_tdc_files

        if tofs_e is not None:
            for b_idx, tofs_e_bunch in enumerate(tofs_e):
                n = len(tofs_e_bunch)
                if n > self.max_ecounts:
                    self.n_e_overflows += 1
                    print(
                        f"WARNING: electron TOF count {n} exceeds "
                        f"max_ecounts={self.max_ecounts} (tID={tID}, bunch={b_idx}); "
                        f"clipping. (overflow #{self.n_e_overflows})"
                    )
                    tofs_e_bunch = tofs_e_bunch[:self.max_ecounts]
                    n = self.max_ecounts
                self.tofs_es[self.idx, b_idx, :n] = tofs_e_bunch
        if tofs_i is not None:
            for b_idx, tofs_i_bunch in enumerate(tofs_i):
                n = len(tofs_i_bunch)
                if n > self.max_icounts:
                    self.n_i_overflows += 1
                    print(
                        f"WARNING: ion TOF count {n} exceeds "
                        f"max_icounts={self.max_icounts} (tID={tID}, bunch={b_idx}); "
                        f"clipping. (overflow #{self.n_i_overflows})"
                    )
                    tofs_i_bunch = tofs_i_bunch[:self.max_icounts]
                    n = self.max_icounts
                self.tofs_is[self.idx, b_idx, :n] = tofs_i_bunch

        self.idx += 1

        if self.idx == self.chunk_size:
            print('full!')
            return True
        else:
            return False

    def dump(self, tID_dset, data_flag_dset, z_dset, z_std_dset, gmd_dset, mpe_dset, hor_pos_dset,
        ver_pos_dset, tofs_es_dset, tofs_is_dset, between_tdc_files_dset):
        roi = np.s_[self.count*self.chunk_size:(self.count+1)*self.chunk_size]
        tID_dset[roi]                 = self.tIDs
        data_flag_dset[roi]           = self.is_datas
        z_dset[roi]                   = self.zs
        z_std_dset[roi]               = self.z_stds
        gmd_dset[roi]                 = self.gmds
        mpe_dset[roi]                 = self.mpes
        hor_pos_dset[roi]             = self.hor_poss
        ver_pos_dset[roi]             = self.ver_poss
        # TOF datasets may be None when the run has no TDC .lst files.
        if tofs_es_dset is not None:
            tofs_es_dset[roi]         = self.tofs_es
        if tofs_is_dset is not None:
            tofs_is_dset[roi]         = self.tofs_is
        between_tdc_files_dset[roi]   = self.between_tdc_filess
        self.count += 1

    def reset(self):
        self.is_datas = np.zeros((self.chunk_size, ), dtype='bool')
        self.tIDs = np.zeros((self.chunk_size, ), dtype='double')
        self.gmds = np.full((self.chunk_size, self.train_length), np.nan, dtype=np.float32)
        self.mpes = np.full((self.chunk_size, ), np.nan, dtype=np.float32)
        self.hor_poss = np.full((self.chunk_size, ), np.nan, dtype=np.float32)
        self.ver_poss = np.full((self.chunk_size, ), np.nan, dtype=np.float32)
        self.zs = np.full((self.chunk_size, self.train_length), np.nan, dtype=np.float32)
        self.z_stds = np.full((self.chunk_size, self.train_length), np.nan, dtype=np.float32)
        # Zero-padded uint32 TOF buffers: 0 means "no hit". (Was np.full(.., np.nan, uint32)
        # which raises an invalid-cast warning and stores 0 anyway.)
        self.tofs_es = np.zeros((self.chunk_size, self.train_length, self.max_ecounts), dtype=np.uint32)
        self.tofs_is = np.zeros((self.chunk_size, self.train_length, self.max_icounts), dtype=np.uint32)
        self.between_tdc_filess = np.zeros((self.chunk_size, ), dtype='bool')

        self.idx = 0
        self.is_e_data = False
        self.is_i_data = False
        print('reset.')

    def finish(self, tID_dset, data_flag_dset, z_dset, z_std_dset, gmd_dset, mpe_dset, hor_pos_dset,
        ver_pos_dset, tofs_es_dset, tofs_is_dset, between_tdc_files_dset):
        self.is_datas            = self.is_datas[:self.idx-1]
        self.tIDs                = self.tIDs[:self.idx-1]
        self.gmds                = self.gmds[:self.idx-1]
        self.mpes                = self.mpes[:self.idx-1]
        self.hor_poss            = self.hor_poss[:self.idx-1]
        self.ver_poss            = self.ver_poss[:self.idx-1]
        self.zs                  = self.zs[:self.idx-1]
        self.z_stds              = self.z_stds[:self.idx-1]
        self.tofs_es             = self.tofs_es[:self.idx-1]
        self.tofs_is             = self.tofs_is[:self.idx-1]
        self.between_tdc_filess  = self.between_tdc_filess[:self.idx-1]

        roi = np.s_[self.count*self.chunk_size:(self.count*self.chunk_size) + self.idx-1]

        tID_dset[roi]               = self.tIDs
        z_dset[roi]                 = self.zs
        z_std_dset[roi]             = self.z_stds
        gmd_dset[roi]               = self.gmds
        mpe_dset[roi]               = self.mpes
        hor_pos_dset[roi]           = self.hor_poss
        ver_pos_dset[roi]           = self.ver_poss
        if tofs_es_dset is not None:
            tofs_es_dset[roi]       = self.tofs_es
        if tofs_is_dset is not None:
            tofs_is_dset[roi]       = self.tofs_is
        data_flag_dset[roi]         = self.is_datas
        between_tdc_files_dset[roi] = self.between_tdc_filess

        (dset.resize((self.count*self.chunk_size) + self.idx-1, axis=0) for dset in
            [tID_dset, z_dset, z_std_dset, gmd_dset, mpe_dset, hor_pos_dset, ver_pos_dset, between_tdc_files_dset]) # 4 missing from here


class DataChunkConfig2():
    """
    Chunked accumulator for config 2 combined-H5 writes.

    Mirrors :class:`DataChunk` but swaps the electron/ion TOF buffers for
    a liquid-jet electron TOF buffer (``liq_tofs_e``, TDC channel 3) and a
    Gotthard VLS spectrum buffer (``vls``). ``dump``/``finish`` take a
    dict keyed by dataset name so the larger field set stays readable.
    """

    def __init__(self, chunk_size, train_length, max_ecounts=50, n_vls_pixels=1280):
        self.chunk_size = chunk_size
        self.train_length = train_length
        self.max_ecounts = max_ecounts
        self.n_vls_pixels = n_vls_pixels
        self.n_le_overflows = 0
        self.reset()
        self.count = 0

    def add_row(self, is_data, tID, gmd, mpe, hor_pos, ver_pos, z, z_std,
                liq_tofs_e, vls, between_tdc_files):
        self.is_datas[self.idx]            = is_data
        self.tIDs[self.idx]                = tID
        self.gmds[self.idx]                = gmd
        self.mpes[self.idx]                = mpe
        self.hor_poss[self.idx]            = hor_pos
        self.ver_poss[self.idx]            = ver_pos
        self.zs[self.idx]                  = z
        self.z_stds[self.idx]              = z_std
        self.between_tdc_filess[self.idx]  = between_tdc_files

        if liq_tofs_e is not None:
            for b_idx, bunch in enumerate(liq_tofs_e):
                n = len(bunch)
                if n > self.max_ecounts:
                    self.n_le_overflows += 1
                    print(
                        f"WARNING: liq electron TOF count {n} exceeds "
                        f"max_ecounts={self.max_ecounts} (tID={tID}, bunch={b_idx}); "
                        f"clipping. (overflow #{self.n_le_overflows})"
                    )
                    bunch = bunch[:self.max_ecounts]
                    n = self.max_ecounts
                self.liq_tofs_es[self.idx, b_idx, :n] = bunch
        if vls is not None:
            self.vls[self.idx] = vls

        self.idx += 1
        if self.idx == self.chunk_size:
            print('full!')
            return True
        return False

    def dump(self, dsets):
        roi = np.s_[self.count*self.chunk_size:(self.count+1)*self.chunk_size]
        dsets['tID'][roi]                = self.tIDs
        dsets['local_DAQ_running'][roi]  = self.is_datas
        dsets['z'][roi]                  = self.zs
        dsets['z_std'][roi]              = self.z_stds
        dsets['gmd'][roi]                = self.gmds
        dsets['mpe'][roi]                = self.mpes
        dsets['hor_pos'][roi]            = self.hor_poss
        dsets['ver_pos'][roi]            = self.ver_poss
        # liq_tofs_e is omitted from the schema when the run has no TDC files.
        if dsets.get('liq_tofs_e') is not None:
            dsets['liq_tofs_e'][roi]     = self.liq_tofs_es
        dsets['vls'][roi]                = self.vls
        dsets['between_tdc_files'][roi]  = self.between_tdc_filess
        self.count += 1

    def reset(self):
        self.is_datas            = np.zeros((self.chunk_size,), dtype='bool')
        self.tIDs                = np.zeros((self.chunk_size,), dtype='double')
        self.gmds                = np.full((self.chunk_size, self.train_length), np.nan, dtype=np.float32)
        self.mpes                = np.full((self.chunk_size,), np.nan, dtype=np.float32)
        self.hor_poss            = np.full((self.chunk_size,), np.nan, dtype=np.float32)
        self.ver_poss            = np.full((self.chunk_size,), np.nan, dtype=np.float32)
        self.zs                  = np.full((self.chunk_size, self.train_length), np.nan, dtype=np.float32)
        self.z_stds              = np.full((self.chunk_size, self.train_length), np.nan, dtype=np.float32)
        self.liq_tofs_es         = np.zeros((self.chunk_size, self.train_length, self.max_ecounts), dtype=np.uint32)
        self.vls                 = np.full((self.chunk_size, self.train_length, self.n_vls_pixels), np.nan, dtype=np.float32)
        self.between_tdc_filess  = np.zeros((self.chunk_size,), dtype='bool')
        self.idx = 0
        print('reset.')

    def finish(self, dsets):
        # Mirror DataChunk.finish: trim to idx-1 (drops the last partial row).
        # This off-by-one is replicated for parity with the existing config 1 writer.
        self.is_datas            = self.is_datas[:self.idx-1]
        self.tIDs                = self.tIDs[:self.idx-1]
        self.gmds                = self.gmds[:self.idx-1]
        self.mpes                = self.mpes[:self.idx-1]
        self.hor_poss            = self.hor_poss[:self.idx-1]
        self.ver_poss            = self.ver_poss[:self.idx-1]
        self.zs                  = self.zs[:self.idx-1]
        self.z_stds              = self.z_stds[:self.idx-1]
        self.liq_tofs_es         = self.liq_tofs_es[:self.idx-1]
        self.vls                 = self.vls[:self.idx-1]
        self.between_tdc_filess  = self.between_tdc_filess[:self.idx-1]

        roi = np.s_[self.count*self.chunk_size:(self.count*self.chunk_size) + self.idx-1]
        dsets['tID'][roi]                = self.tIDs
        dsets['local_DAQ_running'][roi]  = self.is_datas
        dsets['z'][roi]                  = self.zs
        dsets['z_std'][roi]              = self.z_stds
        dsets['gmd'][roi]                = self.gmds
        dsets['mpe'][roi]                = self.mpes
        dsets['hor_pos'][roi]            = self.hor_poss
        dsets['ver_pos'][roi]            = self.ver_poss
        if dsets.get('liq_tofs_e') is not None:
            dsets['liq_tofs_e'][roi]     = self.liq_tofs_es
        dsets['vls'][roi]                = self.vls
        dsets['between_tdc_files'][roi]  = self.between_tdc_filess


class TDCIterator:
    """
    Walks the per-train TDC events file-by-file.

    All three TDC channels (1 = electron, 2 = ion, 3 = liquid-jet electron)
    are decoded from every file regardless of the active config. The
    ``config`` argument only changes what ``__next__`` returns, so callers
    for the unused config still see legitimate (empty) data instead of an
    AttributeError or an extra unwrap step.

    Yields per train:
        - config == 1:  (tID, eventcounts_e,  tofs_e_bunch,
                              eventcounts_i,  tofs_i_bunch)
        - config == 2:  (tID, eventcounts_le, tofs_le_bunch)
    """

    def __init__(self, measurement_fpaths, config: int = 1):
        self._measurement_fpaths = measurement_fpaths
        self.config = int(config)
        if self.config not in (1, 2):
            raise ValueError(f"TDCIterator config must be 1 or 2, got {config!r}")

        # Initialize Decoder object and specify the data that should be decoded
        self.decoder = decoding.Decoder(wanted_data=['channel', 'timedata', 'sweep', 'tagbits'])

        # Get number of sweeps/trainIDs per file specified by the sweep preset
        self.sweeps_per_file = metadata_decoding.from_file(self._measurement_fpaths[0], keywords='swpreset=', paragraph_kw='MPA4A')[0]

        (self.trainIDs_tdc,
         self.eventcounts_e,  self.tofs_e,
         self.eventcounts_i,  self.tofs_i,
         self.eventcounts_le, self.tofs_le) = extract_data_from_single_file(
            self._measurement_fpaths[0], self.decoder, self.sweeps_per_file)
        print('!!!!!!!!!')
        print(self.trainIDs_tdc.shape,
              self.eventcounts_e.shape,  self.tofs_e.shape,
              self.eventcounts_i.shape,  self.tofs_i.shape,
              self.eventcounts_le.shape, self.tofs_le.shape)

        self._index = 0
        self._file_index = 1

    def __iter__(self):
        return self

    def __next__(self):
        self.between_files = False
        if self._index == len(self.trainIDs_tdc):

            if self._file_index < len(self._measurement_fpaths):
                print('Finished TDC file {} of {}.'.format(self._file_index, len(self._measurement_fpaths)))
                (self.trainIDs_tdc,
                 self.eventcounts_e,  self.tofs_e,
                 self.eventcounts_i,  self.tofs_i,
                 self.eventcounts_le, self.tofs_le) = extract_data_from_single_file(
                    self._measurement_fpaths[self._file_index], self.decoder, self.sweeps_per_file)
                self._file_index += 1
                self._index = 0
                self.between_files = True

            else:
                print('Finished TDC file {} of {}.'.format(self._file_index, len(self._measurement_fpaths)))
                raise StopIteration

        # Split each channel's running tofs buffer at the current train boundary.
        # Both configs pay this cost so the unused channels still consume their
        # share of the flat tof stream and stay aligned for later trains.
        tofs_e_bunch,  self.tofs_e  = np.split(self.tofs_e,  [self.eventcounts_e[self._index]])
        tofs_i_bunch,  self.tofs_i  = np.split(self.tofs_i,  [self.eventcounts_i[self._index]])
        tofs_le_bunch, self.tofs_le = np.split(self.tofs_le, [self.eventcounts_le[self._index]])

        self._index += 1
        i = self._index - 1
        if self.config == 1:
            return (self.trainIDs_tdc[i],
                    self.eventcounts_e[i],  tofs_e_bunch,
                    self.eventcounts_i[i],  tofs_i_bunch)
        else:
            return (self.trainIDs_tdc[i],
                    self.eventcounts_le[i], tofs_le_bunch)

    def is_between_files(self):
        return self.between_files

class SDUIterator:
    def __init__(self, measurement_fpaths):
        self._measurement_fpaths = measurement_fpaths
        self._trainIds_z, self._zs, self._z_stds = extract_sdu_data_from_single_file(self._measurement_fpaths[0])
        self._index = 0 
        self._file_index = 1 

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._trainIds_z):
            self._index += 1
            return self._trainIds_z[self._index-1], self._zs[self._index-1], self._z_stds[self._index-1]

        elif self._file_index < len(self._measurement_fpaths):
            self._trainIds_z, self._zs, self._z_stds = extract_sdu_data_from_single_file(self._measurement_fpaths[self._file_index ])
            self._file_index += 1
            self._index = 1
            return self._trainIds_z[0], self._zs[0], self._z_stds[0]
            
        else:
            raise StopIteration

class h5Iterator:
    def __init__(self, h5_paths, keys):
        self._h5_paths = h5_paths
        self._keys = keys
        self._len = len(keys)

        # with h5.File(self._h5_paths[0], 'r') as f:
        #     self._values = [np.array(f[key]) for key in self._keys]
        self.load_file(0)

        # self._values = [np.array(h5.File(self._h5_paths[0], 'r')[key]) for key in self._keys]
        self._index = 0 
        self._file_index = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._values[0]):
            self._index += 1
            return (self._values[key_idx][self._index-1] for key_idx in range(self._len))

        elif self._file_index < len(self._h5_paths):
            # self._values = [h5.File(self._h5_paths[self._file_index], 'r')[key] for key in self._keys]
            # with h5.File(self._h5_paths[self._file_index], 'r') as f:
            #     self._values = [np.array(f[key]) for key in self._keys]
            self.load_file(self._file_index)
            
            self._file_index += 1
            self._index = 1
            return (self._values[key_idx][0] for key_idx in range(self._len))
            
        else:
            raise StopIteration

    def load_file(self, h5_index):
        with h5.File(self._h5_paths[h5_index], 'r') as f:
            self._values = [np.array(f[key]) for key in self._keys]

def decode_tdc_data(tdc_data_folder: str, measurement_name: str, decoded_data_folder: str) -> None:
    """
    Converts the raw TDC data, saved as .lst-files in binary format, to two custom format .dat-files that store the
    trainIDs and corresponding electron and ion time-of-flights respectively. The .dat-files are sorted by trainID.

    At first the binary .lst-file is decoded by the user-written MCS6A_decoding.Decoder object to yield all recorded
    events. The 16bit trainID recorded via the tagbits of the TDC is then checked for errors which are corrected if
    needed before it is converted to the full 32bit trainID. Next, the data is split up into electron and ion events
    and the eventcount corresponding to each trainID is calculated. This processed data is then saved to the .dat-file.

    The custom format for the .dat-files is defined in the following way:
        - 4 bytes:      full trainID                                        (max. value: 4,294,967,296)
        - 2 bytes:      eventcount N                                        (max. value: 65,536)
        - 3 bytes:      tof of one event   } N times                        (max. value: 16,777,216)

        - 4 bytes:      next full trainID
        - ...

    :param tdc_data_folder: absolute fpath of the folder from which the TDC-generated .lst-files should be loaded
    :param measurement_name: basename of the measurement (string in front of the 10-digit trainID of the .lst-files)
    :param decoded_data_folder: aabsolute fpath of the folder to which the decoded data will be saved
    :return: None
    """

    # Get absolute fpaths of all .lst-files of the measurement, sorted by their name and thus by their trainID
    files_in_folder = os.listdir(tdc_data_folder)
    measurement_names = list(sorted(filter(re.compile(measurement_name + r'_\d{10}.lst').match, files_in_folder)))
    measurement_fpaths = [tdc_data_folder + '/' + measurement_name for measurement_name in measurement_names]
    
    if len(measurement_fpaths) == 0:
        raise ValueError(f'No files with measurement_name "{measurement_name}" found in folder "{tdc_data_folder}".')
    
    # Initialize Decoder object and specify the data that should be decoded
    decoder = decoding.Decoder(wanted_data=['channel', 'timedata', 'sweep', 'tagbits'])
    
    # Get number of sweeps/trainIDs per file specified by the sweep preset
    sweeps_per_file = metadata_decoding.from_file(measurement_fpaths[0], keywords='swpreset=', paragraph_kw='MPA4A')[0]
    
    # Declare file paths the decoded data will be saved in
    electron_tof_fpath = fr'{decoded_data_folder}/{measurement_name}_electron_tof.dat'
    ion_tof_fpath = fr'{decoded_data_folder}/{measurement_name}_ion_tof.dat'

    # Create and open the files the decoded data will be saved in; "mode='wb'" -> write in binary
    f_electron = open(electron_tof_fpath, mode='wb')
    f_ion = open(ion_tof_fpath, mode='wb')

    # Iterate over every .lst TDC file of the run
    for fpath in tqdm(measurement_fpaths):

        # Decode and process the binary data from one .lst-file.
        # The function now also returns channel-3 (liq electron) counts/tofs; this
        # legacy writer only stores the e/i channels, so the chan-3 outputs are dropped.
        (trainIDs, eventcounts_e, tofs_e, eventcounts_i, tofs_i,
         _eventcounts_le, _tofs_le) = extract_data_from_single_file(fpath, decoder, sweeps_per_file)

        # Save the decoded and processed data
        write_tof_data_to_file(f_electron, trainIDs, eventcounts_e, tofs_e)
        write_tof_data_to_file(f_ion, trainIDs, eventcounts_i, tofs_i)

    # Close the decoded data files
    f_electron.close()
    f_ion.close()


def extract_data_from_single_file(fpath: str, decoder: decoding.Decoder, sweeps_per_file: int) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decodes a single TDC-generated .lst-file, preprocessed the data and returns it as Numpy arrays.

    First, the binary .lst-file is decoded by the user-written MCS6A_decoding.Decoder object to yield all recorded
    events of that acquisition. Any inconsistencies in the 16bit trainID recorded via the tagbits of the TDC are then
    corrected. After this, the 32bit trainID is reconstructed from the 16bit trainID. The data is separated by TDC
    channel into:

        - channel 1 -> electron tofs (config 1, ``tofs_e``)
        - channel 2 -> ion      tofs (config 1, ``tofs_i``)
        - channel 3 -> liquid-jet electron tofs (config 2, ``liq_tofs_e``)

    Channels that record no events in this file simply return empty arrays; the
    function never raises when a channel is unused by the active configuration.

    Note: 17.12.2020: Added LARGEST_ALLOWED_TOF parameter to stop the tof from overflowing 3 bytes in the custom-
                      format .dat-file.

    :param fpath: absolute filepath of one TDC-generated .lst-file
    :param decoder: user-written decoder object that is used for decoding the raw TDC data
    :param sweeps_per_file: number of sweeps and thus trainIDs the file should contain
    :return: tuple of 7 Numpy arrays:
            1. trainIDs:        all trainIDs contained in the file (length: sweeps_per_file)
            2. eventcounts_e:   #events on channel 1 (electron, config 1) per trainID
            3. tofs_e:          all tof events on channel 1 (length: sum(eventcounts_e))
            4. eventcounts_i:   #events on channel 2 (ion, config 1) per trainID
            5. tofs_i:          all tof events on channel 2 (length: sum(eventcounts_i))
            6. eventcounts_le:  #events on channel 3 (liq electron, config 2) per trainID
            7. tofs_le:         all tof events on channel 3 (length: sum(eventcounts_le))
    """
    # Maximum value that can be stored in an unsigned 3-byte integer; Larger tofs are removed to prevent overflows
    # largest_allowed_tof = 2**24-1
    # Maximum allowed tof should not exceed the 50 bunches at 100kHZ or 100 bunches at 200kHz
    largest_allowed_tof = np.uint32(9969.225 * 405)  # FIXME: Do this? Use variable here?

    # Get a full 32bit trainID from the filename of the TDC-generated .lst-file; with this reference the 16bit trainIDs
    # saved for each event by the TDC can be reconstructed
    trainID_at_start_32bit = int(Path(fpath).stem[-10:])
    print('\nFile:',trainID_at_start_32bit)

    # Decode the .lst-file with the decoder object
    [channel, timedata, sweep, tagbits], _ = decoder.decode_file(fpath)
    sweep=np.int32(sweep) #convert to 32bit int?
    tagbits=np.int32(tagbits) #convert to 32bit int?

    # Convert the raw data to a pandas DataFrame (each row contains data about one event)
    df = pd.DataFrame(data={'channel': channel, 'timedata': timedata, 'sweep': sweep, 'trainID': tagbits})
    
    # Drop empty rows (generated by TDC but does not contain data; channel is 1-based, so channel=0 shows empty data)
    # and tofs that are to large
    df = df[(df['channel'].to_numpy() > 0) & (df['timedata'].to_numpy() <= largest_allowed_tof)]
    
    # Check for inconsistencies in the trainID column and correct them if necessary
    if len(df) > 0:
        df = check_and_correct_trainIDs(df, trainID_at_start_32bit)     # NEW: Extend sweepcounter and trainID to 32bit

    #sort data by sweep number (get rid of any jitters) and timing data to avoid sorting again later
    df=df.sort_values(by=['sweep', 'timedata'])#, 'channel', 'timedata'])#, inplace=True)

    if sweeps_per_file > 500000:
        # if sweep preset > 14 hours, probably continuous scan (set 2^32-1) or manual stop
        # we need to determine the actual amount of sweeps in the file        
        sweeps_per_file=df.iloc[len(df)-1,2]    # read 32 bit sweep number of last event in file (is sorted already)
    elif len(df)>0:
        alt_sweeps=df.iloc[len(df)-1,3]-df.iloc[0,3]+1    # read differences between bunchIDs (is sorted already)
        if alt_sweeps>sweeps_per_file:
            sweeps_per_file=alt_sweeps
            print('Longer bunchIDs intervall in file than sweeps. Lost shots or fast trigger?')
    print('Sweeps:',sweeps_per_file)

    # Calculate array of all trainIDs that should be present in the .lst-file
    if len(df) > 0:
        trainIDs=np.arange(sweeps_per_file)+df.iloc[0,3]-df.iloc[0,2]+1

        # Drop events whose trainIDs fall outside the expected range.
        # `check_and_correct_trainIDs` can leave a handful of stragglers when
        # the sweep↔tagbit alignment had to be guessed (the "Deviations between
        # sweeps and tag bits detected!" warning above) — they'd otherwise
        # break the boolean-mask assignment inside calc_eventcount_per_trainID.
        tid_min, tid_max = int(trainIDs[0]), int(trainIDs[-1])
        tid_col = df['trainID'].to_numpy()
        in_range = (tid_col >= tid_min) & (tid_col <= tid_max)
        n_dropped = int((~in_range).sum())
        if n_dropped > 0:
            print(f"WARNING: dropping {n_dropped} event(s) with trainIDs "
                  f"outside expected range [{tid_min}, {tid_max}]")
            df = df[in_range]
    else:
        # Added this after files with no real data made the script throw an exception
        # Problem: Because not a single event is listed in the file, one can not know the exact offset between the
        # sweep and the trainID; Consequently one has to estimate the offset with the help of the file name
        # From reconstruct_32bit_trainID(): "Usually the first trainID recorded by the TDC is around 4-8 integers
        # higher than the trainID in the respective filename."
        # trainID_offset = trainID_at_start_32bit + 5
        trainIDs = np.arange(sweeps_per_file) + trainID_at_start_32bit + 6

    # Split events by TDC channel:
    #   channel 1 -> electron (config 1, tofs_e)
    #   channel 2 -> ion      (config 1, tofs_i)
    #   channel 3 -> liquid-jet electron (config 2, liq_tofs_e)
    # Channels with no events for the active config remain empty; this is fine.
    electron_events = df[df['channel'] == 1]
    ion_events      = df[df['channel'] == 2]
    liq_e_events    = df[df['channel'] == 3]

    # Per-trainID event counts for each channel.
    eventcounts_e  = calc_eventcount_per_trainID(electron_events, trainIDs)
    eventcounts_i  = calc_eventcount_per_trainID(ion_events,      trainIDs)
    eventcounts_le = calc_eventcount_per_trainID(liq_e_events,    trainIDs)

    # Get all tofs as Numpy arrays sorted by their sweep and thus trainID and timedata.
    # Sorting was applied to the parent df above, so per-channel sub-frames are already sorted.
    tofs_e  = electron_events['timedata'].to_numpy()
    tofs_i  = ion_events['timedata'].to_numpy()
    tofs_le = liq_e_events['timedata'].to_numpy()

    return trainIDs, eventcounts_e, tofs_e, eventcounts_i, tofs_i, eventcounts_le, tofs_le


def calc_eventcount_per_trainID(df: pd.DataFrame, all_trainIDs: np.ndarray) -> np.ndarray:
    """
    Calculates the number of events for each trainID in the DataFrame df.

    In case that a trainID has no events, it is not listed in df. To cope with this, an array of zeros is created with
    each element corresponding to one trainID. The eventcounts that are greater than 0 are then inserted into this
    array. This yields an array of the same size as all_trainIDs with eventcounts corresponding to each trainID.

    As the events generated by the TDC are split up by electron and ion events, the DataFrame df only contains events
    of one type in the current implementation.

    :param df: DataFrame containing the raw decoded data from single TDC-generated .lst-file (only for ion OR electron)
                -> Columns: ['sweep', 'trainID', ...]
    :param trainID_at_start_32bit: full 32bit trainID stored in the filename of each acquisition
    :param all_trainIDs: Numpy array of all trainIDs that should be present in the single .lst-file
    :return: all_eventcounts: Numpy array of all eventcounts (data corresponding to trainID at same index)
    """
    # Count events per trainID, then realign onto the full `all_trainIDs`
    # axis by trainID *value*. `reindex(fill_value=0)` produces a row for
    # every trainID we expect (zero where the channel had no events) and
    # silently drops any unique trainIDs in `df` that fall outside
    # `all_trainIDs` — without `reindex` the assignment relied on the
    # value_counts() order matching ascending all_trainIDs, which breaks
    # when sweep↔tagbit alignment drifts.
    if len(df) == 0:
        return np.zeros(all_trainIDs.shape[0], dtype=np.uint16)
    eventcount_per_train = df['trainID'].value_counts(sort=False)
    all_eventcounts = (
        eventcount_per_train
        .reindex(all_trainIDs, fill_value=0)
        .to_numpy()
        .astype(np.uint16)
    )
    return all_eventcounts


def check_and_correct_trainIDs(df: pd.DataFrame, trainID_at_start_32bit: int) -> pd.DataFrame:
    """
    Checks for errors in the "trainID" column of the DataFrame df and corrects them. Corrected are errors that are
    either caused by a faulty tagbit that never turns on or lags in the tagbit update. Both result in inconsistencies in
    the trainID like decreases or changing offsets between the trainID and the sweep counter.

    First, the difference between consecutive trainIDs is calculated and checked for a decrease which would signal a
    broken tagbit. Additionally, the offset between the trainID and the sweep counter is calculated to make sure this
    is constant over all rows. In case such an inconsistency is found, the correct and constant offset between trainID
    and sweep is found an then used to calculate the correct trainID for every row of the df.

    As the functions detects a broken tagbit by finding a decrease in the tagbit value, the highest-order tagbit that
    can be corrected is tag_max = floor(log_2(N-1)) where N is the number of bunchtrains/sweeps per .lst-file
    (e.g. N=50 -> tag_max=5 -> bit5). The first tagbit (bit0) can not be detected as broken.

    For multiple runs on 12-12-2020 bit1 of the tagbit cable was broken resulting in wrong and inconsistent
    tagbits and consequently wrong trainIDs.

    Note: - Updated this on 17-12-2020, because the df_section was to short to detect a fault at bit1, resulting in
            wrong trainIDs in the decoded data.
          - In run Ar_longscan2_275eV in file Ar_longscan2_275eV_0929507419.lst at sweep 1 the trainID switches once
            from 10336 to 10365, resulting in wrong reconstructed trainIDs. This was manually corrected in the .dat-file
            Excerpt from the raw .lst-file data:
                      channel  timedata  sweep  trainID_16bit
                1155        2   2406910      1          10336
                1156        2   2406635      1          10365  <--
                1157        2   2406849      1          10336
                1158        2   2407105      1          10336

    :param df: DataFrame containing the raw decoded data from a single TDC-generated .lst-file
                -> Columns: ['sweep', 'trainID', ...]
    :return: df: the same DataFrame, but with corrected trainID values
    
    NEW UPDATE 13.10.2023: Extend sweepcounter to 32bit and correct for overflows
    and return 32bit trainIDs (also correct for initial offset to 32bit ID)
    """

    # reconstruct 32bit sweep number
    sweeps=df['sweep'].to_numpy(dtype=np.int32)
    sweep_diff=np.diff(sweeps) # sweep_diff[i] = sweeps[i+1]-[sweeps[i]
    sweep_of=np.flatnonzero(sweep_diff<-32767)  # determine events before 16bit wraparound of sweep#
    sweep_jc=np.flatnonzero(sweep_diff>32767)   # determine events where sweep# jitter occurs during wraparound
    i=0
    while i<len(sweep_of):  # go through all negative jumps of sweep#
        sweeps[sweep_of[i]+1:]+=(i+1)*65536
        i+=1
    i=0
    while i<len(sweep_jc):  # still need to account for possible jitter of sweep# during wraparound
        # positive jump can only occur after wraparound, i.e., these events (an all behind it)
        # have now a too high sweep# (bevause it also leads to doubled wraparounds)
        sweeps[sweep_jc[i]+1:]-=65536
        i+=1
    #sweep counter should now correct
    # can be used to directly reconstruct the 32bit train IDs
    
    # Get the trainID as a Numpy array to speed up execution
    trainID = df['trainID'].to_numpy(dtype=np.int32) # convert to signed 32bit

    # Calculate the difference between two consecutive trainIDs (diff[0] = trainID[1] - trainID[0])
    trainID_diff = np.append(np.diff(trainID), 0)
    
    train_of=np.flatnonzero(trainID_diff<-32767)   #correct for overflows, basically the same as with sweep number
    train_jc=np.flatnonzero(trainID_diff>32767)   # determine events where trainID jitter occurs during wraparound
    i=0
    while i<len(train_of):   # go through all negative jumps of trainID
        trainID[train_of[i]+1:]+=(i+1)*65536
        i+=1
    i=0
    while i<len(train_jc):  # still need to account for possible jitter of trainID during wraparound
        # positive jump can only occur after wraparound, i.e., these events (an all behind it)
        # have now a too high trainID (bevause it also leads to doubled wraparounds)
        sweeps[train_jc[i]+1:]-=65536
        i+=1
    #train ID now does not wrap around at 65536
    
    # diff now between trainID and sweep# as the overflows should be removed now
    trainID_diff = np.append(np.diff(trainID-sweeps), 0)
    # use difference of trainID and sweeps, as this is way more precise
    
    # Get rows where the trainID decreases in the row following it (if the trainID is 65535 this is caused by an
    # expected 16bit overflow and should not be counted)
    #rows_with_decrease_mask = (trainID_diff < 0) # & (trainID != 65535) # this can't happen now
    
    rows_with_decrease_mask = (trainID_diff != 0)    # catch all deviations

    if np.any(rows_with_decrease_mask):
        print('\nDeviations between sweeps and tag bits detected!')
        print('WARNING: Try to figure out the most common distance between sweep# and tagbits. May not be ideal.')
        diff_val, diff_freq = np.unique(trainID-sweeps,return_counts=True)
        trainID_offset=diff_val[np.argmax(diff_freq)]
    else:
        trainID_offset = trainID[0]-sweeps[0]
        print('Perfect match of sweep# and tag bits (train ID)! Offset:', trainID_offset)


    # If there is a decrease in the trainID: find row with correct trainID
    #if np.any(rows_with_decrease_mask):
        """
        # Get the index of the first row right before a decreases of the trainID; This row has the correct trainID value
        #correct_value_pair_index = np.argmax(rows_with_decrease_mask)
        correct_value_pair_index = np.argmin(trainID[0]-sweeps[0])
        trainID_offset = trainID[correct_value_pair_index]-sweeps[correct_value_pair_index]
        print(trainID_diff[correct_value_pair_index], trainID[correct_value_pair_index], sweeps[correct_value_pair_index])
        print(trainID[correct_value_pair_index+1], sweeps[correct_value_pair_index+1])
        # Get the corresponding row
        correct_value_pair = df.iloc[correct_value_pair_index, :]
        # Calculate the correct offset between the sweep and the trainID
        trainID_offset = correct_value_pair['trainID'] - correct_value_pair['sweep']

        correct_trainIDs = True
        
    else:
        # Check for rows where the trainID lags behind, e.g. the difference between trainID and sweep is not constant

        # Calculate the difference(s) between trainID and sweep and get the corresponding counts:
        # Resulting series looks like this:     delta    appearances in data
        #                                       52824    34620
        #                                       52822    31084
        trainID_sweep_delta_freqs = (df['trainID'] - df['sweep']).value_counts()

        # Only look at positive differences (e.g. trainID > sweep) as negative differences are caused by the expected
        # 16bit overflow of the tagbits; This overflow is handled by the function reconstruct_32bit_trainID() later on
        trainID_sweep_delta_freqs = trainID_sweep_delta_freqs[trainID_sweep_delta_freqs.index > 0]

        if len(trainID_sweep_delta_freqs) > 1:
            # The correct offset between trainID and sweep is the one with the highest occurrence in the data
            trainID_offset = trainID_sweep_delta_freqs.index[0]
            correct_trainIDs = True

    if correct_trainIDs:
        # Reconstruct the trainID for every event (row of df) from the corresponding sweep number
        # This calculation can overflow; However, just like the regular 16bit overflow (when the tagbits work correctly)
        # this overflow will be corrected for by the reconstruct_32bit_trainID() function
        df['trainID'] = (df['sweep'] + trainID_offset).astype(np.uint16)
        """
        
    # Get bits 16...31 (last 16 bit) of the trainID_at_start
    trainID_at_start_16bit = np.uint32(int(bin(trainID_at_start_32bit)[-16:], 2))

    # Get bits 0...15 of the trainID_at_start; this data is missing in the trainID from the TDC
    trainId_at_start_last16bit = np.uint32(trainID_at_start_32bit - trainID_at_start_16bit)
    
    if trainID_at_start_32bit > trainId_at_start_last16bit + trainID_offset + 1: # overflow directly at start of the scan
        trainId_at_start_last16bit+=65536

    # Concatenate the the bits 0...15 from the trainID from the TDC and the bits 16...31 calculated before
    trainID+=trainId_at_start_last16bit

    df['trainID'] = trainID.astype(np.uint32)
    df['sweep'] = sweeps.astype(np.uint32)

    return df


def reconstruct_32bit_trainID_tdc(trainIDs_16bit: np.ndarray, trainID_at_start_32bit: int) -> np.ndarray:
    """
    Reconstructs the full 32bit trainID from the 16bit trainID encoded in the tagbits in the TDC-generated .lst-files.

    The full trainID is stored in the filename of each TDC-generated .lst-file. This full trainID then acts as a
    reference to calculate the full 32bit trainID.

    The last 16 bits of the full 32bit trainID found in the filename are the ones missing in the 16bit tagbit values.
    Thus, to reconstruct the full trainID, these last 16 bits only have to be added to the first 16 bits from the
    tagbits. In case the 16bit trainID encoded in the tagbits overflows (is smaller than the trainID from the filename),
    the overflow-bit is also added.

    Usually the first trainID recorded by the TDC is around 4-8 integers higher than the trainID in the respective
    filename.

    :param trainIDs_16bit: Numpy array of 16bit trainIDs encoded in the tagbits in the TDC-generated .lst-files
    :param trainID_at_start_32bit: full 32bit trainID stored in the filename of each acquisition
    :return: reconstructed_32bit_trainIDs: Numpy array of reconstructed full 32bit trainIDs
    """

    # Get bits 16...31 (last 16 bit) of the trainID_at_start
    trainID_at_start_16bit = int(bin(trainID_at_start_32bit)[-16:], 2)

    # Get bits 0...15 of the trainID_at_start; this data is missing in the trainID_16bit from the TDC
    trainId_at_start_last16bit = trainID_at_start_32bit - trainID_at_start_16bit

    # Concatenate the the bits 0...15 from the trainID_16bit from the TDC and the bits 16...31 calculated before
    reconstructed_32bit_trainIDs = trainIDs_16bit + trainId_at_start_last16bit

    # Because the trainID never decreases, a decrease in the value of the bits 0...15 is caused by an overflow
    # <overflow_bits> is a Numpy array containing the missing bit in case the trainID had an overflow
    overflow_bits = np.where(trainIDs_16bit < trainID_at_start_16bit, 2**16, 0).astype(np.uint32)
    reconstructed_32bit_trainIDs = reconstructed_32bit_trainIDs + overflow_bits
    
    return reconstructed_32bit_trainIDs

def decode_sdu_data(sdu_data_folder: str, measurement_name: str, decoded_data_folder: str) -> None:
    """
    Converts the raw SDU data, saved as .txt-files, to two custom format .dat-files that store the trainIDs and
    corresponding SDU z-position and z-position standard deviation respectively. The .dat-files are sorted by trainID.

    For this the .txt-file is read and the data grouped by the trainID. This data is subsequently saved in the
    .dat-file.

    The custom format for the .dat-files is defined in the following way:
        - 4 bytes:      full trainID                            (max. value: 4,294,967,296)
        - 8 bytes:      sdu z-position as a double              (max. value: double)

        - 3 bytes:      next full trainID
        - ...

    Note: 18.12.2020: Added trainID fix

    :param sdu_data_folder: absolute fpath of the folder from which the SDU-generated .txt-files should be read
    :param measurement_name: basename of the measurement (string in front of the 10-digit trainID of the .txt-files)
    :param decoded_data_folder: absolute fpath of the folder to which the decoded data will be saved
    :return:
    """

    # Get absolute fpaths of all .txt-files of the measurement, sorted by their name and thus by their trainID
    files_in_folder = os.listdir(sdu_data_folder)
    measurement_names = list(sorted(filter(re.compile(measurement_name + r'_\d{10}.txt').match, files_in_folder)))
    measurement_fpaths = [sdu_data_folder + '/' + measurement_name for measurement_name in measurement_names]
    
    if len(measurement_fpaths) == 0:
        raise ValueError(f'No files with measurement_name "{measurement_name}" found in folder "{sdu_data_folder}".')
    
    # Declare the file paths the decoded data will be saved in
    sdu_pos_fpath = fr'{decoded_data_folder}/{measurement_name}_sdu_position.dat'
    sdu_pos_std_fpath = fr'{decoded_data_folder}/{measurement_name}_sdu_pos_std.dat'

    # Create and open the files the decoded data will be saved in; "mode='wb'" -> write in binary
    decoded_sdu_pos_file = open(sdu_pos_fpath, mode='wb')
    decoded_sdu_pos_std_file = open(sdu_pos_std_fpath, mode='wb')

    # Iterate over every SDU-generated .txt-file file of the run
    for fpath in tqdm(measurement_fpaths):  # FIXME: Add description? , desc='tst'
        # Decode the data from the .txt file and group by trainID
        trainIds, zs, z_stds = extract_sdu_data_from_single_file(fpath)

        # Save the decoded and processed data
        write_pos_data_to_file(decoded_sdu_pos_file, trainIds, zs)
        write_pos_data_to_file(decoded_sdu_pos_std_file, trainIds, z_stds)

    # Close the decoded data files
    decoded_sdu_pos_file.close()
    decoded_sdu_pos_std_file.close()


def extract_sdu_data_from_single_file(fpath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decodes a single SDU-generated .txt-file and returns the processed data.

    For this the .txt-file is read and the data grouped by the trainID. Because the trainIDs sometimes lag behind, they
    are checked for inconsistencies and corrected if needed. Three vectors containing the trainID, the z-position and
    the z-position std are then returned.

    :param fpath: The absolute filepath of the SDU-generated .txt-file
    :return: 3 Numpy arrays of length n:
            1. trainIds: trainID
            2. z_positions: z-position (corresponding to the trainID at the same index)
            2. z_stds: z-position std (corresponding to the trainID at the same index)
    """
    # Read the SDU-generated .txt-file and convert it to a pandas DataFrame
    df = pd.read_table(fpath, skiprows=1, encoding='cp1252')

    # Rename bID to trainID to prevent confusion
    df = df.rename(columns={'bID': 'trainID'})

    # Check for inconsistencies in the trainID and correct trainIDs that lag behind
    df = check_and_correct_trainIDs_sdu(df)

    trainIds = df['trainID'].to_numpy()
    z_positions = df['z [nm]'].to_numpy()
    z_stds = df['z std'].to_numpy()

    return trainIds, z_positions, z_stds


def check_and_correct_trainIDs_sdu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for inconsistencies in the trainID values of the DataFrame and corrects them.

    The SDU writes one line of data per bunchtrain and thus per trainID to the .txt-files. However, the trainIDs
    the SDU-software writes sometimes lag behind. This results in two (or more, only observed once) consecutive rows in
    the .txt-file having the same trainID. After this, one (or more) trainID is skipped and the trainIDs are in
    synchronisation again.

    This function detects these skipped trainIDs (trainIDs are never skipped inside a .txt-file otherwise). If such a
    "jump" is detected, all trainIDs are reconstructed with the help of their indices in the DataFrame.

    Note: These inconsistencies in the trainID were first noticed on 18.12.2020, resulting in the creation of this
    function.

    :param df: DataFrame containing the raw decoded data from a single SDU-generated .txt-file
                -> Columns: ['trainID', ...]
    :return: df: the same DataFrame, but with corrected trainIDs.
    """

    # Calculate the difference between two consecutive trainIDs (diff[1] = trainID[1] - trainID[0])
    trainID_diff = np.insert(np.diff(df['trainID'].to_numpy().astype(np.int32)), 0, 0)

    # Get rows where the difference to the row before more than one; here the bunchID jumps back to the right value
    rows_with_jump_mask = trainID_diff > 1

    # If there are any jumps: correct the trainID values
    if np.any(rows_with_jump_mask):
        # Get the index of the first row that skips a trainID; This row has the correct trainID value
        correct_value_pair_index = np.argmax(rows_with_jump_mask)
        
        # Get the corresponding row
        correct_value_pair = df.iloc[correct_value_pair_index, :]

        # Calculate the constant difference between the index and the corresponding trainID
        trainID_offset = int(correct_value_pair['trainID'] - correct_value_pair.name)

        # Reconstruct the trainIDs
        df['trainID'] = df.index + trainID_offset
    else:
        # Correct a lag of the trainID in the last 10 rows where the jump following it is not saved in the .txt-file
        
        # Get the rows where the trainID does not increase
        rows_with_no_increase = df['trainID'].iloc[-10:].diff() == 0
        if any(rows_with_no_increase):
            # Get the first row where the trainID still increases; This row has the correct trainID value
            correct_value_pair_index = rows_with_no_increase[~rows_with_no_increase].index[0]
            correct_value_pair = df.loc[correct_value_pair_index, :]

            # Calculate the constant difference between the index and the corresponding trainID
            trainID_offset = int(correct_value_pair['trainID'] - correct_value_pair.name)

            # Reconstruct the trainIDs of the last 5 rows
            df.iloc[-5:, 6] = df.index[-5:] + trainID_offset

    return df


# ---------------------------------------------------------------------------
# Raw-H5 dataset paths
# ---------------------------------------------------------------------------

_GMD_INDEX = "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall/index"
_GMD_VALUE = "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall/value"
_MPE_INDEX = "/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean photon energy/index"
_MPE_VALUE = "/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean photon energy/value"
_HOR_INDEX = "/FL2/Photon Diagnostic/GMD/Average beam position/position hall horizontal/index"
_HOR_VALUE = "/FL2/Photon Diagnostic/GMD/Average beam position/position hall horizontal/value"
_VER_INDEX = "/FL2/Photon Diagnostic/GMD/Average beam position/position hall vertical/index"
_VER_VALUE = "/FL2/Photon Diagnostic/GMD/Average beam position/position hall vertical/value"
_VLS_INDEX = "/FL2/Support Infrastructure/Gotthard/images/index"
_VLS_VALUE = "/FL2/Support Infrastructure/Gotthard/images/value"

_DEFAULT_TRAIN_LENGTH = {1: 101, 2: 110}
_DEFAULT_CHUNK_SIZE   = {1: 1000, 2: 200}

# How many bunches per train carry real GMD / SDU data. For cfg 2 the
# Gotthard VLS has more lines than the FEL has pulses (110 vs 101), so
# the non-VLS fields fill only the first 101 positions of the
# train_length=110 bunch axis and the remainder is NaN. ``None`` means
# "use every bunch slot", matching the cfg 1 layout where GMD spans the
# whole train.
_DEFAULT_DATA_BUNCHES = {1: None, 2: 101}


def _fold_into_bunches(tofs_raw, train_length, folding_parameter, edges):
    """
    Split a sorted per-train flat TOF stream into per-bunch tof arrays.

    Replaces the original ``[tofs_raw[(tofs_raw > b*fp) & (tofs_raw < (b+1)*fp)] - b*fp
    for b in range(train_length)]`` list comprehension, which performs
    ``train_length`` (400 or 100) boolean masks over the full array. Two
    vectorised ``searchsorted`` calls find every bunch boundary in one
    pass; the remaining Python loop just slices the sorted array and
    subtracts a scalar per bunch.

    Parameters
    ----------
    tofs_raw : np.ndarray
        TOF values for one train, sorted ascending in ``timedata`` order
        (the TDC decoder sort already guarantees this).
    train_length, folding_parameter : int, float
    edges : np.ndarray
        Pre-computed ``np.arange(train_length + 1) * folding_parameter``.
        Passed in so we don't re-allocate it per train.

    Returns
    -------
    list of np.ndarray
        Length ``train_length``; entry ``b`` holds the TOFs of bunch
        ``b`` after subtracting ``b * folding_parameter``.
    """
    # Original bounds were strict: tof > b*fp & tof < (b+1)*fp.
    # 'right' on the lower edge and 'left' on the upper edge replicate that.
    starts = np.searchsorted(tofs_raw, edges[:-1], side="right")
    ends   = np.searchsorted(tofs_raw, edges[1:],  side="left")
    out = [None] * train_length
    for b in range(train_length):
        out[b] = tofs_raw[starts[b]:ends[b]] - edges[b]
    return out


def _has_glob_meta(name: str) -> bool:
    """True if ``name`` contains shell-style glob metacharacters."""
    return any(c in name for c in "*?[")


def _list_measurement_files(base_dir: Path, measurement_name: str, ext: str) -> list:
    """
    Locate files matching ``<measurement_name>_<10 digits>.<ext>`` under
    ``base_dir``, supporting both single literal names and shell-style
    glob patterns (``*``, ``?``, ``[``).

    Search strategy
    ---------------
    Literal name (no glob metacharacters):
        - Try the per-measurement subfolder ``base_dir / measurement_name`` first.
        - Fall back to ``base_dir`` itself (flat layout).

    Glob pattern:
        - Scan every subfolder of ``base_dir`` whose name matches the
          pattern, AND ``base_dir`` itself (handles both per-scan
          subfolder and flat layouts in one pass).

    Files in each searched folder are kept only when:
        1. Their name matches ``<measurement_name>_*.<ext>`` (fnmatch).
        2. They end with ``_<10 digits>.<ext>`` (the standard suffix).

    Duplicates (e.g. via symlinks) are removed. The result is sorted by
    the 10-digit trainID encoded in the suffix, so scan boundaries fall
    in chronological order even when multiple scans are concatenated.
    """
    if not base_dir.is_dir():
        return []
    name_pattern = measurement_name + "_*." + ext
    suffix_re = re.compile(r"_\d{10}\." + re.escape(ext) + "$")

    folders = []
    if _has_glob_meta(measurement_name):
        for child in sorted(base_dir.iterdir()):
            if child.is_dir() and fnmatch.fnmatchcase(child.name, measurement_name):
                folders.append(child)
        folders.append(base_dir)
    else:
        subdir = base_dir / measurement_name
        if subdir.is_dir():
            folders.append(subdir)
        else:
            folders.append(base_dir)

    seen = set()
    matches = []
    for folder in folders:
        try:
            entries = os.listdir(folder)
        except OSError:
            continue
        for name in entries:
            if not fnmatch.fnmatchcase(name, name_pattern):
                continue
            if not suffix_re.search(name):
                continue
            p = str(folder / name)
            if p in seen:
                continue
            seen.add(p)
            matches.append(p)
    matches.sort(key=lambda p: int(Path(p).stem[-10:]))
    return matches


def _sanitize_output_stem(measurement_name: str) -> str:
    """Strip glob metacharacters from ``measurement_name`` for use as a filename."""
    cleaned = re.sub(r"[\*\?\[\]]", "", measurement_name).strip("_")
    return (cleaned or "combined") + "_combined"


def _h5_file_order_key(path_str: str):
    """
    Sort raw H5 files by the numeric ``_fileNN_`` index in their name.

    Lexicographic sorting puts ``file100`` before ``file11``, which
    desynchronises every train-aligned iterator and triggers
    ``mpe/gmd/... overshot`` errors at the wrap-around. Numeric sort
    keeps the files in train-ID order. Falls back to plain name sort
    when the pattern is absent.
    """
    name = Path(path_str).name
    if "_file" in name:
        tail = name.split("_file", 1)[1]
        num_str = tail.split("_", 1)[0]
        if num_str.isdigit():
            return (0, int(num_str), name)
    return (1, 0, name)


def _skip_stale_raw_h5(it, next_tID, next_val, current_tID, last_tID,
                       counters, label):
    """
    Drop iterator rows whose train ID has fallen behind ``current_tID``.

    Consecutive raw H5 files for the same run can have overlapping
    train ranges (e.g. file N's last tIDs reappear at the start of
    file N+1). Once the iterator crosses such a boundary it can yield
    train IDs we already passed, so the strict ``>`` overshot check
    fires. This helper consumes those duplicates until either the
    iterator catches up (``next_tID >= current_tID``) or runs out.

    Returns
    -------
    next_tID, next_val, exhausted
        ``next_tID`` clamped to ``last_tID + 1`` on exhaustion so
        downstream comparisons take the "<" / NaN branch.
        ``exhausted`` is True iff the iterator ran out.
    """
    exhausted = False
    while next_tID < current_tID:
        try:
            next_tID, next_val = next(it)
            counters[label] += 1
        except StopIteration:
            next_tID = last_tID + 1
            next_val = None
            exhausted = True
            break
    return next_tID, next_val, exhausted


def _advance_to(it, target_tID, label):
    """
    Pull rows from an ``h5Iterator`` until tID >= target_tID and return that row.

    Parameters
    ----------
    it : h5Iterator
        Iterator yielding ``(tID, value)`` pairs from a raw-H5 dataset.
    target_tID : int
        Train ID to advance to. The first row with ``tID >= target_tID``
        is returned.
    label : str
        Human-readable name used for progress prints.

    Returns
    -------
    tuple
        ``(tID, value)`` for the first row at or above ``target_tID``.
    """
    for t, v in it:
        if t >= target_tID:
            print(f"  {label} reached tID {t} (target {target_tID}).")
            return t, v
    raise RuntimeError(f"{label} iterator exhausted before reaching {target_tID}")


def main(config_no, measurement_name, run_no, output_path=None,
         train_length=None, chunk_size=None,
         max_ecounts=50, max_icounts=120, n_vls_pixels=1280,
         folding_parameter=39876.9, data_bunches=None):
    """
    Decode and align the local-DAQ + raw-H5 streams for one measurement.

    Both configurations follow the same train-ID alignment pattern; only
    the detector layout differs:

    - config 1: TDC channels 1 (electron, ``tofs_e``) and 2 (ion, ``tofs_i``).
    - config 2: TDC channel 3 (liquid-jet electron, ``liq_tofs_e``) plus
      Gotthard VLS spectra read from the raw H5.

    If no TDC ``.lst`` files exist for the measurement, the TOF datasets
    (``tofs_e``/``tofs_i`` for cfg 1, ``liq_tofs_e`` for cfg 2) are
    omitted from the output and ``between_tdc_files`` is written as all
    False. The SDU and raw-H5 streams are still aligned and written
    normally. For cfg 2, ``vls`` is always written regardless.

    Parameters
    ----------
    config_no : int
        1 or 2. Selects the detector layout (and therefore the output schema).
    measurement_name : str
        Basename used to locate the SDU ``.txt`` and TDC ``.lst`` files.
        For each stream the script first tries
        ``config.SDU_DIR / measurement_name`` / ``config.TDC_DIR /
        measurement_name`` (legacy per-measurement subfolder layout)
        and falls back to ``config.SDU_DIR`` / ``config.TDC_DIR``
        themselves when the subfolder does not exist.
    run_no : int
        FLASH run number used to glob the raw H5 files under
        ``config.RAW_H5_DIR``.
    output_path : pathlib.Path, optional
        Output H5 path. Defaults to ``COMBINED_DIR / f"{measurement_name}.h5"``.
    train_length : int, optional
        Bunches per train. Defaults to 400 for config 1 and 100 for config 2.
    chunk_size : int, optional
        Rows per chunked H5 write. Defaults to 1000 for config 1 and
        200 for config 2 (config 2 buffers are larger because of VLS).
    max_ecounts, max_icounts : int
        Zero-padding limits for the per-bunch TDC tof arrays.
    n_vls_pixels : int
        Width of the Gotthard VLS pixel axis (config 2 only).
    folding_parameter : float
        TOF range (100 ps) used to split each sweep's flat tof stream into
        per-bunch lists.
    data_bunches : int, optional
        Number of bunches per train that the non-VLS fields actually
        carry. Bunches in ``[data_bunches, train_length)`` are NaN-filled
        for GMD, ``z``, and ``z_std``. Default is 101 for cfg 2 (matches
        the FEL pulse count when ``train_length = 110``) and ``None``
        for cfg 1 (fills every bunch). VLS always uses the full
        ``train_length``.
    """
    config_no = int(config_no)
    if config_no not in (1, 2):
        raise ValueError(f"config must be 1 or 2, got {config_no!r}")
    if train_length is None:
        train_length = _DEFAULT_TRAIN_LENGTH[config_no]
    if chunk_size is None:
        chunk_size = _DEFAULT_CHUNK_SIZE[config_no]
    if data_bunches is None:
        data_bunches = _DEFAULT_DATA_BUNCHES[config_no]
    # Effective per-train slot count for non-VLS streams. None = the
    # entire train_length.
    _data_n = train_length if data_bunches is None else int(data_bunches)
    if _data_n > train_length:
        raise ValueError(
            f"data_bunches={data_bunches} cannot exceed train_length={train_length}."
        )

    h5_folder    = str(config.RAW_H5_DIR)
    is_glob_name = _has_glob_meta(measurement_name)
    if output_path is None:
        stem = _sanitize_output_stem(measurement_name) if is_glob_name else measurement_name
        output_path = config.COMBINED_DIR / f"{stem}.h5"
    output_path = Path(output_path)

    print(f"Measurement   : {measurement_name}"
          f"{'  (glob pattern)' if is_glob_name else ''}  (run {run_no})")
    print(f"Config        : {config_no}")
    print(f"SDU base      : {config.SDU_DIR}")
    print(f"TDC base      : {config.TDC_DIR}")
    print(f"Raw H5 dir    : {h5_folder}")
    print(f"Output        : {output_path}")
    print(f"train_length  : {train_length}")
    print(f"data_bunches  : {data_bunches}  "
          f"({'all bunches' if data_bunches is None else f'fills first {_data_n}; rest NaN'})")
    print(f"chunk_size    : {chunk_size}")
    print()

    if not Path(h5_folder).is_dir():
        raise FileNotFoundError(f"Raw H5 folder not found: {h5_folder}")

    # --- Locate SDU .txt files -----------------------------------------
    sdu_fpaths = _list_measurement_files(Path(config.SDU_DIR), measurement_name, "txt")
    if not sdu_fpaths:
        raise ValueError(
            f"No SDU .txt files for measurement '{measurement_name}' "
            f"under {config.SDU_DIR}."
        )
    print(f"Found {len(sdu_fpaths)} SDU .txt files.")
    if is_glob_name:
        # Show which scan(s) the wildcard matched, by file-prefix.
        prefixes = sorted({Path(p).stem[:-11] for p in sdu_fpaths})
        print(f"  SDU scan prefixes matched: {prefixes}")

    # --- Locate TDC .lst files -----------------------------------------
    # TDC is optional: if no .lst files are present, we still write a
    # combined H5 containing the SDU + raw-H5 streams. The TOF datasets
    # (tofs_e/tofs_i in cfg 1, liq_tofs_e in cfg 2) are simply omitted
    # from the output schema and `between_tdc_files` is all False.
    tdc_fpaths = _list_measurement_files(Path(config.TDC_DIR), measurement_name, "lst")
    has_tdc = bool(tdc_fpaths)
    if has_tdc:
        print(f"Found {len(tdc_fpaths)} TDC .lst files.")
        if is_glob_name:
            prefixes = sorted({Path(p).stem[:-11] for p in tdc_fpaths})
            print(f"  TDC scan prefixes matched: {prefixes}")
    else:
        print(f"No TDC .lst files for measurement '{measurement_name}' "
              f"under {config.TDC_DIR}; writing SDU + raw-H5 only.")

    # --- Locate raw H5 files for this run ------------------------------
    h5_paths = sorted(
        glob.glob(os.path.join(h5_folder, f"*run{run_no}*.h5")),
        key=_h5_file_order_key,
    )
    if not h5_paths:
        raise ValueError(f"No raw H5 files for run {run_no} in {h5_folder}")
    print(f"Found {len(h5_paths)} raw H5 files for run {run_no}.")

    # --- Determine train-ID range of the measurement -------------------
    first_tID = int(extract_sdu_data_from_single_file(sdu_fpaths[0])[0][0])
    last_tID  = int(extract_sdu_data_from_single_file(sdu_fpaths[-1])[0][-1])
    print(f"Measurement spans train IDs {first_tID} .. {last_tID} ({last_tID - first_tID + 1} trains).")

    # --- Find the first raw H5 file whose train range overlaps ---------
    first_h5_idx = None
    for path_idx, h5_path in enumerate(h5_paths):
        with h5.File(h5_path, "r") as f:
            tids = f[_GMD_INDEX]
            tids_max = int(np.max(tids[...]))
            print(f"  {Path(h5_path).name}: max train ID = {tids_max}")
            if tids_max > first_tID:
                first_h5_idx = path_idx
                break
    if first_h5_idx is None:
        raise RuntimeError("None of the raw H5 files overlap with the SDU train-ID range.")
    print(f"First raw H5 file with overlap: index {first_h5_idx}")

    # --- Iterators -----------------------------------------------------
    sdu_it     = SDUIterator(sdu_fpaths)
    tdc_it     = TDCIterator(tdc_fpaths, config=config_no) if has_tdc else None
    gmd_it     = h5Iterator(h5_paths[first_h5_idx:], [_GMD_INDEX, _GMD_VALUE])
    mpe_it     = h5Iterator(h5_paths[first_h5_idx:], [_MPE_INDEX, _MPE_VALUE])
    hor_pos_it = h5Iterator(h5_paths[first_h5_idx:], [_HOR_INDEX, _HOR_VALUE])
    ver_pos_it = h5Iterator(h5_paths[first_h5_idx:], [_VER_INDEX, _VER_VALUE])
    if config_no == 2:
        vls_it = h5Iterator(h5_paths[first_h5_idx:], [_VLS_INDEX, _VLS_VALUE])

    # --- Fast-forward each raw-H5 iterator to first_tID ----------------
    next_tID_gmd, next_gmd = _advance_to(gmd_it, first_tID, "gmd")
    next_tID_mpe, next_mpe = _advance_to(mpe_it, first_tID, "mpe")
    next_tID_hor_pos, next_hor_pos = _advance_to(hor_pos_it, first_tID, "hor_pos")
    next_tID_ver_pos, next_ver_pos = _advance_to(ver_pos_it, first_tID, "ver_pos")
    if config_no == 2:
        next_tID_vls, next_vls = _advance_to(vls_it, first_tID, "vls")

    next_tID_z, next_z, next_z_std = sdu_it.__next__()
    if has_tdc:
        if config_no == 1:
            (next_tID_tdc,
             next_eventcounts_e, next_tofs_e,
             next_eventcounts_i, next_tofs_i) = tdc_it.__next__()
        else:
            (next_tID_tdc,
             next_eventcounts_le, next_tofs_le) = tdc_it.__next__()

    # --- Output H5 layout (schema + legacy extras) ---------------------
    data_len = last_tID - first_tID
    if config_no == 1:
        chunk = DataChunk(chunk_size, train_length,
                          max_ecounts=max_ecounts, max_icounts=max_icounts)
    else:
        chunk = DataChunkConfig2(chunk_size, train_length,
                                 max_ecounts=max_ecounts,
                                 n_vls_pixels=n_vls_pixels)

    if output_path.exists():
        output_path.unlink()

    with h5.File(output_path, "w") as f_out:
        # Shared schema fields
        tID_dset               = f_out.create_dataset("tID",               (data_len,),               dtype="double")
        data_flag_dset         = f_out.create_dataset("local_DAQ_running", (data_len,),               dtype="bool")
        z_dset                 = f_out.create_dataset("z",                 (data_len, train_length),  dtype=np.float32)
        z_std_dset             = f_out.create_dataset("z_std",             (data_len, train_length),  dtype=np.float32)
        gmd_dset               = f_out.create_dataset("gmd",               (data_len, train_length),  dtype=np.float32)
        mpe_dset               = f_out.create_dataset("mpe",               (data_len,),               dtype=np.float32)
        hor_pos_dset           = f_out.create_dataset("hor_pos",           (data_len,),               dtype=np.float32)
        ver_pos_dset           = f_out.create_dataset("ver_pos",           (data_len,),               dtype=np.float32)
        between_tdc_files_dset = f_out.create_dataset("between_tdc_files", (data_len,),               dtype="bool")

        # TOF datasets are only written when TDC .lst files are present;
        # otherwise the slots stay None and the chunk dump/finish code
        # skips them. `between_tdc_files` is still emitted (all False) so
        # `load_data` downstream can filter without special-casing.
        tofs_e_dset = tofs_i_dset = liq_tofs_e_dset = None
        if config_no == 1:
            if has_tdc:
                tofs_e_dset = f_out.create_dataset(
                    "tofs_e", (data_len, train_length, max_ecounts),
                    dtype=np.uint32, compression="lzf",
                )
                tofs_i_dset = f_out.create_dataset(
                    "tofs_i", (data_len, train_length, max_icounts),
                    dtype=np.uint32, compression="lzf",
                )
        else:
            if has_tdc:
                liq_tofs_e_dset = f_out.create_dataset(
                    "liq_tofs_e", (data_len, train_length, max_ecounts),
                    dtype=np.uint32, compression="lzf",
                )
            vls_dset = f_out.create_dataset(
                "vls", (data_len, train_length, n_vls_pixels),
                dtype=np.float32, compression="lzf",
            )
            dsets_cfg2 = {
                "tID": tID_dset, "local_DAQ_running": data_flag_dset,
                "z": z_dset, "z_std": z_std_dset, "gmd": gmd_dset,
                "mpe": mpe_dset, "hor_pos": hor_pos_dset, "ver_pos": ver_pos_dset,
                "liq_tofs_e": liq_tofs_e_dset, "vls": vls_dset,
                "between_tdc_files": between_tdc_files_dset,
            }

        # Per-stream count of rows dropped because they were duplicates
        # of train IDs we already processed (raw H5 file boundary overlap).
        n_skipped_overlap = {"gmd": 0, "mpe": 0, "hor_pos": 0,
                             "ver_pos": 0, "vls": 0}

        # Per-train TOF bunch edges (np.arange(0..train_length) * fp);
        # precomputed once so the bunch-folder doesn't re-allocate.
        bunch_edges = np.arange(train_length + 1, dtype=np.float64) * folding_parameter

        # --- Main alignment loop --------------------------------------
        for tID in range(first_tID, last_tID + 1):

            # ---- gmd (per-bunch) -----------------------------------
            # Skip duplicate rows left over from raw H5 file overlap.
            next_tID_gmd, next_gmd, gmd_exhausted = _skip_stale_raw_h5(
                gmd_it, next_tID_gmd, next_gmd, tID, last_tID,
                n_skipped_overlap, "gmd",
            )
            if gmd_exhausted:
                print(f"Stopped by gmd on tID {tID} (during overlap skip).")
                break

            if tID < next_tID_gmd:
                gmd = np.full(train_length, np.nan, dtype=np.float32)
            elif tID == next_tID_gmd:
                # next_gmd is shape (8, n_bunches_raw); index 0 = per-pulse
                # intensity. Pad to the full train_length so VLS-padded
                # schemas (cfg 2: train_length=110, gmd typically 101)
                # don't blow up at chunk-buffer assignment.
                gmd_src = np.asarray(next_gmd[0][:_data_n], dtype=np.float32)
                gmd = np.full(train_length, np.nan, dtype=np.float32)
                gmd[:gmd_src.shape[0]] = gmd_src
                try:
                    next_tID_gmd, next_gmd = gmd_it.__next__()
                except StopIteration:
                    print(f"Stopped by gmd on tID {next_tID_gmd}")
                    break
            # tID > next_tID_gmd cannot happen — the skip above ensures it.

            # ---- mean photon energy --------------------------------
            next_tID_mpe, next_mpe, _ = _skip_stale_raw_h5(
                mpe_it, next_tID_mpe, next_mpe, tID, last_tID,
                n_skipped_overlap, "mpe",
            )
            if tID < next_tID_mpe:
                mpe = np.nan
            elif tID == next_tID_mpe:
                mpe = next_mpe
                try:
                    next_tID_mpe, next_mpe = mpe_it.__next__()
                except StopIteration:
                    print(f"MPE stopped on {next_tID_mpe}. Subsequent are NaN.")
                    next_tID_mpe = last_tID + 1
                    mpe = np.nan

            # ---- horizontal beam position --------------------------
            next_tID_hor_pos, next_hor_pos, _ = _skip_stale_raw_h5(
                hor_pos_it, next_tID_hor_pos, next_hor_pos, tID, last_tID,
                n_skipped_overlap, "hor_pos",
            )
            if tID < next_tID_hor_pos:
                hor_pos = np.nan
            elif tID == next_tID_hor_pos:
                hor_pos = next_hor_pos
                try:
                    next_tID_hor_pos, next_hor_pos = hor_pos_it.__next__()
                except StopIteration:
                    print(f"hor_pos stopped on {next_tID_hor_pos}. Subsequent are NaN.")
                    next_tID_hor_pos = last_tID + 1
                    hor_pos = np.nan

            # ---- vertical beam position ----------------------------
            next_tID_ver_pos, next_ver_pos, _ = _skip_stale_raw_h5(
                ver_pos_it, next_tID_ver_pos, next_ver_pos, tID, last_tID,
                n_skipped_overlap, "ver_pos",
            )
            if tID < next_tID_ver_pos:
                ver_pos = np.nan
            elif tID == next_tID_ver_pos:
                ver_pos = next_ver_pos
                try:
                    next_tID_ver_pos, next_ver_pos = ver_pos_it.__next__()
                except StopIteration:
                    print(f"ver_pos stopped on {next_tID_ver_pos}. Subsequent are NaN.")
                    next_tID_ver_pos = last_tID + 1
                    ver_pos = np.nan

            # ---- vls (config 2 only) -------------------------------
            vls = None
            if config_no == 2:
                next_tID_vls, next_vls, _ = _skip_stale_raw_h5(
                    vls_it, next_tID_vls, next_vls, tID, last_tID,
                    n_skipped_overlap, "vls",
                )
                if tID < next_tID_vls:
                    vls = np.full((train_length, n_vls_pixels), np.nan, dtype=np.float32)
                elif tID == next_tID_vls:
                    raw_vls = np.asarray(next_vls, dtype=np.float32)
                    vls_buf = np.full((train_length, n_vls_pixels), np.nan, dtype=np.float32)
                    m = min(raw_vls.shape[0], train_length)
                    p = min(raw_vls.shape[1] if raw_vls.ndim > 1 else n_vls_pixels, n_vls_pixels)
                    vls_buf[:m, :p] = raw_vls[:m, :p]
                    vls = vls_buf
                    try:
                        next_tID_vls, next_vls = vls_it.__next__()
                    except StopIteration:
                        print(f"VLS stopped on {next_tID_vls}. Subsequent are NaN.")
                        next_tID_vls = last_tID + 1

            # ---- SDU (delay stage z) -------------------------------
            # z / z_std are scalar per train; expand to a (train_length,)
            # array and NaN-fill the bunch slots beyond _data_n so VLS-
            # padded schemas don't carry stale broadcasts in those slots.
            if tID < next_tID_z:
                z = np.full(train_length, np.nan, dtype=np.float32)
                z_std = np.full(train_length, np.nan, dtype=np.float32)
                is_data = False
            elif tID == next_tID_z:
                z = np.full(train_length, np.nan, dtype=np.float32)
                z_std = np.full(train_length, np.nan, dtype=np.float32)
                z[:_data_n] = next_z
                z_std[:_data_n] = next_z_std
                is_data = True
                try:
                    next_tID_z, next_z, next_z_std = sdu_it.__next__()
                except StopIteration:
                    print(f"Stopped by SDU on tID {next_tID_z}")
                    break
            else:
                raise ValueError(f"SDU overshot: tID={tID}, next={next_tID_z}")

            # ---- TDC (per-bunch tof lists) -------------------------
            # When the run has no TDC .lst files, every train gets empty
            # TOFs and between_tdc_files=False; the chunk buffers carry
            # zero-padded placeholders that are never flushed to disk
            # (the TOF datasets weren't created).
            if config_no == 1:
                if not has_tdc:
                    tofs_e = None
                    tofs_i = None
                    between_tdc_files = False
                elif tID < next_tID_tdc:
                    tofs_e = None
                    tofs_i = None
                    between_tdc_files = tdc_it.is_between_files()
                elif tID == next_tID_tdc:
                    tofs_e_raw = next_tofs_e
                    tofs_i_raw = next_tofs_i
                    # Sorted within a sweep -> bunch boundaries via two
                    # vectorised searchsorted calls instead of N boolean
                    # masks across the full tofs array.
                    tofs_e = _fold_into_bunches(
                        tofs_e_raw, train_length, folding_parameter, bunch_edges,
                    )
                    tofs_i = _fold_into_bunches(
                        tofs_i_raw, train_length, folding_parameter, bunch_edges,
                    )
                    between_tdc_files = False
                    try:
                        (next_tID_tdc,
                         next_eventcounts_e, next_tofs_e,
                         next_eventcounts_i, next_tofs_i) = tdc_it.__next__()
                    except StopIteration:
                        print(f"Stopped by TDC on tID {next_tID_tdc}")
                        break
                else:
                    raise ValueError(f"TDC overshot: tID={tID}, next={next_tID_tdc}")

                chunk_full = chunk.add_row(
                    is_data, tID, gmd, mpe, hor_pos, ver_pos, z, z_std,
                    tofs_e, tofs_i, between_tdc_files,
                )
                if chunk_full:
                    chunk.dump(
                        tID_dset, data_flag_dset, z_dset, z_std_dset,
                        gmd_dset, mpe_dset, hor_pos_dset, ver_pos_dset,
                        tofs_e_dset, tofs_i_dset, between_tdc_files_dset,
                    )
                    chunk.reset()

            else:  # config 2
                if not has_tdc:
                    liq_tofs_e = None
                    between_tdc_files = False
                elif tID < next_tID_tdc:
                    liq_tofs_e = None
                    between_tdc_files = tdc_it.is_between_files()
                elif tID == next_tID_tdc:
                    liq_raw = next_tofs_le
                    liq_tofs_e = _fold_into_bunches(
                        liq_raw, train_length, folding_parameter, bunch_edges,
                    )
                    between_tdc_files = False
                    try:
                        (next_tID_tdc,
                         next_eventcounts_le, next_tofs_le) = tdc_it.__next__()
                    except StopIteration:
                        print(f"Stopped by TDC on tID {next_tID_tdc}")
                        break
                else:
                    raise ValueError(f"TDC overshot: tID={tID}, next={next_tID_tdc}")

                chunk_full = chunk.add_row(
                    is_data, tID, gmd, mpe, hor_pos, ver_pos, z, z_std,
                    liq_tofs_e, vls, between_tdc_files,
                )
                if chunk_full:
                    chunk.dump(dsets_cfg2)
                    chunk.reset()

            if tID % 1000 == 0:
                print(f"  tID {tID}  ({tID - first_tID}/{last_tID - first_tID})")

        if config_no == 1:
            chunk.finish(
                tID_dset, data_flag_dset, z_dset, z_std_dset,
                gmd_dset, mpe_dset, hor_pos_dset, ver_pos_dset,
                tofs_e_dset, tofs_i_dset, between_tdc_files_dset,
            )
        else:
            chunk.finish(dsets_cfg2)

    # --- Completion summary ----------------------------------------
    print(f"\n=== {output_path.name} written successfully ===")
    n_skipped_total = sum(n_skipped_overlap.values())
    if n_skipped_total:
        print("raw-H5 overlap rows skipped (duplicate train IDs across file boundaries):")
        for k, v in n_skipped_overlap.items():
            print(f"  {k:<8s} {v}")
    with h5.File(output_path, "r") as f_out:
        n_trains  = f_out["tID"].shape[0]
        n_bunches = f_out["gmd"].shape[1]
        print(f"Trains written  : {n_trains}")
        print(f"Bunches/train   : {n_bunches}")
        print("Datasets:")
        for k in sorted(f_out.keys()):
            d = f_out[k]
            print(f"  /{k:<22} shape={d.shape!s:<24} dtype={d.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write a combined H5 file from one or more beamtime measurements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python write_h5.py delay_scan3 1 48346\n"
            "  python write_h5.py glycine_scan1 2 54609 -o glycine_scan1.h5\n"
            "  python write_h5.py 'glycine_WL_scan*' 1 48346 -o glycine_WL.h5\n"
            "    (quote the pattern to stop the shell expanding it; matches\n"
            "     glycine_WL_scan1, glycine_WL_scan2, ... in both SDU and TDC dirs)\n"
        ),
    )
    parser.add_argument("measurement_name", type=str,
                        help="Basename used to locate SDU .txt and TDC .lst files. "
                             "May be a shell-style glob pattern (e.g. 'glycine_WL_scan*') "
                             "to concatenate multiple scans into one combined H5. The "
                             "files are sorted by trainID across scans.")
    parser.add_argument("config", type=int, choices=(1, 2),
                        help="Experiment configuration: 1 = e+i TOF, 2 = liq eTOF + VLS.")
    parser.add_argument("run_no", type=int,
                        help="FLASH run number used to glob raw H5 files.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output H5 path. Defaults to COMBINED_DIR/<measurement>.h5.")
    parser.add_argument("--train-length", type=int, default=101,
                        help="Bunches per train (default: 400 for cfg 1, 110 for cfg 2).")
    parser.add_argument("--data-bunches", type=int, default=101,
                        help="Bunches per train carrying real GMD/SDU data. "
                             "Slots beyond this are NaN. Default: 101 for cfg 2 "
                             "(VLS=110, FEL=101), full train_length for cfg 1.")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Rows per chunked write (default: 1000 for cfg 1, 200 for cfg 2).")
    parser.add_argument("--max-ecounts", type=int, default=50,
                        help="Zero-padding limit for electron TOF arrays.")
    parser.add_argument("--max-icounts", type=int, default=120,
                        help="Zero-padding limit for ion TOF arrays (cfg 1 only).")
    parser.add_argument("--n-vls-pixels", type=int, default=1280,
                        help="Width of the Gotthard VLS pixel axis (cfg 2 only).")
    parser.add_argument("--folding-parameter", type=float, default=39876.9,
                        help="TOF range (100 ps) used to split sweep events into per-bunch lists.")
    args = parser.parse_args()
    out_path = Path(args.output) if args.output else None
    main(
        config_no=args.config,
        measurement_name=args.measurement_name,
        run_no=args.run_no,
        output_path=out_path,
        train_length=args.train_length,
        chunk_size=args.chunk_size,
        max_ecounts=args.max_ecounts,
        max_icounts=args.max_icounts,
        n_vls_pixels=args.n_vls_pixels,
        folding_parameter=args.folding_parameter,
        data_bunches=args.data_bunches,
    )
