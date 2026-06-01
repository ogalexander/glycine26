from typing import Tuple
import os
import sys
# sys.path.append('/asap3/flash/gpfs/fl24/2023/data/11015651/processed/analysis_tools/decoding_script/beamtime_scripts_2021')
#sys.path.append('/asap3/flash/gpfs/fl24/2023/data/11015651/processed/analysis_tools/decoding_script')
sys.path.append('/asap3/flash/gpfs/fl24/2026/data/11022188/processed/analysis_tools/decoding_script')
import re
import glob

from tqdm import tqdm
import numpy as np
import pandas as pd

from beamtime_scripts_2021.util import write_pos_data_to_file
# from util import write_pos_data_to_file

from pathlib import Path

from beamtime_scripts_2021.MCS6A_decoding import decoding, metadata_decoding
from beamtime_scripts_2021.util import write_tof_data_to_file

import h5py as h5 
class DataChunk():
    def __init__(self, chunk_size, train_length, max_ecounts=50, max_icounts=120):
        self.chunk_size = chunk_size
        self.train_length = train_length
        self.max_ecounts = max_ecounts
        self.max_icounts = max_icounts
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
                self.tofs_es[self.idx, b_idx, :len(tofs_e_bunch)] = tofs_e_bunch
        if tofs_i is not None:
            for b_idx, tofs_i_bunch in enumerate(tofs_i):
                self.tofs_is[self.idx, b_idx, :len(tofs_i_bunch)] = tofs_i_bunch

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
        tofs_es_dset[roi]             = self.tofs_es
        tofs_is_dset[roi]             = self.tofs_is
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
        self.tofs_es = np.full((self.chunk_size, self.train_length, self.max_ecounts), np.nan, dtype=np.uint32)
        self.tofs_is = np.full((self.chunk_size, self.train_length, self.max_icounts), np.nan, dtype=np.uint32)
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
        tofs_es_dset[roi]           = self.tofs_es
        tofs_is_dset[roi]           = self.tofs_is
        data_flag_dset[roi]         = self.is_datas
        between_tdc_files_dset[roi] = self.between_tdc_filess

        (dset.resize((self.count*self.chunk_size) + self.idx-1, axis=0) for dset in 
            [tID_dset, z_dset, z_std_dset, gmd_dset, mpe_dset, hor_pos_dset, ver_pos_dset, between_tdc_files_dset]) # 4 missing from here

class TDCIterator:
    def __init__(self, measurement_fpaths):
        self._measurement_fpaths = measurement_fpaths

        # Initialize Decoder object and specify the data that should be decoded
        self.decoder = decoding.Decoder(wanted_data=['channel', 'timedata', 'sweep', 'tagbits'])
        
        # Get number of sweeps/trainIDs per file specified by the sweep preset
        self.sweeps_per_file = metadata_decoding.from_file(self._measurement_fpaths[0], keywords='swpreset=', paragraph_kw='MPA4A')[0]

        self.trainIDs_tdc, self.eventcounts_e, self.tofs_e, self.eventcounts_i, self.tofs_i = extract_data_from_single_file(self._measurement_fpaths[0], self.decoder, self.sweeps_per_file)
        print('!!!!!!!!!')
        print(self.trainIDs_tdc.shape, self.eventcounts_e.shape, self.tofs_e.shape, self.eventcounts_i.shape, self.tofs_i.shape)

        self._index = 0 
        self._file_index = 1 

    def __iter__(self):
        return self

    # def __next__(self):
    #     if self._index < len(self.trainIDs_tdc):
    #         tofs_e_bunch, self.tofs_e = np.split(self.tofs_e, [self.eventcounts_e[self._index]])
    #         tofs_i_bunch, self.tofs_i = np.split(self.tofs_i, [self.eventcounts_i[self._index]])
    #         self._index += 1
    #         return self.trainIDs_tdc[self._index-1], self.eventcounts_e[self._index-1], tofs_e_bunch, self.eventcounts_i[self._index-1], tofs_i_bunch
    #         # return self.trainIDs_tdc[self._index-1], self.eventcounts_e[self._index-1], self.tofs_e[self._index-1], self.eventcounts_i[self._index-1], self.tofs_i[self._index-1]

    #     elif self._file_index < len(self._measurement_fpaths):
    #         print('Finished TDC file {} of {}.'.format(self._file_index, len(self._measurement_fpaths)))
    #         self.trainIDs_tdc, self.eventcounts_e, self.tofs_e, self.eventcounts_i, self.tofs_i = extract_data_from_single_file(self._measurement_fpaths[self._file_index], self.decoder, self.sweeps_per_file)
    #         # print(self.trainIDs_tdc.shape, self.eventcounts_e.shape, self.tofs_e.shape, self.eventcounts_i.shape, self.tofs_i.shape)
    #         self._file_index += 1
    #         self._index = 1
    #         return self.trainIDs_tdc[0], self.eventcounts_e[0], self.tofs_e[0], self.eventcounts_i[0], self.tofs_i[0]
            
    #     else:
    #         print('Finished TDC file {} of {}.'.format(self._file_index, len(self._measurement_fpaths)))
    #         raise StopIteration

    def __next__(self):
        self.between_files = False
        if self._index == len(self.trainIDs_tdc):

            if self._file_index < len(self._measurement_fpaths):
                print('Finished TDC file {} of {}.'.format(self._file_index, len(self._measurement_fpaths)))
                self.trainIDs_tdc, self.eventcounts_e, self.tofs_e, self.eventcounts_i, self.tofs_i = extract_data_from_single_file(self._measurement_fpaths[self._file_index], self.decoder, self.sweeps_per_file)
                self._file_index += 1
                self._index = 0
                self.between_files = True

            else:
                print('Finished TDC file {} of {}.'.format(self._file_index, len(self._measurement_fpaths)))
                raise StopIteration

        tofs_e_bunch, self.tofs_e = np.split(self.tofs_e, [self.eventcounts_e[self._index]])
        tofs_i_bunch, self.tofs_i = np.split(self.tofs_i, [self.eventcounts_i[self._index]])

        self._index += 1
        return self.trainIDs_tdc[self._index-1], self.eventcounts_e[self._index-1], tofs_e_bunch, self.eventcounts_i[self._index-1], tofs_i_bunch
        # return self.trainIDs_tdc[self._index-1], self.eventcounts_e[self._index-1], self.tofs_e[self._index-1], self.eventcounts_i[self._index-1], self.tofs_i[self._index-1]

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

        # Decode and process the binary data from one .lst-file
        trainIDs, eventcounts_e, tofs_e, eventcounts_i, tofs_i = extract_data_from_single_file(fpath, decoder, sweeps_per_file)
        
        # Save the decoded and processed data
        write_tof_data_to_file(f_electron, trainIDs, eventcounts_e, tofs_e)
        write_tof_data_to_file(f_ion, trainIDs, eventcounts_i, tofs_i)

    # Close the decoded data files
    f_electron.close()
    f_ion.close()


def extract_data_from_single_file(fpath: str, decoder: decoding.Decoder, sweeps_per_file: int) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decodes a single TDC-generated .lst-file, preprocessed the data and returns it as Numpy arrays.

    First, the binary .lst-file is decoded by the user-written MCS6A_decoding.Decoder object to yield all recorded
    events of that acquisition. Any inconsistencies in the 16bit trainID recorded via the tagbits of the TDC are then
    corrected. After this, the 32bit trainID is reconstructed from the 16bit trainID. The data is separated into events
    from the electron and the ion channel and the number of events on each channel corresponding to each trainID is
    calculated. Finally, five Numpy arrays with the trainIDs and eventcounts and tofs for both channels are returned.

    Note: 17.12.2020: Added LARGEST_ALLOWED_TOF parameter to stop the tof from overflowing 3 bytes in the custom-
                      format .dat-file.

    :param fpath: absolute filepath of one TDC-generated .lst-file
    :param decoder: user-written decoder object that is used for decoding the raw TDC data
    :param sweeps_per_file: number of sweeps and thus trainIDs the file should contain
    :return: tuple of 5 Numpy arrays:
            1. trainIDs: all trainIDs contained in the file (length: sweeps_per_file)
            2. eventcounts_e: number of events on the electron channel (corresponding to the trainID at the same index)
            3. tofs_e: all tof events on the electron channel (for all trainIDs, length: sum(eventcounts_e))
            4. eventcounts_i: number of events on the ion channel (corresponding to the trainID at the same index)
            5. tofs_i: all tof events on the ion channel (for all trainIDs, length: sum(eventcounts_i))
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

    # Create separate DataFrame for electron and ion events
    electron_events = df[df['channel'] == 1]
    ion_events = df[df['channel'] == 2]
    
    # Calculate array of all trainIDs that should be present in the .lst-file
    if len(df) > 0:
        trainIDs=np.arange(sweeps_per_file)+df.iloc[0,3]-df.iloc[0,2]+1
    else:
        # Added this after files with no real data made the script throw an exception
        # Problem: Because not a single event is listed in the file, one can not know the exact offset between the 
        # sweep and the trainID; Consequently one has to estimate the offset with the help of the file name
        # From reconstruct_32bit_trainID(): "Usually the first trainID recorded by the TDC is around 4-8 integers 
        # higher than the trainID in the respective filename."
        # trainID_offset = trainID_at_start_32bit + 5
        trainIDs = np.arange(sweeps_per_file) + trainID_at_start_32bit + 6
    
    # Get the number of tof electron events corresponding to each trainID (same for ions)
    eventcounts_e = calc_eventcount_per_trainID(electron_events, trainIDs)
    eventcounts_i = calc_eventcount_per_trainID(ion_events, trainIDs)

    # Get all tofs as Numpy arrays sorted by their sweep and thus trainID and timedata
    #tofs_e = electron_events[['timedata', 'sweep']].sort_values(by=['sweep', 'timedata'])['timedata'].to_numpy()
    #tofs_i = ion_events[['timedata', 'sweep']].sort_values(by=['sweep', 'timedata'])['timedata'].to_numpy()
    #NEW: sorted already above
    tofs_e = electron_events['timedata'].to_numpy()
    tofs_i = ion_events['timedata'].to_numpy()

    return trainIDs, eventcounts_e, tofs_e, eventcounts_i, tofs_i


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
    # Get the number of events per trainID
    eventcount_per_train = df['trainID'].value_counts(sort=False)#.sort_index() # sort unnecessary because already sorted
    
    trainIDs = eventcount_per_train.index.to_numpy()
    
    # Get the eventcounts as a simple Numpy vector
    eventcounts = eventcount_per_train.to_numpy()

    # Create array of zeros with the same length as all_trainIDs
    all_eventcounts = np.zeros(all_trainIDs.shape[0], dtype=np.uint16)

    # Insert the eventcounts (> 0) into said array; Every trainID now has a corresponding eventcount, even if it is 0
    trainIDs_with_events = np.isin(all_trainIDs, trainIDs)
    all_eventcounts[trainIDs_with_events] = eventcounts

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
        d
        f['trainID'] = df.index + trainID_offset
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


if __name__ == '__main__':

    base_path = '/asap3/flash/gpfs/fl24/2023/data/11015651'
    # data_folder = os.path.join(base_path, 'processed', 'local_DAQ', 'delay_scan3')
    data_folder = os.path.join(base_path, 'processed', 'local_DAQ', 'short_scans')

    # measurement_name = r'delay_scan3'
    # run_no = 48346

    # measurement_name = r'short_scan_275.5eV'
    # run_no = 48413
    # fname_out_h5 = 'short_scan_275pt5eV.h5'


    # measurement_name = r'short_scan_277.5eV'
    # run_no = 48415
    # fname_out_h5 = 'short_scan_277pt5eV.h5'

    # measurement_name = r'short_scan_279.5eV'
    # run_no = 48417
    # fname_out_h5 = 'short_scan_279pt5eV.h5'

    # measurement_name = r'short_scan_281.5eV'
    # run_no = 48419
    # fname_out_h5 = 'short_scan_281pt5eV.h5'

    # measurement_name = r'short_scan_270.5eV'
    # run_no = 48424
    # fname_out_h5 = 'short_scan_270pt5eV.h5'

    measurement_name = r'short_scan_268.0eV'
    run_no = 48425
    fname_out_h5 = 'short_scan_268eV.h5'


    decoded_data_folder = os.path.join('home/ogalex', 'glycine', 'hdf5', 'short_scans')
    # decoded_data_folder = os.path.join('home/ogalex', 'glycine', 'hdf5', 'delay_scan3')
    folding_parameter = 9969.225
    train_lenth = 400
    max_electrons_bunch = 50
    max_ions_bunch = 120

    # Look for data files
    files_in_folder = os.listdir(data_folder)

    ## first the stu files
    measurement_names = list(sorted(filter(re.compile(measurement_name + r'_\d{10}.txt').match, files_in_folder)))
    measurement_fpaths = [data_folder + '/' + measurement_name for measurement_name in measurement_names]
    
    # print(measurement_fpaths)

    if len(measurement_fpaths) == 0:
        raise ValueError(f'No files with measurement_name "{measurement_name}" found in folder "{data_folder}".')
    
    print('{} sdu data files found.'.format(len(measurement_fpaths), run_no))

    ## now the electron/ion files
    ### Get absolute fpaths of all .lst-files of the measurement, sorted by their name and thus by their trainID
    measurement_names_tdc = list(sorted(filter(re.compile(measurement_name + r'_\d{10}.lst').match, files_in_folder)))
    measurement_fpaths_tdc = [data_folder + '/' + measurement_name for measurement_name in measurement_names_tdc]
    
    if len(measurement_fpaths_tdc) == 0:
        raise ValueError(f'No files with measurement_name "{measurement_name}" found in folder "{data_folder}".')

    ## lastly, the online h5 files
    h5_folder = os.path.join(base_path, 'raw', 'hdf', 'online-0', 'fl2user1')
    print(h5_folder + '/*run'+str(run_no)+'*.h5')
    h5_paths = sorted(glob.glob(h5_folder + '/*run'+str(run_no)+'*.h5'))
    
    print('{} h5 files found for run {}.'.format(len(h5_paths), run_no))
    print(h5_paths)

    # find the first and last trainID in the measurement
    first_tID = extract_sdu_data_from_single_file(measurement_fpaths[0])[0][0]
    last_tID  = extract_sdu_data_from_single_file(measurement_fpaths[-1])[0][-1]

    print('Dataset includes bunch IDs from {} until {}.'.format(first_tID, last_tID))

    for path_idx, h5_path in enumerate(h5_paths):

        f =  h5.File(h5_path,'r')
        tIDs = f['/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall/index']
        print(max(tIDs))
        if max(tIDs)>first_tID:
            first_h5_idx = path_idx
            break

    
    chunk_size = 1000
    train_length = 400
    chunk_idx = 0

    chunk = DataChunk(chunk_size, train_length, max_ecounts=max_electrons_bunch, max_icounts=max_ions_bunch)

    sdu_it = SDUIterator(measurement_fpaths)
    next_tID_z, next_z, next_z_std = sdu_it.__next__()

    tdc_it = TDCIterator(measurement_fpaths_tdc)

    gmd_it = h5Iterator(h5_paths[first_h5_idx:], [
        '/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall/index', 
        '/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall/value'])
    mpe_it = h5Iterator(h5_paths[first_h5_idx:], [
        '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean photon energy/index', 
        '/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean photon energy/value'])
    hor_pos_it = h5Iterator(h5_paths[first_h5_idx:], [
        '/FL2/Photon Diagnostic/GMD/Average beam position/position hall horizontal/index', 
        '/FL2/Photon Diagnostic/GMD/Average beam position/position hall horizontal/value'])
    ver_pos_it = h5Iterator(h5_paths[first_h5_idx:], [
        '/FL2/Photon Diagnostic/GMD/Average beam position/position hall vertical/index', 
        '/FL2/Photon Diagnostic/GMD/Average beam position/position hall vertical/value'])


    for tID_gmd, gmd in gmd_it:
        if tID_gmd < first_tID:
            if tID_gmd%500 ==0:
                print (tID_gmd, first_tID)
        elif tID_gmd >= first_tID:
            print('Finished on {}!'.format(tID_gmd))
            next_tID_gmd, next_gmd = tID_gmd, gmd
            break
        # else:
        #     raise ValueError('Failed on {} > {}'.format(tID_gmd, first_tID))


    for tID_mpe, mpe in mpe_it:
        if tID_mpe < first_tID:
            if tID_mpe%500 ==0:
                print (tID_mpe, first_tID)
        elif tID_mpe >= first_tID:
            print('Finished on {}!'.format(tID_mpe))
            next_tID_mpe, next_mpe = tID_mpe, mpe
            break

    for tID_hor_pos, hor_pos in hor_pos_it:
        if tID_hor_pos < first_tID:
            if tID_hor_pos%500 ==0:
                print (tID_hor_pos, first_tID)
        elif tID_hor_pos >= first_tID:
            print('Finished on {}!'.format(tID_hor_pos))
            next_tID_hor_pos, next_hor_pos = tID_hor_pos, hor_pos
            break

    for tID_ver_pos, ver_pos in ver_pos_it:
        if tID_ver_pos < first_tID:
            if tID_ver_pos%500 ==0:
                print (tID_ver_pos, first_tID)
        elif tID_ver_pos >= first_tID:
            print('Finished on {}!'.format(tID_ver_pos))
            next_tID_ver_pos, next_ver_pos = tID_ver_pos, ver_pos
            break
    # get the starting values for each of the iterators
    # next_tID_gmd, next_gmd = gmd_it.__next__()
    # next_tID_mpe, next_mpe = mpe_it.__next__()
    # next_tID_hor_pos, next_hor_pos = hor_pos_it.__next__()
    # next_tID_ver_pos, next_ver_pos = ver_pos_it.__next__()
    next_tID_z, next_z, next_z_std = sdu_it.__next__()
    next_tID_tdc, next_eventcounts_e, next_tofs_e, next_eventcounts_i, next_tofs_i = tdc_it.__next__()
    
    data_len = last_tID - first_tID

    dt = h5.vlen_dtype(np.dtype('int32'))

    # with h5.File('/asap3/flash/gpfs/fl24/2023/data/11015651/processed/oga/short_scan_275pt5eV.h5', 'a') as f_out:
    with h5.File('/asap3/flash/gpfs/fl24/2023/data/11015651/processed/oga/{}'.format(fname_out_h5), 'a') as f_out:
    # with h5.File('/asap3/flash/gpfs/fl24/2023/data/11015651/processed/oga/delay_scan3_120824.h5', 'a') as f_out:

        tID_dset               = f_out.create_dataset('tID', (data_len, ), maxshape=(data_len, ), dtype='double')
        data_flag_dset         = f_out.create_dataset('local_DAQ_running', (data_len, ), maxshape=(data_len, ), dtype='bool')
        z_dset                 = f_out.create_dataset('z', (data_len, 400), maxshape=(data_len, 400), dtype = np.float32)
        z_std_dset             = f_out.create_dataset('z_std', (data_len, 400), maxshape=(data_len, 400), dtype = np.float32)
        gmd_dset               = f_out.create_dataset('gmd', (data_len, 400), maxshape=(data_len, 400), dtype = np.float32)
        mpe_dset               = f_out.create_dataset('mpe', (data_len, ), maxshape=(data_len,), dtype = np.float32)
        hor_pos_dset           = f_out.create_dataset('hor_pos', (data_len, ), maxshape=(data_len,), dtype = np.float32)
        ver_pos_dset           = f_out.create_dataset('ver_pos', (data_len, ), maxshape=(data_len,), dtype = np.float32)
        # tofs_e_dset            = f_out.create_dataset('tofs_e', (data_len, 400), maxshape=(data_len, 400), dtype = dt)
        # tofs_i_dset            = f_out.create_dataset('tofs_i', (data_len, 400), maxshape=(data_len, 400), dtype = dt)
        tofs_e_dset            = f_out.create_dataset('tofs_e', (data_len, 400, max_electrons_bunch), maxshape=(data_len, 400, max_electrons_bunch), dtype = np.uint32, compression='gzip')
        tofs_i_dset            = f_out.create_dataset('tofs_i', (data_len, 400, max_ions_bunch), maxshape=(data_len, 400, max_ions_bunch), dtype = np.uint32, compression='gzip')
        between_tdc_files_dset = f_out.create_dataset('between_tdc_files', (data_len, ), maxshape=(data_len, ), dtype='bool')

        for tID in range(first_tID, last_tID+1):
            # print(tID)

            # iterate through the gmd
            if tID < next_tID_gmd:
                gmd = np.nan
            elif tID == next_tID_gmd:
                gmd = next_gmd[0]
                try:
                    next_tID_gmd, next_gmd = gmd_it.__next__()
                except StopIteration:
                    print('Stopped by gmd on tID {}'.format(next_tID_gmd))
                    break
            else:
                raise ValueError('Faled because {} > {}'.format(tID, next_tID_gmd))

            # iterate through the mean photon energy
            if tID < next_tID_mpe:
                mpe = np.nan
            elif tID == next_tID_mpe:
                mpe = next_mpe
                try:
                    next_tID_mpe, next_mpe = mpe_it.__next__()
                except StopIteration:
                    print('MPE stopped on {}. All subsequent are nans'.format(next_tID_mpe))
                    next_tID_mpe = last_tID+1
                    mpe = np.nan
                    # break
            else:
                raise ValueError('Failed because {} > {}'.format(tID, next_tID_mpe))

            # iterate through the horizontal position
            if tID < next_tID_hor_pos:
                hor_pos = np.nan
            elif tID == next_tID_hor_pos:
                hor_pos = next_hor_pos
                try:
                    next_tID_hor_pos, next_hor_pos = hor_pos_it.__next__()
                except StopIteration:
                    print('hor_pos stopped on {}. All subsequent are nans'.format(next_tID_hor_os))
                    next_tID_hor_pos = last_tID+1
                    hor_pos = np.nan
                    # break
            else:
                raise ValueError('Failed because {} > {}'.format(tID, next_tID_hor_pos))

            # iterate through the vertical position
            if tID < next_tID_ver_pos:
                ver_pos = np.nan
            elif tID == next_tID_ver_pos:
                ver_pos = next_ver_pos
                try:
                    next_tID_ver_pos, next_ver_pos = ver_pos_it.__next__()
                except StopIteration:
                    print('ver_pos stopped on {}. All subsequent are nans'.format(next_tID_ver_os))
                    next_tID_ver_pos = last_tID+1
                    ver_pos = np.nan
                    # break
            else:
                raise ValueError('Faled because {} > {}'.format(tID, next_tID_ver_pos))

            # iterate through the sdu
            if tID < next_tID_z:
                z = np.nan
                z_std = np.nan
                is_data = False
            elif tID == next_tID_z:
                z = next_z
                z_std = next_z_std
                is_data = True
                try:
                    next_tID_z, next_z, next_z_std = sdu_it.__next__()
                except StopIteration:
                    print('Stopped by stu on tID {}'.format(next_tID_z))
                    break
            else:
                raise ValueError

            # iterate through the tdc
            if tID < next_tID_tdc:
                eventcounts_e = 0
                tofs_e = None
                eventcounts_i = 0
                tofs_i = None
                between_tdc_files = tdc_it.is_between_files() # Check whether there are no counts because there were none for that short or whether it is in a gap between files

            elif tID == next_tID_tdc:
                eventcounts_e = next_eventcounts_e
                tofs_e = next_tofs_e
                eventcounts_i = next_eventcounts_i
                tofs_i = next_tofs_i

                tofs_e = [tofs_e[(tofs_e>(b_idx*folding_parameter))*(tofs_e<((b_idx+1)*folding_parameter))] - b_idx*folding_parameter for b_idx in range(train_length)]
                tofs_i = [tofs_i[(tofs_i>(b_idx*folding_parameter))*(tofs_i<((b_idx+1)*folding_parameter))] - b_idx*folding_parameter for b_idx in range(train_length)]

                between_tdc_files = False
                
                try:
                    next_tID_tdc, next_eventcounts_e, next_tofs_e, next_eventcounts_i, next_tofs_i = tdc_it.__next__()
                except StopIteration:
                    print('Stopped by stu on tID {}'.format(next_tID_z))
                    break
            else:
                raise ValueError

            chunk_full = chunk.add_row(is_data, tID, gmd, mpe, hor_pos, ver_pos, z , z_std, tofs_e, tofs_i, between_tdc_files)
            if chunk_full:
                chunk.dump(tID_dset, data_flag_dset, z_dset, z_std_dset, gmd_dset, mpe_dset, hor_pos_dset, ver_pos_dset,
                    tofs_e_dset, tofs_i_dset, between_tdc_files_dset)
                chunk.reset()

            if tID%400 == 0:
                print(tID, first_tID, last_tID)
                # print('****')
                # print(gmd, mpe, hor_pos, ver_pos)

        chunk.finish(tID_dset, data_flag_dset, z_dset, z_std_dset, gmd_dset, mpe_dset, hor_pos_dset, ver_pos_dset, 
            tofs_e_dset, tofs_i_dset, between_tdc_files_dset)
