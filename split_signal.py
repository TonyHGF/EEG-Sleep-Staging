import numpy as np
import pyedflib

def split_edf_by_annotations(edf_file, start_times, durations, events):
    # read edf file
    with pyedflib.EdfReader(edf_file) as edf:
        signals = []
        for i in range(edf.signals_in_file):
            signals.append(edf.readSignal(i))

        # results
        split_signals = []
        len_result = len(start_times)
        for i in range(len_result):
            start_time = start_times[i]
            duration = durations[i]
            event = events[i]
            start_sample = int(start_time * edf.getSampleFrequency(0))
            end_sample = start_sample + int(duration * edf.getSampleFrequency(0))

            
            event_signals = []
            # for signal in signals:
            for j in range(3):
                signal = signals[j]
                split_signal = signal[start_sample:end_sample]
                event_signals.append(split_signal)

            split_signals.append((event, event_signals))

        return split_signals

def split_edf_by_30s(split_signals):
    new_split_signals = []
    for (event, event_signals) in split_signals:
        duration = len(event_signals[0])
        for start in range(0, 3000, duration):
            if(start + 3000 >= duration):break
            new_event_signals = []
            for event_signal in event_signals:
                new_event_signals.append(event_signal[start : start + 3000])
            new_split_signals.append((event, new_event_signals))
    return new_split_signals