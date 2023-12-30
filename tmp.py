import os
file_PSG = []
file_Hyp = []
for dirname, _, filenames in os.walk('../database/sleep-cassette'):
    for filename in filenames:
        if filename[9] == 'P':
            file_PSG.append(os.path.join(dirname, filename))
        else:
            file_Hyp.append(os.path.join(dirname, filename))

N = len(file_PSG)