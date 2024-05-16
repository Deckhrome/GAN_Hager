import os
from zipfile import ZipFile, BadZipFile
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import pickle

def ImgFrMat(mat_file, save_directory):
    mat_data = loadmat(mat_file)
    fs = 4000
    name_val = ['current']
    current = np.squeeze(mat_data['current'])
    voltage = np.squeeze(mat_data['voltage'])

    for name in name_val:
        try:
            plt.figure()
            if name == 'current':
                f, t, Zxx = stft(current, fs=fs, nperseg=38*10)
            elif name == 'voltage':
                f, t, Zxx = stft(voltage, fs=fs, nperseg=38*10)
            else:
                f, t, Zxx = stft(np.squeeze(mat_data[name]), fs=fs, nperseg=38*1)

            x_real = np.real(Zxx)
            x_imag = np.imag(Zxx)
            x_angle = np.angle(Zxx) / np.pi

            array1 = np.repeat(x_real, 16, axis=1)
            array2 = np.repeat(x_imag, 16, axis=1)
            array3 = np.repeat(x_angle, 16, axis=1)

            array1 = 1 * ((array1 - np.min(array1)) / (np.max(array1) - np.min(array1)))
            array2 = 1 * ((array2 - np.min(array2)) / (np.max(array2) - np.min(array2)))
            array3 = 1 * ((array3 - np.min(array3)) / (np.max(array3) - np.min(array3)))

            rgb_image = np.dstack((array1, array2, array3))
            plt.imshow(rgb_image)

            plt.axis('off')
            file_base_name = os.path.splitext(os.path.basename(mat_file))[0]
            file_png = os.path.join(save_directory, f'{file_base_name}_{name}.png')

            if os.path.isfile(file_png):
                os.remove(file_png)

            plt.savefig(file_png, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()

            print(f'Processed {name}, saved as {file_png}')
        except Exception as e:
            print(f'Error processing {name}: {e}')


def list_mat_files(directory):
    mat_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".mat"):
            mat_files.append(os.path.join(directory, filename))
    return mat_files

mat_directory = "extracted/SDIA/"
save_directory  = "spectrogram/img_save/"
all_path = list_mat_files(mat_directory)

for mat_path in all_path:
    ImgFrMat(mat_path,save_directory)
