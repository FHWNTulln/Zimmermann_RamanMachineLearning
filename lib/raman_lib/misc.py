import numpy as np
import pandas as pd
import os
import logging
import tqdm
from math import floor, log10
from pathlib import Path

from .opus_converter import convert_opus


def mode(x):
    values, counts = np.unique(x, return_counts=True)
    i = counts.argmax()
    return values[i]


def load_data(path):
    if not isinstance(path, Path):
        path = Path(path)
    suffix = path.suffix

    if suffix == ".csv" or suffix == ".txt" or suffix == ".tsv":
        data = pd.read_csv(path)

    else:
        data = []
        labels = []

        for dir in os.listdir(path):
            print(dir)
            for file in os.listdir(os.path.join(path, dir)):
                filepath = os.path.join(path, dir, file)

                if filepath.endswith(".csv"):
                    data.append(np.loadtxt(filepath, sep=","))
                elif filepath.endswith(".tsv"):
                    data.append(np.loadtxt(filepath, sep="\t"))
                elif filepath.endswith(".txt"):
                    data.append(np.loadtxt(filepath))
                else:
                    try:
                        data.append(convert_opus(filepath))
                    except:
                        raise ValueError(
                            f"File {file} does not match any inplemented file format."
                             "Use either plaintext (.csv, .tsv, .txt) or"
                             "binary OPUS (.0, .1, ...) files")

                labels.append(dir)
    
        data = np.asarray(data)

        data = pd.DataFrame(data[:,:,1], columns=data[0,:,0])
        data.insert(0, "label", labels)

    return data


def int_float(s):
    try:
        n = int(s)
    except ValueError:
        n = float(s)
    finally:
        return n


class TqdmStreamHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)
