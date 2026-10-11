"""
Basic metadata validation for audio samples
"""
import logging
from os import listdir

import soundfile as sf

log = logging.getLogger(__name__)

def is_loadable(path) -> bool:
    try:
        info = sf.info(path)
    except (sf.LibsndfileError, RuntimeError):
        return False
    return info.frames > 0 and info.samplerate > 0

def are_samples_valid(path) -> bool:
    status = True
    errors = []

    for file in listdir(path):
        if not is_loadable(path + file):
            errors.append(file)
            status = False

    if not status:
        log.error(f"Invalid samples found in {path}: {errors}")
    else:
        log.info(f"Valid samples found in {path}")
    return status