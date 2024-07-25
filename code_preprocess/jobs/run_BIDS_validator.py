#!/usr/bin/env python
import os
import sys
from bids_validator import BIDSValidator


if len(sys.argv) == 2:
    BIDS_dir = str(sys.argv[1])
else:
    raise Exception("too many arguments are being passed to python")

print(BIDSValidator().is_bids(f'{BIDS_dir}'))