"""Allows ``python -m bythebooktest``."""

import sys

from .cli import main

sys.exit(main())
