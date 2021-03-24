#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .version import __version__  # noqa
from .machine import Machine  # noqa
from .tpf import TPFMachine  # noqa

import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
