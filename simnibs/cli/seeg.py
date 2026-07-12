# -*- coding: utf-8 -*-
"""
command line tool for sEEG depth-electrode head models (conforming build + field sampling)
This program is part of the SimNIBS package.
Please check on www.simnibs.org how to cite our work in publications.

Thin console-script wrapper around ``simnibs.simulation.seeg.__main__`` so the
``conforming`` / ``fields`` subcommands are reachable as the first-class ``seeg`` command
(consistent with the other simnibs CLI entry points), not only via
``python -m simnibs.simulation.seeg``. Note the primary, recommended way to add electrodes
is ``charm --seeg <spec.json>``; this command covers the standalone build + post-solve
sampling helpers.

This program is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either version 3
of the License, or any later version.
"""

import sys

from simnibs.simulation.seeg.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
