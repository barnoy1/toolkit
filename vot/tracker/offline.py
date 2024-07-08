import sys
import os
import time
import re
import subprocess
import shutil
import shlex
import socket as socketio
import tempfile
import logging
import unittest
from pathlib import Path
from typing import Tuple
from threading import Thread, Lock

import numpy as np

import colorama
import yaml

from trax import TraxException
from trax.client import Client
from trax.image import FileImage
from trax.region import Region as TraxRegion
from trax.region import Polygon as TraxPolygon
from trax.region import Mask as TraxMask
from trax.region import Rectangle as TraxRectangle

import vot.region
from vot.dataset import Frame, DatasetException, Sequence
from vot.region import Region, Polygon, Rectangle, Mask, Special
from vot.tracker import Tracker, TrackerRuntime, TrackerException, Objects, ObjectStatus
from vot.tracker.trax import LogAggregator, normalize_paths, ColorizedOutput, TrackerProcess
from vot.utilities import to_logical, to_number, normalize_path

PORT_POOL_MIN = 9090
PORT_POOL_MAX = 65535

logger = logging.getLogger("vot")


class OfflineState(object):
    def __init__(self, region):
        self.properties = dict()
        self.region = region

class TrackerStreamer(object):
    def __init__(self, tracker, arguments):
        self.trajectory_list = None
        self._tracker = tracker
        config_file = arguments['gt_sequences']
        with open(config_file, 'r') as fp:
            config = yaml.load(fp, Loader=yaml.BaseLoader)

        self.sequence_dict = dict(
            gt=config['gt_sequences'],
            inference=arguments['regression']
        )

    def initialize(self, frame, new, properties):
        sequence_key = frame.sequence.identifier
        inference_file = self.sequence_dict.get('inference').get(sequence_key)[0]
        from vot.region.io import read_trajectory
        self.trajectory_list = read_trajectory(inference_file)
        state = OfflineState(region=self.trajectory_list[0])
        time = frame.index
        return state, time

    def update(self, frame, new, properties):
        time = frame.index
        state = OfflineState(region=self.trajectory_list[frame.index])
        return state, time


class TraxTrackerRuntimeOffline(TrackerRuntime):
    """ The TraX tracker runtime. This class is used to run a tracker using the TraX protocol."""

    def __init__(self, tracker: Tracker, command: str, log: bool = False, timeout: int = 30, linkpaths=None,
                 envvars=None, arguments=None, socket=False, restart=False, onerror=None):
        """ Initializes the TraX tracker runtime.

        Args:
            tracker: The tracker to be run.
            command: The command to run the tracker.
            log: Whether to log the output of the tracker.
            timeout: The timeout in seconds for the tracker to respond.
            linkpaths: The paths to be added to the PATH environment variable.
            envvars: The environment variables to be set for the tracker.
            arguments: The arguments to be passed to the tracker.
            socket: Whether to use a socket to communicate with the tracker.
            restart: Whether to restart the tracker if it crashes.
            onerror: The error handler to be called if the tracker crashes.
        """
        super().__init__(tracker)
        self._command = command

        self._tracker = tracker
        if linkpaths is None:
            linkpaths = []
        if isinstance(linkpaths, str):
            linkpaths = linkpaths.split(os.pathsep)
        linkpaths = normalize_paths(linkpaths, tracker)
        self._socket = to_logical(socket)
        self._restart = to_logical(restart)
        if not log:
            self._output = LogAggregator()
        else:
            self._output = None
        self._timeout = to_number(timeout, min_n=1)
        self._arguments = arguments
        self._onerror = onerror
        self._workdir = None

        if sys.platform.startswith("win"):
            pathvar = "PATH"
        else:
            pathvar = "LD_LIBRARY_PATH"

        envvars[pathvar] = envvars[pathvar] + os.pathsep + os.pathsep.join(
            linkpaths) if pathvar in envvars else os.pathsep.join(linkpaths)
        envvars["TRAX"] = "1"

        self._envvars = envvars
        self._process = TrackerStreamer(tracker, arguments)

    @property
    def tracker(self) -> Tracker:
        """ The associated tracker object. """
        return self._tracker

    @property
    def multiobject(self):
        """ Whether the tracker supports multiple objects."""
        self._connect()
        return self._process._multiobject

    def _connect(self):
        """ Connects to the tracker. This method is used to connect to the tracker. It starts the tracker process if it is not running yet."""
        pass

    def _error(self, exception):
        """ Handles an error. This method is used to handle an error. It calls the error handler if it is set."""
        workdir = None
        timeout = False
        if not self._output is None:
            if not self._process is None:
                if self._process.alive:
                    self._process.terminate()

                self._output("Process exited with code ({})\n".format(self._process.returncode))
                timeout = self._process.interrupted
                self._workdir = self._process.workdir
            else:
                self._output("Process not alive anymore, unable to retrieve return code\n")

        log = str(self._output)

        try:

            if not self._onerror is None and isinstance(self._onerror, callable):
                self._onerror(log, workdir)

        except Exception as e:
            logger.exception("Error during error handler for runtime of tracker %s", self._tracker.identifier,
                             exc_info=e)

        if timeout:
            raise TrackerException("Tracker interrupted, it did not reply in {} seconds".format(self._timeout),
                                   tracker=self._tracker, \
                                   tracker_log=log if not self._output is None else None)

        raise TrackerException(exception, tracker=self._tracker, \
                               tracker_log=log if not self._output is None else None)

    def restart(self):
        """ Restarts the tracker. This method is used to restart the tracker. It stops the tracker process and starts it again."""
        try:
            self.stop()
            self._connect()
        except TraxException as e:
            self._error(e)

    def initialize(self, frame: Frame, new: Objects = None, properties: dict = None) -> Tuple[Objects, float]:
        """ Initializes the tracker. This method is used to initialize the tracker. It starts the tracker process if it is not running yet.

        Args:
            frame: The initial frame.
            new: The initial objects.
            properties: The initial properties.

        Returns:
            A tuple containing the initial objects and the initial score.
        """
        try:
            tproperties = dict(self._arguments)

            if not properties is None:
                tproperties.update(properties)

            return self._process.initialize(frame, new, properties)
        except TraxException as e:
            self._error(e)

    def update(self, frame: Frame, new: Objects = None, properties: dict = None) -> Tuple[Objects, float]:
        """ Updates the tracker. This method is used to update the tracker state with a new frame.

        Args:
            frame: The current frame.
            new: The current objects.
            properties: The current properties.

        Returns:
            A tuple containing the updated objects and the updated score.
        """
        try:
            if properties is None:
                properties = dict()
            return self._process.update(frame, new, properties)
        except TraxException as e:
            self._error(e)

    def stop(self):
        """ Stops the tracker. This method is used to stop the tracker. It stops the tracker process."""
        pass

    def __del__(self):
        """ Destructor. This method is used to stop the tracker process when the object is deleted."""
        self.stop()


def escape_path(path):
    """ Escapes a path. This method is used to escape a path.

    Args:
        path: The path to escape.

    Returns:
        The escaped path.
    """
    if sys.platform.startswith("win"):
        return path.replace("\\\\", "\\").replace("\\", "\\\\")
    else:
        return path


def trax_offline_adapter(tracker, command, envvars, paths="", log: bool = False, timeout: int = 30, linkpaths=None,
                         arguments=None, python=None, socket=False, restart=False, **kwargs):
    """ Creates a Python adapter for a tracker. This method is used to create a Python adapter for a tracker.

    Args:
        tracker: The tracker to create the adapter for.
        command: The command to run the tracker.
        envvars: The environment variables to set.
        paths: The paths to add to the Python path.
        log: Whether to log the tracker output.
        timeout: The timeout in seconds.
        linkpaths: The paths to link.
        arguments: The arguments to pass to the tracker.
        python: The Python interpreter to use.
        socket: Whether to use a socket to communicate with the tracker.
        restart: Whether to restart the tracker after each frame.
        kwargs: Additional keyword arguments.

    Returns:
        The Python TraX runtime object.
    """
    return TraxTrackerRuntimeOffline(tracker, command, log=log, timeout=timeout, linkpaths=linkpaths, envvars=envvars,
                                     arguments=arguments, socket=socket, restart=restart)
