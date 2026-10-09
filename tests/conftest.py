"""Shared test setup.

The automated tests cover pure logic only, so they must run on a machine with
no microphone, no PortAudio, no Whisper weights and no API key. Heavy or
hardware-bound modules are replaced with stubs when they cannot be imported.
"""

import os
import sys
import types

import pytest

# Make the project root importable (tests/ is a package, modules live above it)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _stub_if_unavailable(name, **attrs):
    try:
        __import__(name)
    except (ImportError, OSError):  # OSError: PortAudio library missing
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


_stub_if_unavailable("sounddevice")
_stub_if_unavailable("faster_whisper", WhisperModel=object)


class _FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel (no weights downloaded)."""

    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def speech_analyzer(monkeypatch):
    import speech_analyzer as module

    monkeypatch.setattr(module, "WhisperModel", _FakeWhisperModel)
    return module.SpeechAnalyzer()
