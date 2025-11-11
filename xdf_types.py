"""Type definitions for XDF converted JSON data structures."""

from typing import Literal, TypedDict, Union


class FileHeaderInfo(TypedDict):
    """File header information."""

    version: list[str]
    datetime: list[str]


class FileHeader(TypedDict):
    """Top-level file header."""

    info: FileHeaderInfo


class StreamInfo(TypedDict):
    """Metadata about a data stream."""

    name: str
    type: str
    channel_count: str
    nominal_srate: str
    channel_format: str
    stream_id: int
    source_id: str
    created_at: str
    uid: str
    session_id: str
    hostname: str
    desc: list[Union[dict, None]]


class ClockOffsetMeasurement(TypedDict):
    """A single clock offset measurement."""

    time: list[str]
    value: list[str]


class ClockOffsetCollection(TypedDict):
    """Collection of clock offset measurements."""

    offset: list[ClockOffsetMeasurement]


class StreamFooterInfo(TypedDict):
    """Footer information summarizing the stream."""

    first_timestamp: list[str]
    last_timestamp: list[str]
    sample_count: list[str]
    clock_offsets: list[ClockOffsetCollection]


class StreamFooter(TypedDict):
    """Stream footer containing summary information."""

    info: StreamFooterInfo


class Stream(TypedDict):
    """A single data stream within the XDF file."""

    info: StreamInfo
    time_series: list[list[str]]
    time_stamps: list[float]
    footer: StreamFooter


class XDFData(TypedDict):
    """Root structure for XDF converted JSON data."""

    file_header: FileHeader
    streams: list[Stream]
