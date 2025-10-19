"""Functions to read and write MS-MINT files."""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import numpy as np
import io
import logging
import pathlib
from typing import Union, Optional, List, Dict, Any, Callable, Tuple, Literal, cast
from pathlib import Path as P
from datetime import date
from pyteomics import mzxml, mzml

try:
    from pyteomics import mzmlb

    MZMLB_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import pyteomics.mzmlb:\n{e}")  # Fixed typo: Cound → Could
    MZMLB_AVAILABLE = False


MS_FILE_COLUMNS = [
    "scan_id",
    "ms_level",
    "polarity",
    "scan_time",
    "mz",
    "intensity",
]


def ms_file_to_df(fn: Union[str, P], read_only: bool = False) -> Optional[pd.DataFrame]:
    """Read MS file and convert it to a pandas DataFrame.

    Args:
        fn: Filename or path to the MS file.
        read_only: Whether to only read the file without converting to DataFrame
            (for testing purposes). Default is False.

    Returns:
        DataFrame containing MS data, or None if the file cannot be read.
    """
    fn = str(fn)

    try:
        if fn.lower().endswith(".mzxml"):
            df = mzxml_to_df(fn, read_only=read_only)
        elif fn.lower().endswith(".mzml"):
            df = mzml_to_df(fn, read_only=read_only)
        elif fn.lower().endswith("hdf"):
            df = pd.read_hdf(fn)
        elif fn.lower().endswith(".feather"):
            df = pd.read_feather(fn)
        elif fn.lower().endswith(".parquet"):
            df = read_parquet(fn, read_only=read_only)
        elif fn.lower().endswith(".mzmlb"):
            df = mzmlb_to_df__pyteomics(fn, read_only=read_only)
        else:
            logging.error(f"Cannot read file {fn} of type {type(fn)}")
            return None
    except IndexError as e:
        logging.warning(f"{e}: {fn}")
        return None

    if read_only:
        return df
    else:
        # Compatibility with old schema
        df = df.rename(
            columns={
                "retentionTime": "scan_time",
                "intensity array": "intensity",
                "m/z array": "mz",
            }
        )
        if "scan_id" not in df.columns:
            df["scan_id"] = 0
        if "ms_level" not in df.columns:
            df["ms_level"] = 1
        # Set datatypes
        set_dtypes(df)
    return df


def mzxml_to_df(
    fn: Union[str, pathlib.Path],
    read_only: bool = False,
    time_unit_in_file: Literal["min", "sec"] = "min",
) -> Optional[pd.DataFrame]:
    """Read mzXML file and convert it to pandas DataFrame.

    Args:
        fn: Filename or path to the mzXML file.
        read_only: Whether to only read the file without converting to DataFrame
            (for testing purposes). Default is False.
        time_unit_in_file: The time unit used in the mzXML file.
            Must be either 'sec' or 'min'. Default is 'min'.

    Returns:
        DataFrame containing MS data, or None if read_only is True.

    Raises:
        AssertionError: If the filename does not end with '.mzxml'.
    """
    assert str(fn).lower().endswith(".mzxml"), fn

    with mzxml.MzXML(fn) as ms_data:
        data = [x for x in ms_data]

    if read_only:
        return None

    data = [_extract_mzxml(x) for x in data]
    df = pd.json_normalize(data, sep="_")

    # Convert retention time to seconds
    if time_unit_in_file == "min":
        df["scan_time"] = df["scan_time"].astype(np.float64) * 60.0

    df = df.explode(["mz", "intensity"])
    set_dtypes(df)
    return df.reset_index(drop=True)


def _extract_mzxml(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant data from mzXML spectrum.

    Args:
        data: Dictionary containing mzXML spectrum data.

    Returns:
        Dictionary with extracted scan information.
    """

    # Function modified to export a dictionary with extracted scan information from either MS1 or MS2
    ms_level = data["msLevel"]

    ms_data = {
        "scan_id": data["num"],
        "ms_level": data["msLevel"],
        "polarity": data.get("polarity"),
        "scan_time": data["retentionTime"],
        "mz": data["m/z array"],
        "intensity": data["intensity array"],
    }

    if ms_level == 2:
        polarity_str = 'Positive' if data.get("polarity") == '+' else 'Negative'
        mz_precursor = data['precursorMz'][0]['precursorMz']
        mz = data["m/z array"][0]

        filterLine_to_ELMAVEN = ' '.join([
            polarity_str,
            # 'ESI', 'SRM', 'ms2',
            f"{mz_precursor:.3f}",
            f"[{mz:.3f}]"
        ])
        ms_data |= {
            "mz_precursor": mz_precursor,
            "filterLine": data["filterLine"],
            "filterLine_to_ELMAVEN": filterLine_to_ELMAVEN
        }

    return ms_data


def mzml_to_pandas_df_pyteomics(fn: Union[str, P], **kwargs) -> Optional[pd.DataFrame]:
    """Deprecated function to read mzML files.

    Args:
        fn: Filename or path to the mzML file.
        **kwargs: Additional arguments passed to mzml_to_df.

    Returns:
        DataFrame containing MS data, or None if read_only is True.
    """
    # warnings.warn("mzml_to_pandas_df_pyteomics() is deprecated use mzxml_to_df() instead", DeprecationWarning)
    return mzml_to_df(fn, **kwargs)


def mzml_to_df(fn: Union[str, P], read_only: bool = False) -> Optional[pd.DataFrame]:
    """Read mzML file and convert it to pandas DataFrame using the mzML library.

    Args:
        fn: Filename or path to the mzML file.
        read_only: Whether to only read the file without converting to DataFrame
            (for testing purposes). Default is False.

    Returns:
        DataFrame containing MS data, or None if read_only is True.

    Raises:
        AssertionError: If the filename does not end with '.mzml'.
    """
    assert str(fn).lower().endswith(".mzml"), fn

    # Read mzML file using pyteomics
    with mzml.read(str(fn)) as reader:
        # Initialize empty lists for the slices and the attributes
        slices = []
        # Loop through the spectra and extract the data
        for spectrum in reader:
            time_unit = spectrum["scanList"]["scan"][0]["scan start time"].unit_info
            if read_only:
                continue
            # Extract the scan ID, retention time, m/z values, and intensity values
            scan_id = int(spectrum["id"].split("=")[-1])
            rt = spectrum["scanList"]["scan"][0]["scan start time"]
            if time_unit == "minute":
                rt = rt * 60.0
            mz = np.array(spectrum["m/z array"], dtype=np.float64)
            intensity = np.array(spectrum["intensity array"], dtype=np.float64)
            if "positive scan" in spectrum.keys():
                polarity = "+"
            elif "negative scan" in spectrum.keys():
                polarity = "-"
            else:
                polarity = None
            ms_level = spectrum["ms level"]
            slices.append(
                pd.DataFrame(
                    {
                        "scan_id": scan_id,
                        "mz": mz,
                        "intensity": intensity,
                        "polarity": polarity,
                        "ms_level": ms_level,
                        "scan_time": rt,
                    }
                )
            )
    if read_only:
        return None
    df = pd.concat(slices)
    df["intensity"] = df["intensity"].astype(int)
    df = df[MS_FILE_COLUMNS].reset_index(drop=True)
    set_dtypes(df)
    return df


def set_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Set appropriate data types for MS data columns.

    Args:
        df: DataFrame containing MS data.

    Returns:
        DataFrame with appropriate data types.
    """
    dtypes = dict(
        mz=np.float32,
        scan_id=np.int64,
        ms_level=np.int8,
        scan_time=np.float32,
        intensity=np.int64,
        # add data types for the new columns from MS2 data
        mz_precursor=np.float32,
        filterLine=str,
        filterLine_to_ELMAVEN=str,
    )

    for var, dtype in dtypes.items():
        if var in df.columns and not df[var].dtype == dtype:
            df[var] = df[var].astype(dtype)

    return df


def _extract_mzml(data: Any, time_unit: str) -> Dict[str, Any]:
    """Extract relevant data from mzML spectrum.

    Args:
        data: Object containing mzML spectrum data.
        time_unit: Time unit used in the mzML file.

    Returns:
        Dictionary with extracted scan information.
    """
    RT = data.scan_time_in_minutes() * 60
    peaks = data.peaks("centroided")
    return {
        "scan_id": data["id"],
        "ms_level": data.ms_level,
        "polarity": "+" if data["positive scan"] else "-",
        "scan_time": RT,
        "mz": peaks[:, 0].astype("float64"),
        "intensity": peaks[:, 1].astype("int64"),
    }


extract_mzml = np.vectorize(_extract_mzml)


def read_parquet(fn: Union[str, P], read_only: bool = False) -> pd.DataFrame:
    """Read parquet file and return a pandas DataFrame.

    Args:
        fn: Filename or path to the parquet file.
        read_only: Whether to return the DataFrame as-is without formatting.
            Default is False.

    Returns:
        DataFrame containing MS data.
    """
    df = pd.read_parquet(fn)
    if read_only or (
        len(df.columns) == len(MS_FILE_COLUMNS) and all(df.columns == MS_FILE_COLUMNS)
    ):
        return df
    else:
        return format_thermo_raw_file_reader_parquet(df)


def format_thermo_raw_file_reader_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame from Thermo Raw File Reader to MS-MINT standard format.

    Args:
        df: DataFrame from Thermo Raw File Reader.

    Returns:
        Formatted DataFrame in MS-MINT standard format.
    """
    df = (
        df[["ScanNumber", "MsOrder", "RetentionTime", "Intensities", "Masses"]]
        .set_index(
            [
                "ScanNumber",
                "MsOrder",
                "RetentionTime",
            ]
        )
        .apply(pd.Series.explode)
        .reset_index()
        .rename(
            columns={
                "ScanNumber": "scan_id",
                "MsOrder": "ms_level",
                "RetentionTime": "scan_time",
                "Masses": "mz",
                "Intensities": "intensity",
            }
        )
    )
    df["scan_time"] = df["scan_time"] * 60
    df["polarity"] = None
    df["intensity"] = df.intensity.astype(np.float64)
    df = df[MS_FILE_COLUMNS]
    return df


def mzmlb_to_df__pyteomics(fn: Union[str, P], read_only: bool = False) -> Optional[pd.DataFrame]:
    """Read mzMLb file and convert it to pandas DataFrame using the pyteomics library.

    Args:
        fn: Filename or path to the mzMLb file.
        read_only: Whether to only read the file without converting to DataFrame
            (for testing purposes). Default is False.

    Returns:
        DataFrame containing MS data, or None if read_only is True.
    """
    if not MZMLB_AVAILABLE:
        logging.error("mzmlb support is not available")
        return None

    with mzmlb.MzMLb(fn) as ms_data:
        data = [x for x in ms_data]

    if read_only:
        return None

    data = list(extract_mzmlb(data))
    df = (
        pd.DataFrame.from_dict(data)
        .set_index(["index", "retentionTime", "polarity"])
        .apply(pd.Series.explode)
        .reset_index()
        .rename(
            columns={
                "index": "scan_id",
                "retentionTime": "scan_time",
                "m/z array": "mz",
                "ms level": "ms_level",
                "intensity array": "intensity",
            }
        )
    )

    # mzMLb starts scan index with 0
    df["scan_id"] = df["scan_id"] + 1

    df = df[MS_FILE_COLUMNS]
    return df


def _extract_mzmlb(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant data from mzMLb spectrum.

    Args:
        data: Dictionary containing mzMLb spectrum data.

    Returns:
        Dictionary with extracted scan information.
    """
    cols = ["index", "ms level", "polarity", "retentionTime", "m/z array", "intensity array"]
    data["retentionTime"] = data["scanList"]["scan"][0]["scan start time"] * 60
    if "positive scan" in data.keys():
        data["polarity"] = "+"
    elif "negative scan" in data.keys():
        data["polarity"] = "-"
    else:
        data["polarity"] = None
    return {c: data[c] for c in cols}


extract_mzmlb = np.vectorize(_extract_mzmlb)


def df_to_numeric(df: pd.DataFrame) -> None:
    """Convert dataframe columns to numeric types where possible.

    Args:
        df: DataFrame to convert. Modified in-place.
    """
    for col in df.columns:
        df.loc[:, col] = pd.to_numeric(df[col], errors="ignore")


def export_to_excel(
    mint: "ms_mint.Mint.Mint", fn: Optional[Union[str, P]] = None
) -> Optional[io.BytesIO]:
    """Export MINT state to Excel file.

    Args:
        mint: Mint instance containing data to export.
        fn: Output filename. If None, returns a file buffer instead of writing to disk.

    Returns:
        BytesIO buffer if fn is None, otherwise None.
    """
    date_string = str(date.today())
    if fn is None:
        file_buffer = io.BytesIO()
        writer = pd.ExcelWriter(file_buffer)
    else:
        writer = pd.ExcelWriter(fn)
    # Write into file
    mint.targets.to_excel(writer, sheet_name="Targets", index=False)
    mint.results.to_excel(writer, sheet_name="Results", index=False)
    meta = pd.DataFrame({"MINT_version": [mint.version], "Date": [date_string]}).T[0]
    meta.to_excel(writer, sheet_name="Metadata", index=True, header=False)
    # Close writer and maybe return file buffer
    writer.close()
    if fn is None:
        file_buffer.seek(0)
        return file_buffer
    return None


def convert_ms_file_to_feather(fn: Union[str, P], fn_out: Optional[Union[str, P]] = None) -> str:
    """Convert MS file to feather format.

    Args:
        fn: Filename or path to the MS file to convert.
        fn_out: Output filename or path. If None, uses the same path with '.feather' extension.

    Returns:
        Path to the generated feather file.
    """
    fn = P(fn)
    df = ms_file_to_df(fn)
    if df['ms_level'].unique() == [1]:
        ms_name = 'ms1'
    elif df['ms_level'].unique() == [2]:
        ms_name = 'ms2'
    else:
        ms_name = 'msunknown'

    if fn_out is None:
        fn_out = fn.with_suffix(".feather")
    # change the filename to add the ms level
    fn_out = fn_out.with_name(f"{fn_out.stem}_{ms_name}{fn_out.suffix}")

    if df is not None:
        df = df.reset_index(drop=True)
        df.to_feather(fn_out)
    return str(fn_out)



def convert_ms_file_to_parquet(fn: Union[str, P], fn_out: Optional[Union[str, P]] = None) -> str:
    """Convert MS file to parquet format.

    Args:
        fn: Filename or path to the MS file to convert.
        fn_out: Output filename or path. If None, uses the same path with '.parquet' extension.

    Returns:
        Path to the generated parquet file.
    """
    fn = P(fn)
    if fn_out is None:
        fn_out = fn.with_suffix(".parquet")
    df = ms_file_to_df(fn)
    if df is not None:
        df = df.reset_index(drop=True)
        df.to_parquet(fn_out)
    return str(fn_out)

def convert_mzxml_to_parquet_pl(file_path: str, time_unit='min', remove_original: bool = False,
                                tmp_dir: Optional[str] = None):
    file_path = pathlib.Path(file_path)
    # TODO: is this needed?
    # T.fix_first_emtpy_line_after_upload_workaround(file_path)
    from pyteomics import mzxml
    import polars as pl

    ms_level = 0
    polarity = None
    time_factor = 60 if time_unit in ['minutes', 'min'] else 1

    with mzxml.read(file_path.as_posix()) as ms_file_data:
        ms_data = []
        for i, data in enumerate(ms_file_data, start=1):
            ms_level = int(data.get("msLevel", 0))

            filterLine_ELMAVEN = None
            mz_precursor = None

            polarity_str = 'Positive' if data.get("polarity") == '+' else 'Negative'
            if not polarity:
                polarity = polarity_str
            if ms_level == 2:
                mz_precursor = float(data['precursorMz'][0]['precursorMz'])
                mz = data["m/z array"][0]
                filterLine_ELMAVEN = ' '.join([
                    polarity_str,
                    # 'ESI', 'SRM', 'ms2',
                    f"{mz_precursor:.3f}",
                    f"[{mz:.3f}]"
                ])
            ms_data.append(
                dict(
                    ms_file_label=file_path.stem,
                    scan_id=int(data.get("num") or 0),  # scan id
                    mz=[float(v) for v in data.get("m/z array", [])],  # mz
                    intensity=[float(v) for v in data.get("intensity array", [])],  # intensity
                    scan_time=float(data.get("retentionTime", 0.0)) * time_factor,  # scan time
                    mz_precursor=mz_precursor,  # mz precursor
                    filterLine=data.get("filterLine"),  # filter line
                    filterLine_ELMAVEN=filterLine_ELMAVEN  # filter line ELMAVEN
                )
            )

        df = pl.from_dicts(ms_data)
        df = df.explode(["mz", "intensity"])
        df = df.sort('mz' if ms_level == 1 else 'filterLine')

        if not tmp_dir:
            tmp_dir = tempfile.mkdtemp()
        tmp_fn = pathlib.Path(tmp_dir, f"{file_path.stem}.parquet")
        df.write_parquet(tmp_fn)
        if remove_original:
            os.remove(file_path)
    return file_path, file_path.stem, ms_level, polarity, tmp_fn.as_posix()


import pyarrow as pa
from pyarrow import parquet as pq
import re
import base64
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator


_RT_SECONDS = re.compile(
    r"^P(?:T(?:(?P<h>\d+(?:\.\d+)?)H)?(?:(?P<m>\d+(?:\.\d+)?)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?)$",
    re.I
)

def rt_to_seconds(val) -> float:
    """Convierte retentionTime a segundos (si viene PT…); si ya es numérico, lo devuelve tal cual."""
    if isinstance(val, (int, float)):
        return float(val)
    s = (val or "").strip()
    # si ya viene "0.12345" lo tomamos como segundos
    try:
        return float(s)
    except ValueError:
        pass
    m = _RT_SECONDS.match(s)
    if not m:
        return 0.0
    h = float(m.group("h") or 0.0)
    mi = float(m.group("m") or 0.0)
    se = float(m.group("s") or 0.0)
    return h*3600.0 + mi*60.0 + se


def _decode_peaks_optimized(attrs: Dict[str, str], text: Optional[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    OPTIMIZACIÓN CLAVE: Decodifica mz e intensity en UNA SOLA operación
    usando structured arrays (como pyteomics).

    Esto es ~2-3x más rápido que decodificar por separado.
    """
    if not text:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Determinar dtype según precisión
    dt = np.float32 if attrs.get("precision") == "32" else np.float64

    # CLAVE: Crear structured dtype para ambos arrays (mz, intensity)
    # byteorder '>' = big-endian (network byte order)
    endian = ">" if attrs.get("byteOrder") in ("network", "big") else "<"
    dtype = np.dtype([("mz", dt), ("intensity", dt)]).newbyteorder(endian)

    # Decodificar base64
    raw = base64.b64decode(text)

    # Descomprimir si es necesario
    if attrs.get("compressionType") == "zlib":
        raw = zlib.decompress(raw)

    # UNA SOLA conversión de bytes a arrays
    arr = np.frombuffer(raw, dtype=dtype)

    # Extraer campos del structured array (sin copia, solo vistas)
    return arr["mz"], arr["intensity"]

def iter_mzxml_fast(path: str | Path, *, decode_binary: bool = True) -> Iterator[Dict[str, Any]]:

    from lxml import etree  # ← IMPORTACIÓN CRÍTICA

    path = Path(path)

    # CAMBIO CLAVE: lxml.etree con remove_comments=True
    context = etree.iterparse(
        path.as_posix(),
        events=("start", "end"),
        remove_comments=True,  # Acelera el parsing
        huge_tree=False,       # Seguridad (default)
    )

    # Get root para limpiar memoria
    _, root = next(context)

    current: Dict[str, Any] = {}
    have_peaks = False

    for ev, elem in context:
        # lxml usa .tag directamente (sin namespace por defecto en mzXML)
        tag = elem.tag
        if '}' in tag:  # Solo si hay namespace
            tag = tag.rsplit("}", 1)[-1]

        if ev == "start" and tag == "scan":
            a = elem.attrib
            current = {
                "num": int(a.get("num", "0")),
                "msLevel": int(a.get("msLevel", "0")),
                "retentionTime": rt_to_seconds(a.get("retentionTime", "0")),
                "polarity": (
                    "Positive" if a.get("polarity") == "+"
                    else ("Negative" if a.get("polarity") == "-" else None)
                ),
                "filterLine": a.get("filterLine"),
            }
            have_peaks = False

        elif ev == "end" and tag == "precursorMz":
            txt = (elem.text or "").strip()
            if txt:
                try:
                    current["precursorMz"] = float(txt)
                except ValueError:
                    pass

        elif ev == "end" and tag == "peaks":
            if decode_binary:
                # OPTIMIZACIÓN CLAVE: Usa la versión optimizada
                mz, it = _decode_peaks_optimized(elem.attrib, (elem.text or "").strip() or None)
                current["m/z array"] = mz
                current["intensity array"] = it
                have_peaks = True
            else:
                current["peaks"] = {"attrs": dict(elem.attrib), "text": elem.text}

        elif ev == "end" and tag == "scan":
            # ELMAVEN-like extra (opcional)
            if current.get("msLevel") == 2 and have_peaks:
                pol_str = current.get("polarity") or ""
                prec = current.get("precursorMz")
                mz_arr = current.get("m/z array", [])
                mz0 = float(mz_arr[0]) if len(mz_arr) else None
                if prec is not None and mz0 is not None:
                    current["filterLine_ELMAVEN"] = f"{pol_str} {prec:.3f} [{mz0:.3f}]"

            yield current
            root.clear()  # libera memoria



BATCH_SIZE_POINTS = 50_000_000


def _build_table_from_lists(lists_dict: Dict[str, List], ms_level: int) -> pa.Table:
    """Helper para convertir las listas actuales en un pa.Table."""
    if ms_level == 1:
        arrays_dict = {
            'ms_file_label': pa.array(lists_dict['labels'], type=pa.string()),
            'scan_id': pa.array(lists_dict['scan_ids'], type=pa.int32()),
            'mz': pa.array(lists_dict['mzs'], type=pa.float64()),
            'intensity': pa.array(lists_dict['intensities'], type=pa.float64()),
            'scan_time': pa.array(lists_dict['scan_times'], type=pa.float64()),
        }
    else:  # MS2
        arrays_dict = {
            'ms_file_label': pa.array(lists_dict['labels'], type=pa.string()),
            'scan_id': pa.array(lists_dict['scan_ids'], type=pa.int32()),
            'mz': pa.array(lists_dict['mzs'], type=pa.float64()),
            'intensity': pa.array(lists_dict['intensities'], type=pa.float64()),
            'scan_time': pa.array(lists_dict['scan_times'], type=pa.float64()),
            'mz_precursor': pa.array(lists_dict['mz_precursors'], type=pa.float64()),
            'filterLine': pa.array(lists_dict['filterLines'], type=pa.string()),
            'filterLine_ELMAVEN': pa.array(lists_dict['filterLines_ELMAVEN'], type=pa.string()),
        }
    return pa.Table.from_pydict(arrays_dict)


def _init_lists() -> Dict[str, List]:
    """Helper para inicializar/resetear las listas."""
    return {
        'labels': [], 'scan_ids': [], 'scan_times': [],
        'mzs': [], 'intensities': [],
        'mz_precursors': [], 'filterLines': [], 'filterLines_ELMAVEN': []
    }


def convert_mzxml_to_parquet_fast_batches(
        file_path: str,
        time_unit: str = "min",
        remove_original: bool = False,
        tmp_dir: Optional[str] = None,
):

    file_path = Path(file_path)
    time_factor = 60.0 if time_unit == "min" else 1.0
    file_stem = file_path.stem

    # --- INICIO LÓGICA DE LOTES ---
    table_batches = []
    current_lists = _init_lists()
    # --- FIN LÓGICA DE LOTES ---

    ms_level = None
    polarity = None
    first_scan = True

    total_points = 0  # Contador para los lotes

    for data in iter_mzxml_fast(file_path.as_posix(), decode_binary=True):
        mz_arr = data.get("m/z array")
        if mz_arr is None or len(mz_arr) == 0:
            continue

        inten_arr = data.get("intensity array")
        n_points = len(mz_arr)
        total_points += n_points

        if first_scan:
            ms_level = int(data.get("msLevel", 0))
            polarity = "Positive" if data.get("polarity") == "+" else "Negative"
            first_scan = False

        scan_id = int(data.get("num", 0))
        scan_time = float(data.get("retentionTime", 0.0)) * time_factor

        current_lists['labels'].extend([file_stem] * n_points)
        current_lists['scan_ids'].extend([scan_id] * n_points)
        current_lists['scan_times'].extend([scan_time] * n_points)
        current_lists['mzs'].extend(mz_arr)
        current_lists['intensities'].extend(inten_arr)

        if ms_level == 2:
            mz_prec = None
            fline = data.get("filterLine")
            fline_elm = None
            try:
                mz_prec = float(data["precursorMz"][0]["precursorMz"])
                if mz_prec is not None and n_points > 0:
                    fline_elm = f"{polarity} {mz_prec:.3f} [{mz_arr[0]:.3f}]"
            except (KeyError, IndexError, TypeError):
                pass
            current_lists['mz_precursors'].extend([mz_prec] * n_points)
            current_lists['filterLines'].extend([fline] * n_points)
            current_lists['filterLines_ELMAVEN'].extend([fline_elm] * n_points)
        else:
            current_lists['mz_precursors'].extend([None] * n_points)
            current_lists['filterLines'].extend([None] * n_points)
            current_lists['filterLines_ELMAVEN'].extend([None] * n_points)


        # --- INICIO LÓGICA DE LOTES ---
        # Comprobar si el lote actual ha superado el umbral
        if total_points > BATCH_SIZE_POINTS:
            table_batches.append(_build_table_from_lists(current_lists, ms_level))
            # Resetear listas y contador
            current_lists = _init_lists()
            total_points = 0
        # --- FIN LÓGICA DE LOTES ---

    # --- INICIO LÓGICA DE LOTES ---
    # Añadir el último lote si queda algo
    if total_points > 0:
        table_batches.append(_build_table_from_lists(current_lists, ms_level))

    # Si no se leyó nada (archivo vacío o inválido)
    if not table_batches:
        print(f"Advertencia: No se encontraron datos válidos en {file_path}")
        # Retornar con valores nulos o manejar como error
        return 0, file_path, file_stem, 1, "Unknown", None

    table = pa.concat_tables(table_batches)

    if ms_level is None: ms_level = 1
    if polarity is None: polarity = "Unknown"

    if not tmp_dir:
        tmp_dir = tempfile.mkdtemp()
    tmp_fn = pathlib.Path(tmp_dir, f"{file_stem}.parquet")

    pq.write_table(
        table,
        tmp_fn,
    )

    if remove_original:
        try:
            os.remove(file_path)
        except OSError:
            pass

    return file_path, file_stem, ms_level, polarity, tmp_fn.as_posix()