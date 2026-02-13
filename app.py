import io
import zipfile
import os
from pathlib import Path

try:
    from google.cloud import storage
except Exception:
    storage = None
from dataclasses import dataclass
from datetime import date, datetime
from calendar import monthrange

import pandas as pd
import streamlit as st
from dateutil import parser as dtparser
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# -----------------------------
# PDF font handling
# Some PDF viewers fake-bold base fonts by drawing the same text multiple times.
# Embedding a real TTF font avoids the "duplicated/offset text" look.
# -----------------------------
PDF_FONT_REG = "Helvetica"
PDF_FONT_BOLD = "Helvetica-Bold"

def _init_pdf_fonts():
    global PDF_FONT_REG, PDF_FONT_BOLD
    if getattr(_init_pdf_fonts, "_done", False):
        return
    _init_pdf_fonts._done = True

    candidates = [
        ("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("LiberationSans", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ]

    for base_name, reg_path, bold_path in candidates:
        if os.path.exists(reg_path) and os.path.exists(bold_path):
            try:
                pdfmetrics.registerFont(TTFont(base_name, reg_path))
                pdfmetrics.registerFont(TTFont(base_name + "-Bold", bold_path))
                PDF_FONT_REG = base_name
                PDF_FONT_BOLD = base_name + "-Bold"
                return
            except Exception:
                pass


# -----------------------------
# Branding (optional banner image at top of each PDF page)
# Put your banner image in the repo at: assets/OUSD.png
# You can override the path with env var BRAND_BANNER_PATH
# -----------------------------
BRAND_BANNER_PATH = os.environ.get("BRAND_BANNER_PATH", "assets/OUSD.png")
# Header bar color under the banner (slightly darker OUSD blue).
# Override in Cloud Run env vars:
#   HEADER_BAR_COLOR_HEX="#0B5A8A"
#   HEADER_BAR_TEXT_COLOR_HEX="#FFFFFF"
HEADER_BAR_COLOR_HEX = os.environ.get("HEADER_BAR_COLOR_HEX", "#0B5A8A")
HEADER_BAR_TEXT_COLOR_HEX = os.environ.get("HEADER_BAR_TEXT_COLOR_HEX", "#FFFFFF")
_BRAND_CACHE = {"img": None, "size": None, "path": None}

def _load_brand_banner():
    """Returns (ImageReader, (w,h)) or (None, None) if not found."""
    if _BRAND_CACHE["img"] is not None:
        return _BRAND_CACHE["img"], _BRAND_CACHE["size"]
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    cand = os.path.join(base_dir, BRAND_BANNER_PATH)
    if not os.path.exists(cand):
        _BRAND_CACHE["img"] = None
        _BRAND_CACHE["size"] = None
        _BRAND_CACHE["path"] = None
        return None, None
    try:
        img = ImageReader(cand)
        _BRAND_CACHE["img"] = img
        _BRAND_CACHE["size"] = img.getSize()
        _BRAND_CACHE["path"] = cand
        return img, _BRAND_CACHE["size"]
    except Exception:
        _BRAND_CACHE["img"] = None
        _BRAND_CACHE["size"] = None
        _BRAND_CACHE["path"] = None
        return None, None

def _draw_brand_banner(c: canvas.Canvas, page_w: float, page_h: float, max_h: float = 0.65 * inch) -> float:
    """
    Draws the district banner at the very top of the page if available.
    Returns the y-coordinate of the bottom of the banner (or page_h if none).
    """
    img, size = _load_brand_banner()
    if img is None or not size:
        return page_h
    iw, ih = size
    if not iw or not ih:
        return page_h

    aspect = iw / ih
    draw_w = page_w
    draw_h = draw_w / aspect

    if draw_h > max_h:
        draw_h = max_h
        draw_w = draw_h * aspect

    x = (page_w - draw_w) / 2
    y = page_h - draw_h
    try:
        c.drawImage(img, x, y, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
        return y
    except Exception:
        return page_h

def _draw_report_header(c: canvas.Canvas, title: str, right_title: str = "", right_sub: str = "") -> float:
    """
    Draws (optional) banner + black title bar. Returns y-coordinate just below the title bar.
    """
    width, height = letter
    margin = 0.70 * inch
    x0 = margin
    xR = width - margin

    banner_bottom = _draw_brand_banner(c, width, height)  # y of bottom of banner (or height)

    bar_h = 0.62 * inch
    bar_top = banner_bottom
    bar_bottom = bar_top - bar_h

    c.saveState()
    c.setFillColor(HexColor(HEADER_BAR_COLOR_HEX))
    c.rect(0, bar_bottom, width, bar_h, stroke=0, fill=1)
    c.setFillColor(HexColor(HEADER_BAR_TEXT_COLOR_HEX))
    c.setFont(PDF_FONT_BOLD, 16)
    c.drawString(x0, bar_top - 0.34 * inch, title)
    if right_title:
        c.setFont(PDF_FONT_REG, 10)
        c.drawRightString(xR, bar_top - 0.34 * inch, right_title)
    if right_sub:
        c.setFont(PDF_FONT_REG, 8.5)
        c.setFillColor(colors.lightgrey)
        c.drawRightString(xR, bar_top - 0.54 * inch, right_sub)
        c.setFillColor(HexColor(HEADER_BAR_TEXT_COLOR_HEX))
    c.restoreState()

    return bar_bottom

try:
    from PyPDF2 import PdfMerger
except Exception:  # pragma: no cover
    PdfMerger = None

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="IAB/FIAB Progress Reports", layout="wide")

GROWTH_THRESHOLDS_DEFAULT = {
    "green_min": 20,
    "yellow_min": 1,
    "red_max": -1,
}

DEMOGRAPHIC_COLS = [
    "EnglishLanguageAcquisitionStatus",
    "Language",
    "HispanicOrLatinoEthnicity",
    "AmericanIndianOrAlaskaNative",
    "Asian",
    "BlackOrAfricanAmerican",
    "White",
    "NativeHawaiianOrOtherPacificIslander",
    "DemographicRaceTwoOrMoreRaces",
    "Filipino",
]

# Results-file aliases (your sample uses these)
RESULTS_ALIASES = {
    "Error Band Min": "ScaleScoreErrorBandMin",
    "Error Band Max": "ScaleScoreErrorBandMax",
    "LastOrSurname": "LastName",
}

REQUIRED_RESULTS = ["StudentIdentifier", "SubmitDateTime", "AssessmentName", "ScaleScore"]


@dataclass
class Window:
    label: str
    start: date
    end: date


# -----------------------------
# CSV loading (handles cp1252/latin1 + preserves IDs)
# -----------------------------
def read_csv_any(uploaded_file) -> pd.DataFrame:
    """
    Reads a Streamlit-uploaded CSV with flexible encoding and dtype=str to preserve IDs.
    Uses caching keyed on raw file bytes for performance.
    """
    if uploaded_file is None:
        return None
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    return read_csv_bytes(raw)


@st.cache_data(show_spinner=False)
def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(raw), dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw), dtype=str, encoding_errors="replace")



# -----------------------------
# Reference-data auto load (GCS / assets)
# -----------------------------

def _norm_school_name(s: str) -> str:
    s = str(s or "").strip().lower()
    s = " ".join(s.split())
    return s

@st.cache_data(show_spinner=False)
def _read_local_bytes(path_str: str) -> bytes:
    return Path(path_str).read_bytes()

@st.cache_data(show_spinner=False)
def _download_gcs_bytes(bucket: str, blob: str) -> bytes:
    if storage is None:
        raise RuntimeError("google-cloud-storage is not installed in this runtime.")
    client = storage.Client()
    return client.bucket(bucket).blob(blob).download_as_bytes()

def _try_load_reference_raw():
    """Return (roster_raw, crosswalk_raw, section_raw, status_dict)."""
    status = {"source": None, "details": {}}

    bucket = os.environ.get("IAB_DATA_BUCKET", "").strip()
    roster_obj = os.environ.get("IAB_ROSTER_OBJECT", "roster.csv").strip()
    crosswalk_obj = os.environ.get("IAB_CROSSWALK_OBJECT", "crosswalk.csv").strip()
    section_obj = os.environ.get("IAB_SECTIONMAP_OBJECT", "sectionmap.csv").strip()

    roster_raw = crosswalk_raw = section_raw = None

    if bucket:
        try:
            roster_raw = _download_gcs_bytes(bucket, roster_obj)
            status["details"]["Roster"] = f"GCS: gs://{bucket}/{roster_obj}"
        except Exception as e:
            status["details"]["Roster"] = f"GCS load failed: {e}"

        try:
            crosswalk_raw = _download_gcs_bytes(bucket, crosswalk_obj)
            status["details"]["Crosswalk"] = f"GCS: gs://{bucket}/{crosswalk_obj}"
        except Exception as e:
            status["details"]["Crosswalk"] = f"GCS load failed: {e}"

        try:
            section_raw = _download_gcs_bytes(bucket, section_obj)
            status["details"]["SectionMap"] = f"GCS: gs://{bucket}/{section_obj}"
        except Exception as e:
            status["details"]["SectionMap"] = f"GCS load failed: {e}"

        if roster_raw is not None and section_raw is not None:
            status["source"] = "gcs"

    # Fallback to local assets (dev only)
    if status["source"] is None:
        assets_dir = Path(__file__).parent / "assets"
        for key, fname in [("Roster","roster.csv"), ("Crosswalk","crosswalk.csv"), ("SectionMap","sectionmap.csv")]:
            try:
                raw = _read_local_bytes(str(assets_dir / fname))
                if key == "Roster":
                    roster_raw = raw
                elif key == "Crosswalk":
                    crosswalk_raw = raw
                else:
                    section_raw = raw
                status["details"][key] = f"assets/{fname}"
            except Exception:
                pass

        if roster_raw is not None and section_raw is not None:
            status["source"] = "assets"

    return roster_raw, crosswalk_raw, section_raw, status




@st.cache_data(show_spinner=False)
def normalize_results_cached(raw: bytes) -> pd.DataFrame:
    return normalize_results(read_csv_bytes(raw))


@st.cache_data(show_spinner=False)
def normalize_crosswalk_cached(raw: bytes) -> pd.DataFrame:
    return normalize_crosswalk(read_csv_bytes(raw))


@st.cache_data(show_spinner=False)
def normalize_sectionmap_cached(raw: bytes) -> pd.DataFrame:
    return normalize_sectionmap(read_csv_bytes(raw))


@st.cache_data(show_spinner=False)
def normalize_roster_cached(raw: bytes) -> pd.DataFrame:
    return normalize_roster(read_csv_bytes(raw))

def strip_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
            # restore real NaNs from "nan"
            df[c] = df[c].replace({"nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "": pd.NA})
    return df


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    df = strip_df(df)

    # rename aliases
    for old, new in RESULTS_ALIASES.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Guarantee these exist (some exports omit names)
    for col in ("FirstName", "LastName", "SchoolName", "SchoolYear", "Subject", "GradeLevelWhenAssessed", "Status", "Reporting Category"):
        if col not in df.columns:
            df[col] = pd.NA

    # Required columns check
    missing = [c for c in REQUIRED_RESULTS if c not in df.columns]
    if missing:
        raise ValueError(f"Results CSV missing required columns: {missing}")

    # Parse datetime
    df["SubmitDateTimeParsed"] = df["SubmitDateTime"].apply(lambda x: pd.NaT if pd.isna(x) else dtparser.parse(str(x)))

    # Numeric fields
    for c in ("ScaleScore", "ScaleScoreErrorBandMin", "ScaleScoreErrorBandMax"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # IDs as strings
    df["StudentIdentifier"] = df["StudentIdentifier"].astype(str)

    return df


def normalize_crosswalk(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    df = strip_df(df)
    # Expected: StudentIdentifier, StudentID
    if "StudentIdentifier" not in df.columns or "StudentID" not in df.columns:
        raise ValueError("Crosswalk CSV must have headers: StudentIdentifier, StudentID")
    df["StudentIdentifier"] = df["StudentIdentifier"].astype(str)
    df["StudentID"] = df["StudentID"].astype(str)
    return df[["StudentIdentifier", "StudentID"]]


def normalize_sectionmap(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    df = strip_df(df)
    need = ["SectionNumber", "TeacherName", "Subject"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"SectionMap CSV missing required columns: {missing}")

    df["SectionNumber"] = df["SectionNumber"].astype(str)
    df["TeacherName"] = df["TeacherName"].astype(str)

    # Optional school/site identifier columns (recommended when section numbers collide across schools)
    if "SchoolCode" in df.columns:
        df["SchoolCode"] = df["SchoolCode"].astype(str)
    if "SchoolName" in df.columns:
        df["SchoolName"] = df["SchoolName"].astype(str)
        df["SchoolNameNorm"] = df["SchoolName"].map(_norm_school_name)

    # Subject can be blank for elementary "homeroom" sections.
    # Treat blank/CORE/HOMEROOM/ELEM/BOTH as "Both" so those students appear in BOTH Math and ELA outputs.
    subj_raw = df["Subject"].fillna("").astype(str).str.strip()
    subj_up = subj_raw.str.upper()

    subj_norm = pd.Series("Both", index=df.index)

    math_mask = subj_up.str.contains("MATH", na=False)
    ela_mask = subj_up.str.contains(r"ELA|ENGLISH|LANGUAGE|READ", regex=True, na=False)

    # If it matches both, keep Both.
    subj_norm.loc[math_mask & ~ela_mask] = "Math"
    subj_norm.loc[ela_mask & ~math_mask] = "ELA"

    both_mask = subj_up.str.contains("BOTH|CORE|HOMEROOM|HOME ROOM|ELEM|ELEMENTARY", regex=True, na=False) | (subj_raw == "")
    subj_norm.loc[both_mask] = "Both"

    df["SubjectNorm"] = subj_norm
    return df


def normalize_roster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes roster into: StudentIdentifier, StudentID, SectionNumber, LastName, FirstName

    Your CK sample has header: StudentID,SectionNumber,LastName,FirstName,,  (two blank headers)
    and rows like: StudentIdentifier, StudentID, Student#, Section#, LastName, FirstName
    """
    if df is None:
        return None
    df = strip_df(df)

    # If there are empty/Unnamed columns, try to map them to names
    unnamed = [c for c in df.columns if c.startswith("Unnamed") or c == ""]
    if len(unnamed) >= 2:
        # Prefer the last two unnamed cols as name fields if they look like text
        last_col, first_col = unnamed[-2], unnamed[-1]
        df = df.rename(columns={last_col: "LastNameText", first_col: "FirstNameText"})
        # If existing LastName/FirstName are numeric (mis-shifted), replace
        if "LastName" in df.columns and "FirstName" in df.columns:
            # If LastName column is mostly digits, treat it as not a name
            digits_ratio = df["LastName"].astype(str).str.fullmatch(r"\d+").mean()
            if digits_ratio > 0.6:
                df["LastName"] = df["LastNameText"]
                df["FirstName"] = df["FirstNameText"]
        else:
            df["LastName"] = df["LastNameText"]
            df["FirstName"] = df["FirstNameText"]

    # Heuristic: if StudentID looks like 9–10 digits (StudentIdentifier) and SectionNumber looks like 4–6 digits (StudentID)
    if "StudentID" in df.columns and "SectionNumber" in df.columns:
        def median_digits(s: pd.Series) -> float:
            v = s.astype(str).str.replace(r"\D", "", regex=True)
            v = v[v != ""]
            if v.empty:
                return 0
            return float(v.str.len().median())

        md_studentid = median_digits(df["StudentID"])
        md_section = median_digits(df["SectionNumber"])

        if md_studentid >= 9 and md_section <= 7:
            # interpret as StudentIdentifier + StudentID
            df = df.rename(columns={"StudentID": "StudentIdentifier", "SectionNumber": "StudentID"})
        else:
            # otherwise keep names as-is
            pass

    # Detect section number column
    # In CK sample, "FirstName" column is actually SectionNumber (numeric), and real names are in the extra columns.
    # If FirstName is mostly digits AND (SectionNumber exists but is StudentID after rename), use FirstName as SectionNumber.
    if "SectionNumber" not in df.columns:
        if "FirstName" in df.columns:
            df["SectionNumber"] = df["FirstName"]
    else:
        # if FirstName is digits-heavy, treat it as SectionNumber
        if "FirstName" in df.columns:
            digits_ratio = df["FirstName"].astype(str).str.fullmatch(r"\d+").mean()
            if digits_ratio > 0.6:
                df["SectionNumber"] = df["FirstName"]
    # Required final columns
    need = ["StudentIdentifier", "StudentID", "SectionNumber", "LastName", "FirstName"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Roster CSV couldn't be normalized; missing columns after heuristics: {missing}")

    # Optional school/site identifier columns (recommended when section numbers collide across schools)
    if "SchoolCode" in df.columns:
        df["SchoolCode"] = df["SchoolCode"].astype(str)
    if "SchoolName" in df.columns:
        df["SchoolName"] = df["SchoolName"].astype(str)
        df["SchoolNameNorm"] = df["SchoolName"].map(_norm_school_name)

    df["StudentIdentifier"] = df["StudentIdentifier"].astype(str)
    df["StudentID"] = df["StudentID"].astype(str)
    df["SectionNumber"] = df["SectionNumber"].astype(str)

    out_cols = need.copy()
    if "SchoolCode" in df.columns:
        out_cols.append("SchoolCode")
    if "SchoolName" in df.columns:
        out_cols.append("SchoolName")
    if "SchoolNameNorm" in df.columns:
        out_cols.append("SchoolNameNorm")

    return df[out_cols]



# -----------------------------
# Date/window helpers
# -----------------------------
def month_window(year: int, month: int) -> Window:
    last_day = monthrange(year, month)[1]
    return Window(
        label=f"{date(year, month, 1).strftime('%b %Y')}",
        start=date(year, month, 1),
        end=date(year, month, last_day),
    )


def fallback_oct_feb(df_results: pd.DataFrame) -> tuple[Window, Window]:
    # Use SchoolYear if present like "2025-26"
    if "SchoolYear" in df_results.columns and df_results["SchoolYear"].notna().any():
        sy = str(df_results["SchoolYear"].dropna().iloc[0])
        m = sy.replace(" ", "")
        # 2025-26
        try:
            start_year = int(m.split("-")[0])
            end_part = m.split("-")[1]
            end_year = int(str(start_year)[:2] + end_part) if len(end_part) == 2 else int(end_part)
        except Exception:
            start_year = df_results["SubmitDateTimeParsed"].dropna().min().year
            end_year = df_results["SubmitDateTimeParsed"].dropna().max().year
        return month_window(start_year, 10), month_window(end_year, 2)

    # otherwise infer from dates
    dts = df_results["SubmitDateTimeParsed"].dropna()
    if dts.empty:
        y = date.today().year
        return month_window(y, 10), month_window(y + 1, 2)
    min_dt = dts.min()
    start_year = min_dt.year
    end_year = start_year + 1
    return month_window(start_year, 10), month_window(end_year, 2)


def detect_two_windows(df_results: pd.DataFrame) -> tuple[Window, Window]:
    dts = df_results["SubmitDateTimeParsed"].dropna()
    if dts.empty:
        return fallback_oct_feb(df_results)

    ym = dts.dt.to_period("M").astype(str)
    counts = ym.value_counts()

    if len(counts) >= 2:
        top2 = counts.index[:2].tolist()
        top2_dt = [datetime.strptime(x, "%Y-%m") for x in top2]
        top2_dt.sort()
        w1 = month_window(top2_dt[0].year, top2_dt[0].month)
        w2 = month_window(top2_dt[1].year, top2_dt[1].month)
        return w1, w2

    return fallback_oct_feb(df_results)


def fmt_date(x) -> str:
    if pd.isna(x):
        return "N/A"
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.strftime("%b %d, %Y")
    return str(x)


# -----------------------------
# Core logic: pick attempts + growth
# -----------------------------



def _wrap_text(c: canvas.Canvas, text: str, max_width: float, font_name: str, font_size: float) -> list[str]:
    """Word-wrap text to fit within max_width on a ReportLab canvas."""
    if text is None:
        return [""]
    text = str(text)
    c.setFont(font_name, font_size)
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    line = words[0]
    for w in words[1:]:
        trial = line + " " + w
        if c.stringWidth(trial, font_name, font_size) <= max_width:
            line = trial
        else:
            lines.append(line)
            line = w
    lines.append(line)
    return lines

def _draw_card(c: canvas.Canvas, x: float, y: float, w: float, h: float, title: str | None = None, title_size: int = 11,
               fill=colors.whitesmoke, stroke=colors.lightgrey, radius: float = 10):
    """Rounded rectangle card with an optional title. (x,y) is bottom-left."""
    c.saveState()
    c.setStrokeColor(stroke)
    c.setFillColor(fill)
    c.roundRect(x, y, w, h, radius, stroke=1, fill=1)
    if title:
        c.setFillColor(colors.black)
        c.setFont(PDF_FONT_BOLD, title_size)
        c.drawString(x + 12, y + h - 18, title)
    c.restoreState()


def pick_latest_in_window(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    d0 = pd.Timestamp(window.start)
    d1 = pd.Timestamp(window.end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask = (df["SubmitDateTimeParsed"] >= d0) & (df["SubmitDateTimeParsed"] <= d1)
    sub = df.loc[mask].copy()
    if sub.empty:
        return sub
    sub.sort_values(["StudentIdentifier", "AssessmentName", "SubmitDateTimeParsed"], inplace=True)
    latest = sub.groupby(["StudentIdentifier", "AssessmentName"], as_index=False).tail(1)
    return latest


def build_growth_table(df: pd.DataFrame, w1: Window, w2: Window) -> pd.DataFrame:
    base = pick_latest_in_window(df, w1).copy()
    foll = pick_latest_in_window(df, w2).copy()

    keep_cols = [
        "StudentIdentifier", "FirstName", "LastName",
        "AssessmentName", "SchoolName", "SchoolYear", "Subject", "GradeLevelWhenAssessed",
        "ScaleScore", "ScaleScoreErrorBandMin", "ScaleScoreErrorBandMax", "Reporting Category", "Status",
        "SubmitDateTimeParsed",
    ] + [c for c in DEMOGRAPHIC_COLS if c in df.columns]

    base = base[[c for c in keep_cols if c in base.columns]].copy()
    foll = foll[[c for c in keep_cols if c in foll.columns]].copy()

    base = base.rename(columns={
        "ScaleScore": "BaselineScore",
        "ScaleScoreErrorBandMin": "BaselineBandMin",
        "ScaleScoreErrorBandMax": "BaselineBandMax",
        "Reporting Category": "BaselineCategory",
        "Status": "BaselineStatus",
        "SubmitDateTimeParsed": "BaselineDate",
    })

    foll = foll.rename(columns={
        "ScaleScore": "FollowupScore",
        "ScaleScoreErrorBandMin": "FollowupBandMin",
        "ScaleScoreErrorBandMax": "FollowupBandMax",
        "Reporting Category": "FollowupCategory",
        "Status": "FollowupStatus",
        "SubmitDateTimeParsed": "FollowupDate",
    })

    merged = pd.merge(
        base,
        foll,
        on=["StudentIdentifier", "AssessmentName"],
        how="outer",
        suffixes=("", "_y"),
    )

    # Prefer non-null names from either side
    for col in ["FirstName", "LastName", "SchoolName", "SchoolYear", "Subject", "GradeLevelWhenAssessed"]:
        cy = f"{col}_y"
        if cy in merged.columns:
            merged[col] = merged[col].fillna(merged[cy])
            merged.drop(columns=[cy], inplace=True)

    for col in DEMOGRAPHIC_COLS:
        cy = f"{col}_y"
        if col in merged.columns and cy in merged.columns:
            merged[col] = merged[col].fillna(merged[cy])
            merged.drop(columns=[cy], inplace=True)

    merged["Growth"] = pd.to_numeric(merged["FollowupScore"], errors="coerce") - pd.to_numeric(merged["BaselineScore"], errors="coerce")
    return merged


# -----------------------------
# PDF generation
# -----------------------------
def growth_color(growth: float, thresholds: dict):
    if pd.isna(growth):
        return colors.black
    if growth >= thresholds["green_min"]:
        return colors.green
    if growth >= thresholds["yellow_min"]:
        return colors.darkgoldenrod
    if growth <= thresholds["red_max"]:
        return colors.red
    return colors.black


def draw_student_one_pager(c: canvas.Canvas, row: pd.Series, w1: Window, w2: Window, thresholds: dict, teacher_label: str = "", subject_label: str = ""):
    """Student-facing 1-page PDF. Uses embedded fonts (when available) and generous spacing to prevent overlap."""
    _init_pdf_fonts()

    width, height = letter
    margin = 0.70 * inch
    x0 = margin
    xR = width - margin
    # ---------- Header ----------
    header_bottom = _draw_report_header(
        c,
        title="IAB/FIAB Growth Report",
        right_title="Latest attempt in each window",
        right_sub=f"Generated: {datetime.now().strftime('%b %d, %Y')} • Source: CA Interim (IAB/FIAB)",
    )

    def _month_label(w: Window) -> str:
        try:
            if w.start.month == w.end.month and w.start.year == w.end.year:
                return w.start.strftime("%b")
            return f"{w.start.strftime('%b')}–{w.end.strftime('%b')}"
        except Exception:
            return "Window"

    def _fit_bold(text, max_w, start=16, min_size=12):
        size = start
        s = str(text)
        while size > min_size and c.stringWidth(s, PDF_FONT_BOLD, size) > max_w:
            size -= 1
        return size

    # ---------- Data ----------
    student_name = f"{row.get('LastName','')}, {row.get('FirstName','')}".strip(", ").strip() or "Student"
    sid = str(row.get("StudentIdentifier","") or "")
    grade = str(row.get("GradeLevelWhenAssessed","") or "")

    def _grade_ordinal(g: str) -> str:
        s = str(g or "").strip()
        if not s:
            return ""
        if s.upper().startswith("G") and len(s) > 1:
            s = s[1:]
        if s.upper() in ("K", "KG", "KINDER", "KINDERGARTEN"):
            return "K"
        try:
            n = int(float(s))
        except Exception:
            return s
        if n % 100 in (11, 12, 13):
            suf = "th"
        else:
            suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"

    grade_lbl = _grade_ordinal(grade)
    schoolyear = str(row.get("SchoolYear","") or "")
    assessment = str(row.get("AssessmentName","") or "")

    # ---------- Student card ----------
    y_top = header_bottom - 0.35 * inch

    student_h = 1.18 * inch
    student_y = y_top - student_h
    card_w = width - 2 * margin
    _draw_card(c, x0, student_y, card_w, student_h, title=None, fill=colors.whitesmoke)

    c.setFont(PDF_FONT_BOLD, 9)
    c.setFillColor(colors.grey)
    c.drawString(x0 + 12, student_y + student_h - 16, "STUDENT")
    c.setFillColor(colors.black)

    name_size = _fit_bold(student_name, card_w - 24, start=16, min_size=12)
    c.setFont(PDF_FONT_BOLD, name_size)
    c.drawString(x0 + 12, student_y + student_h - 38, student_name)

    c.setFont(PDF_FONT_REG, 10)
    c.drawString(x0 + 12, student_y + student_h - 56, f"SSID: {sid}   •   Grade: {grade_lbl}   •   School Year: {schoolyear}")
    if teacher_label:
        c.drawString(x0 + 12, student_y + student_h - 72, f"Teacher: {teacher_label}   •   Subject: {subject_label}")

    # ---------- Assessment card ----------
    assess_h = 0.72 * inch
    assess_y = student_y - 0.18 * inch - assess_h
    _draw_card(c, x0, assess_y, card_w, assess_h, title=None, fill=colors.whitesmoke)

    c.setFont(PDF_FONT_BOLD, 9)
    c.setFillColor(colors.grey)
    c.drawString(x0 + 12, assess_y + assess_h - 16, "ASSESSMENT")
    c.setFillColor(colors.black)

    c.setFont(PDF_FONT_REG, 11)
    a_lines = _wrap_text(c, assessment, max_width=card_w - 24, font_name=PDF_FONT_REG, font_size=11)
    y_line = assess_y + assess_h - 38
    for ln in a_lines[:2]:
        c.drawString(x0 + 12, y_line, ln)
        y_line -= 14

    # ---------- Score row ----------
    gap = 10
    w3 = (width - 2 * margin - 2 * gap) / 3
    h3 = 2.35 * inch
    y_row_top = assess_y - 0.08 * inch
    y_cards = y_row_top - h3

    x1 = x0
    x2 = x0 + w3 + gap
    x3 = x0 + 2 * (w3 + gap)

    # Baseline
    _draw_card(c, x1, y_cards, w3, h3, title=f"Baseline ({_month_label(w1)})", fill=colors.whitesmoke)
    b_date = fmt_date(row.get("BaselineDate", pd.NaT))
    b_score = row.get("BaselineScore", pd.NA)
    b_min = row.get("BaselineBandMin", pd.NA)
    b_max = row.get("BaselineBandMax", pd.NA)
    b_cat = str(row.get("BaselineCategory","") or "")
    b_status = str(row.get("BaselineStatus","") or "")

    c.setFont(PDF_FONT_BOLD, 28)
    c.drawString(x1 + 14, y_cards + h3 - 70, "N/A" if pd.isna(b_score) else f"{int(b_score)}")

    c.setFont(PDF_FONT_REG, 9)
    c.setFillColor(colors.grey)
    c.drawString(x1 + 14, y_cards + h3 - 88, f"Taken: {b_date}")
    c.setFillColor(colors.black)

    c.setFont(PDF_FONT_REG, 10)
    c.drawString(x1 + 14, y_cards + h3 - 110, f"Score range: {'' if pd.isna(b_min) else int(b_min)}–{'' if pd.isna(b_max) else int(b_max)}")
    c.drawString(x1 + 14, y_cards + h3 - 126, f"Category: {b_cat}")
    c.drawString(x1 + 14, y_cards + h3 - 142, f"Status: {b_status}")

    # Follow-up
    _draw_card(c, x2, y_cards, w3, h3, title=f"Follow-up ({_month_label(w2)})", fill=colors.whitesmoke)
    f_date = fmt_date(row.get("FollowupDate", pd.NaT))
    f_score = row.get("FollowupScore", pd.NA)
    f_min = row.get("FollowupBandMin", pd.NA)
    f_max = row.get("FollowupBandMax", pd.NA)
    f_cat = str(row.get("FollowupCategory","") or "")
    f_status = str(row.get("FollowupStatus","") or "")

    c.setFont(PDF_FONT_BOLD, 28)
    c.drawString(x2 + 14, y_cards + h3 - 70, "N/A" if pd.isna(f_score) else f"{int(f_score)}")

    c.setFont(PDF_FONT_REG, 9)
    c.setFillColor(colors.grey)
    c.drawString(x2 + 14, y_cards + h3 - 88, f"Taken: {f_date}")
    c.setFillColor(colors.black)

    c.setFont(PDF_FONT_REG, 10)
    c.drawString(x2 + 14, y_cards + h3 - 110, f"Score range: {'' if pd.isna(f_min) else int(f_min)}–{'' if pd.isna(f_max) else int(f_max)}")
    c.drawString(x2 + 14, y_cards + h3 - 126, f"Category: {f_cat}")
    c.drawString(x2 + 14, y_cards + h3 - 142, f"Status: {f_status}")

    # Growth
    _draw_card(c, x3, y_cards, w3, h3, title="Growth", fill=colors.whitesmoke)
    g = row.get("Growth", pd.NA)
    g_color = growth_color(g, thresholds)

    # Mini sparkline (baseline → follow-up) inside Growth card for a more "report-like" feel
    try:
        bs = float(b_score) if (b_score is not None and not pd.isna(b_score)) else None
        fs = float(f_score) if (f_score is not None and not pd.isna(f_score)) else None
    except Exception:
        bs, fs = None, None

    if (bs is not None) or (fs is not None):
        spark_w = w3 - 28
        spark_h = 0.22 * inch
        spark_x = x3 + 14
        spark_top = y_cards + h3 - 34
        spark_y = spark_top - spark_h

        c.saveState()
        c.setLineWidth(2)
        c.setStrokeColor(g_color if (bs is not None and fs is not None) else colors.grey)
        c.setFillColor(colors.white)

        xL = spark_x + 6
        xR2 = spark_x + spark_w - 6

        if (bs is not None) and (fs is not None):
            vmin = min(bs, fs)
            vmax = max(bs, fs)
            if vmax == vmin:
                vmax = vmin + 1
            pad = (vmax - vmin) * 0.20
            low = vmin - pad
            high = vmax + pad

            def _mapy(v):
                return spark_y + (v - low) / (high - low) * spark_h

            yL = _mapy(bs)
            yR = _mapy(fs)

            c.line(xL, yL, xR2, yR)

            # Arrowhead on the right end of the sparkline
            dx = xR2 - xL
            dy = yR - yL
            L = (dx * dx + dy * dy) ** 0.5
            if L > 0:
                ux = dx / L
                uy = dy / L
                arrow_len = 10
                arrow_w = 5
                xb = xR2 - ux * arrow_len
                yb = yR - uy * arrow_len
                px = -uy
                py = ux
                xA = xb + px * arrow_w
                yA = yb + py * arrow_w
                xB = xb - px * arrow_w
                yB = yb - py * arrow_w

                pth = c.beginPath()
                pth.moveTo(xR2, yR)
                pth.lineTo(xA, yA)
                pth.lineTo(xB, yB)
                pth.close()
                c.setFillColor(g_color)
                c.setStrokeColor(g_color)
                c.drawPath(pth, stroke=0, fill=1)
                c.setFillColor(colors.white)

            c.setStrokeColor(g_color)
            c.setFillColor(g_color)
            c.circle(xL, yL, 3, stroke=1, fill=1)
            c.setFillColor(colors.black)

        else:
            # Only one point available
            v = bs if bs is not None else fs
            yM = spark_y + spark_h / 2
            xM = (xL + xR2) / 2
            c.setStrokeColor(colors.grey)
            c.circle(xM, yM, 3, stroke=1, fill=1)

        c.restoreState()

    pill_h = 0.82 * inch
    pill_w = w3 - 28
    pill_x = x3 + 14
    pill_y = y_cards + h3 - 118

    c.saveState()
    c.setFillColor(g_color)
    c.roundRect(pill_x, pill_y, pill_w, pill_h, 12, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont(PDF_FONT_BOLD, 32)
    gtxt = "N/A" if pd.isna(g) else f"{int(g):+d}"
    c.drawCentredString(pill_x + pill_w / 2, pill_y + 24, gtxt)
    c.restoreState()

    c.setFont(PDF_FONT_REG, 9)
    c.setFillColor(colors.grey)
    c.drawCentredString(x3 + w3 / 2, pill_y - 16, "Points (Follow-up − Baseline)")
    c.setFillColor(colors.black)

    if (not pd.isna(b_score)) and (not pd.isna(f_score)):
        c.setFont(PDF_FONT_BOLD, 12)
        c.drawCentredString(x3 + w3 / 2, pill_y - 32, f"{int(b_score)}  →  {int(f_score)}")
        # Error band overlap indicator (helps interpret small changes)
        overlap_txt = "N/A"
        try:
            if (not pd.isna(b_min)) and (not pd.isna(b_max)) and (not pd.isna(f_min)) and (not pd.isna(f_max)):
                overlap = max(float(b_min), float(f_min)) <= min(float(b_max), float(f_max))
                overlap_txt = "Yes" if overlap else "No"
        except Exception:
            overlap_txt = "N/A"

        c.setFont(PDF_FONT_REG, 9)
        c.setFillColor(colors.grey)
        c.drawCentredString(x3 + w3 / 2, pill_y - 48, f"Score ranges overlap: {overlap_txt}")
        c.setFillColor(colors.black)


    # ---------- Footer (single) ----------
    footer = (
        f"Baseline test date: {('N/A' if b_date in ('', 'N/A') else b_date)}  •  "
        f"Follow-up test date: {('N/A' if f_date in ('', 'N/A') else f_date)}  •  "
        f"Highlight: Green ≥ {thresholds['green_min']}  •  Gold ≥ {thresholds['yellow_min']}  •  Red ≤ {thresholds['red_max']}"
    )
    c.setFont(PDF_FONT_REG, 8.5)
    c.setFillColor(colors.grey)
    flines = _wrap_text(c, footer, max_width=(width - 2 * margin), font_name=PDF_FONT_REG, font_size=8.5)
    y_footer = y_cards - 0.50 * inch
    for i, ln in enumerate(flines[:3]):
        c.drawString(x0, y_footer - 10 * i, ln)
    c.setFillColor(colors.black)


def make_single_student_pdf(row: pd.Series, w1: Window, w2: Window, thresholds: dict, teacher_label: str = "", subject_label: str = "") -> bytes:
    b = io.BytesIO()
    cc = canvas.Canvas(b, pagesize=letter)
    draw_student_one_pager(cc, row, w1, w2, thresholds, teacher_label=teacher_label, subject_label=subject_label)
    cc.save()
    return b.getvalue()


def make_student_packets(growth_df: pd.DataFrame, w1: Window, w2: Window, thresholds: dict, teacher_label: str = "", subject_label: str = "") -> tuple[bytes, dict]:
    """
    Performance:
    - Preferred path (fast): generate each student page ONCE, then merge pages with PyPDF2 PdfMerger.
    - Fallback (compatible): if PyPDF2 isn't available, use the original ReportLab 2-pass approach.
    """
    out = growth_df.sort_values(["LastName", "FirstName", "StudentIdentifier"], na_position="last").copy()

    # Fallback if PyPDF2 not installed in the environment
    if PdfMerger is None:
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        for _, row in out.iterrows():
            draw_student_one_pager(c, row, w1, w2, thresholds, teacher_label=teacher_label, subject_label=subject_label)
            c.showPage()
        c.save()
        combined_bytes = buf.getvalue()

        individual = {}
        for _, row in out.iterrows():
            page_bytes = make_single_student_pdf(row, w1, w2, thresholds, teacher_label=teacher_label, subject_label=subject_label)
            last = str(row.get("LastName", "") or "Last").strip()
            first = str(row.get("FirstName", "") or "First").strip()
            sid = str(row.get("StudentIdentifier", "") or "ID").strip()
            fname = f"{last}_{first}_{sid}.pdf".replace(" ", "_")
            individual[fname] = page_bytes

        return combined_bytes, individual

    # Fast path
    individual = {}
    merger = PdfMerger()

    for _, row in out.iterrows():
        page_bytes = make_single_student_pdf(row, w1, w2, thresholds, teacher_label=teacher_label, subject_label=subject_label)

        last = str(row.get("LastName", "") or "Last").strip()
        first = str(row.get("FirstName", "") or "First").strip()
        sid = str(row.get("StudentIdentifier", "") or "ID").strip()
        fname = f"{last}_{first}_{sid}.pdf".replace(" ", "_")

        individual[fname] = page_bytes
        merger.append(io.BytesIO(page_bytes))

    combined_buf = io.BytesIO()
    merger.write(combined_buf)
    merger.close()

    return combined_buf.getvalue(), individual

def make_summary_pdf(growth_df: pd.DataFrame, assessment_name: str, w1: Window, w2: Window, teacher_label: str = "", subject_label: str = "") -> bytes:
    df = growth_df.copy()
    _init_pdf_fonts()
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    left = 0.75 * inch
    header_bottom = _draw_report_header(
        c,
        title="Teacher/Class Summary",
        right_title="",
        right_sub=f"Generated: {datetime.now().strftime('%b %d, %Y')}",
    )
    top = header_bottom - 0.55 * inch

    # Page 1 (Overview)
    c.setFont(PDF_FONT_REG, 11)

    margin = left
    right = width - margin
    usable_w = right - left

    thresholds = GROWTH_THRESHOLDS_DEFAULT

    # --- Meta card (teacher / assessment / windows) ---
    meta_h = 1.45 * inch
    meta_y_top = top
    meta_y = meta_y_top - meta_h
    _draw_card(c, left, meta_y, usable_w, meta_h, title=None, fill=colors.whitesmoke)

    c.setFillColor(colors.grey)
    c.setFont(PDF_FONT_BOLD, 9)
    c.drawString(left + 14, meta_y_top - 18, "CLASS")
    c.setFillColor(colors.black)
    c.setFont(PDF_FONT_BOLD, 16)
    if teacher_label:
        c.drawString(left + 14, meta_y_top - 40, f"{teacher_label}  •  {subject_label}")
    else:
        c.drawString(left + 14, meta_y_top - 40, "Teacher/Class Summary")

    c.setFont(PDF_FONT_REG, 10.5)
    y_line = meta_y_top - 52
    # Assessment (wrap if needed)
    assess_lines = _wrap_text(c, f"Assessment: {assessment_name}", usable_w - 28, PDF_FONT_REG, 10.5)
    for i, line in enumerate(assess_lines[:2]):
        c.drawString(left + 14, y_line - i * 12, line)
    y_line -= 14 + (12 * (min(2, len(assess_lines)) - 1))
    c.setFillColor(colors.grey)
    c.setFont(PDF_FONT_REG, 9.2)
    c.drawString(left + 14, y_line, f"Baseline window: {w1.start.strftime('%b %d, %Y')} – {w1.end.strftime('%b %d, %Y')}")
    y_line -= 11
    c.drawString(left + 14, y_line, f"Follow-up window: {w2.start.strftime('%b %d, %Y')} – {w2.end.strftime('%b %d, %Y')}")
    y_line -= 11
    c.drawString(left + 14, y_line, "Rule: uses the latest attempt in each window")
    c.setFillColor(colors.black)

    # --- Counts ---
    total = len(df)
    baseline_present = df["BaselineScore"].notna()
    follow_present = df["FollowupScore"].notna()
    both_mask = baseline_present & follow_present
    both = int(both_mask.sum())
    baseline_only = int((baseline_present & ~follow_present).sum())
    follow_only = int((~baseline_present & follow_present).sum())

    # Growth stats (only students with both attempts)
    g = df.loc[both_mask, "Growth"].dropna()
    mean_g = float(g.mean()) if not g.empty else float("nan")
    median_g = float(g.median()) if not g.empty else float("nan")

    improved_pct = float((g > 0).mean() * 100) if not g.empty else 0.0
    decline_pct = float((g < 0).mean() * 100) if not g.empty else 0.0
    nochange_pct = float((g == 0).mean() * 100) if not g.empty else 0.0

    # Score range overlap (only where bands exist)
    bands_ok = both_mask & df["BaselineBandMin"].notna() & df["BaselineBandMax"].notna() & df["FollowupBandMin"].notna() & df["FollowupBandMax"].notna()
    if bands_ok.any():
        bmin = df.loc[bands_ok, "BaselineBandMin"].astype(float)
        bmax = df.loc[bands_ok, "BaselineBandMax"].astype(float)
        fmin = df.loc[bands_ok, "FollowupBandMin"].astype(float)
        fmax = df.loc[bands_ok, "FollowupBandMax"].astype(float)
        overlap = ~((bmax < fmin) | (fmax < bmin))
        overlap_pct = float(overlap.mean() * 100)
        overlap_n = int(overlap.size)
    else:
        overlap_pct = 0.0
        overlap_n = 0

    # --- KPI cards (2 rows x 4) ---
    gap = 0.18 * inch
    kpi_h = 0.90 * inch
    kpi_w = (usable_w - 3 * gap) / 4

    def _kpi(x, y, label, value, sub=None):
        _draw_card(c, x, y, kpi_w, kpi_h, title=None, fill=colors.whitesmoke)
        c.setFillColor(colors.grey)
        c.setFont(PDF_FONT_BOLD, 8.6)
        c.drawString(x + 12, y + kpi_h - 18, str(label).upper())
        c.setFillColor(colors.black)
        c.setFont(PDF_FONT_BOLD, 20)
        c.drawString(x + 12, y + 22, str(value))
        if sub:
            c.setFillColor(colors.grey)
            c.setFont(PDF_FONT_REG, 8.5)
            c.drawString(x + 12, y + 10, str(sub))
            c.setFillColor(colors.black)

    y_kpi_top = meta_y - 0.18 * inch
    y_row1 = y_kpi_top - kpi_h
    x = left
    _kpi(x, y_row1, "Students included", total, "baseline or follow-up")
    x += kpi_w + gap
    _kpi(x, y_row1, "Both attempts", both, "growth computed")
    x += kpi_w + gap
    _kpi(x, y_row1, "Baseline only", baseline_only)
    x += kpi_w + gap
    _kpi(x, y_row1, "Follow-up only", follow_only)

    y_row2 = y_row1 - gap - kpi_h
    x = left
    _kpi(x, y_row2, "Mean growth", "—" if g.empty else f"{mean_g:.1f}")
    x += kpi_w + gap
    _kpi(x, y_row2, "Median growth", "—" if g.empty else f"{median_g:.1f}")
    x += kpi_w + gap
    _kpi(x, y_row2, "% improved", "—" if g.empty else f"{improved_pct:.0f}%")
    x += kpi_w + gap
    _kpi(x, y_row2, "% declined", "—" if g.empty else f"{decline_pct:.0f}%")

    # small note under KPIs
    note_y = y_row2 - 0.16 * inch
    c.setFillColor(colors.grey)
    c.setFont(PDF_FONT_REG, 8.6)
    extra = f"% no change: {nochange_pct:.0f}%" if not g.empty else "% no change: —"
    if overlap_n:
        extra += f"   •   Score ranges overlap: {overlap_pct:.0f}% (n={overlap_n})"
    c.drawString(left + 2, note_y, extra)
    c.setFillColor(colors.black)

    # --- Growth distribution chart card ---
    dist_h = 1.75 * inch
    dist_y_top = note_y - 0.20 * inch
    dist_y = dist_y_top - dist_h
    _draw_card(c, left, dist_y, usable_w, dist_h, title="Growth Distribution", title_size=12, fill=colors.whitesmoke)

    if g.empty:
        c.setFont(PDF_FONT_REG, 10)
        c.setFillColor(colors.grey)
        c.drawString(left + 14, dist_y + dist_h/2, "No growth computed (missing baseline or follow-up scores).")
        c.setFillColor(colors.black)
    else:
        buckets = [
            ("≤ -20", int((g <= -20).sum()), colors.red),
            ("-19 to -1", int(((g >= -19) & (g <= -1)).sum()), colors.red),
            ("0", int((g == 0).sum()), colors.grey),
            ("1 to 19", int(((g >= 1) & (g <= 19)).sum()), colors.gold),
            (f"≥ {thresholds['green_min']}", int((g >= thresholds["green_min"]).sum()), colors.green),
        ]
        max_cnt = max(cnt for _, cnt, _ in buckets) if buckets else 1
        x_label = left + 14
        x_bar = left + 150
        x_bar_max = left + usable_w - 18
        bar_max_w = x_bar_max - x_bar

        y0 = dist_y + dist_h - 42
        row_h = 18
        bar_h = 12

        c.setFont(PDF_FONT_REG, 9.2)
        for i, (lab, cnt, col) in enumerate(buckets):
            yy = y0 - i * row_h
            pct = (cnt / max(1, int(g.size))) * 100
            c.setFillColor(colors.black)
            c.drawString(x_label, yy, lab)
            wbar = (cnt / max_cnt) * bar_max_w if max_cnt else 0
            # Bar
            c.setFillColor(col)
            bar_y = yy - (bar_h / 2) - 1
            c.rect(x_bar, bar_y, wbar, bar_h, stroke=0, fill=1)

            # Label: draw inside the bar (black) when the bar reaches the label area
            label = f"{cnt} ({pct:.0f}%)"
            lbl_w = c.stringWidth(label, PDF_FONT_REG, 9.2)
            inside = (wbar >= (bar_max_w - (lbl_w + 10)))
            if inside:
                c.setFillColor(colors.black)
                c.drawRightString(x_bar + wbar - 6, yy - 2, label)
            else:
                c.setFillColor(colors.grey)
                c.drawRightString(x_bar_max, yy, label)

        c.setFillColor(colors.black)

    # --- Top movers (2 cards) ---
    movers_h = 1.75 * inch
    movers_y_top = dist_y - 0.20 * inch
    movers_y = movers_y_top - movers_h
    col_gap = 0.18 * inch
    card_w = (usable_w - col_gap) / 2

    _draw_card(c, left, movers_y, card_w, movers_h, title="Top Gains", title_size=12, fill=colors.whitesmoke)
    _draw_card(c, left + card_w + col_gap, movers_y, card_w, movers_h, title="Largest Drops", title_size=12, fill=colors.whitesmoke)

    df_both = df.loc[both_mask].copy()
    if df_both.empty:
        c.setFont(PDF_FONT_REG, 10)
        c.setFillColor(colors.grey)
        c.drawString(left + 14, movers_y + movers_h/2, "No students with both attempts.")
        c.setFillColor(colors.black)
    else:
        df_both = df_both.sort_values("Growth", ascending=False)
        top_gains = df_both.head(4)
        top_drops = df_both.sort_values("Growth", ascending=True).head(4)

        def _name(r):
            ln = str(r.get("LastName", "") or r.get("StudentLastName", "") or "").strip()
            fn = str(r.get("FirstName", "") or r.get("StudentFirstName", "") or "").strip()
            if fn and ln:
                return f"{fn} {ln}"
            return ln or fn

        def _truncate_text(txt: str, max_w: float, font_name: str, font_size: float) -> str:
            s = str(txt or "")
            if c.stringWidth(s, font_name, font_size) <= max_w:
                return s
            ell = "…"
            # Trim until it fits
            lo, hi = 0, len(s)
            # Simple backwards trim
            while hi > 0 and c.stringWidth(s[:hi] + ell, font_name, font_size) > max_w:
                hi -= 1
            return (s[:hi] + ell) if hi > 0 else ell

        def _draw_list(x0, rows):
            yy = movers_y + movers_h - 28
            for _, r in rows.iterrows():
                nm = _name(r)
                gr = r.get("Growth", 0)
                try:
                    gr_i = int(round(float(gr)))
                except Exception:
                    gr_i = 0
                bsc = r.get("BaselineScore", None)
                fsc = r.get("FollowupScore", None)
                try:
                    bsc_i = "—" if pd.isna(bsc) else str(int(round(float(bsc))))
                except Exception:
                    bsc_i = "—"
                try:
                    fsc_i = "—" if pd.isna(fsc) else str(int(round(float(fsc))))
                except Exception:
                    fsc_i = "—"

                c.setFillColor(colors.black)
                c.setFont(PDF_FONT_REG, 9.2)
                nm_draw = _truncate_text(nm, card_w - 90, PDF_FONT_REG, 9.2)
                c.drawString(x0 + 14, yy, nm_draw)
                c.setFont(PDF_FONT_BOLD, 10)
                c.drawRightString(x0 + card_w - 14, yy, f"{gr_i:+d}")
                c.setFillColor(colors.grey)
                c.setFont(PDF_FONT_REG, 8.2)
                c.drawString(x0 + 14, yy - 10, f"{bsc_i} → {fsc_i}")
                c.setFillColor(colors.black)
                yy -= 28

        _draw_list(left, top_gains)
        _draw_list(left + card_w + col_gap, top_drops)

    # Legend / definitions
    c.setFillColor(colors.grey)
    c.setFont(PDF_FONT_REG, 8.2)
    legend = "Definitions: Growth = Follow-up − Baseline (latest attempt per window). Score range = reported range around the score (margin of error)."
    for i, line in enumerate(_wrap_text(c, legend, usable_w, PDF_FONT_REG, 8.2)[:2]):
        c.drawString(left, movers_y - 14 - (i * 10), line)
    c.setFillColor(colors.black)

    c.showPage()

    # Page 2 - Demographics
    header_bottom = _draw_report_header(
        c,
        title="Demographics Summary (Growth)",
        right_title="",
        right_sub=f"Generated: {datetime.now().strftime('%b %d, %Y')}",
    )
    top = header_bottom - 0.55 * inch
    c.setFont(PDF_FONT_REG, 11)
    if teacher_label:
        c.drawString(left, top - 22, f"Teacher: {teacher_label}    Subject: {subject_label}")
        y0 = top - 38
    else:
        y0 = top - 22
    c.drawString(left, y0, f"Assessment: {assessment_name}")

    y = y0 - 30
    # ELAS
    c.setFont(PDF_FONT_BOLD, 12)
    c.drawString(left, y, "EnglishLanguageAcquisitionStatus")
    c.setFont(PDF_FONT_REG, 11)
    y -= 16
    if "EnglishLanguageAcquisitionStatus" in df.columns:
        for val, sub in df.groupby("EnglishLanguageAcquisitionStatus", dropna=False):
            gg = sub["Growth"].dropna()
            v = "Blank" if pd.isna(val) else str(val)
            line = f"{v}: n={len(sub)}, both={gg.size}, mean={gg.mean():.1f}" if gg.size else f"{v}: n={len(sub)}, both=0"
            c.drawString(left, y, line)
            y -= 14
    else:
        c.drawString(left, y, "Column not present in file.")
        y -= 14

    y -= 10
    # Hispanic
    c.setFont(PDF_FONT_BOLD, 12)
    c.drawString(left, y, "HispanicOrLatinoEthnicity")
    c.setFont(PDF_FONT_REG, 11)
    y -= 16
    if "HispanicOrLatinoEthnicity" in df.columns:
        for val, sub in df.groupby("HispanicOrLatinoEthnicity", dropna=False):
            gg = sub["Growth"].dropna()
            v = "Blank" if pd.isna(val) else str(val)
            line = f"{v}: n={len(sub)}, both={gg.size}, mean={gg.mean():.1f}" if gg.size else f"{v}: n={len(sub)}, both=0"
            c.drawString(left, y, line)
            y -= 14
    else:
        c.drawString(left, y, "Column not present in file.")
        y -= 14

    y -= 10
    # Race flags
    c.setFont(PDF_FONT_BOLD, 12)
    c.drawString(left, y, "Race Flags (counts)")
    c.setFont(PDF_FONT_REG, 11)
    y -= 16
    race_flags = [
        "AmericanIndianOrAlaskaNative",
        "Asian",
        "BlackOrAfricanAmerican",
        "White",
        "NativeHawaiianOrOtherPacificIslander",
        "DemographicRaceTwoOrMoreRaces",
        "Filipino",
    ]
    present = [c0 for c0 in race_flags if c0 in df.columns]
    if present:
        for c0 in present:
            cnt = (pd.to_numeric(df[c0], errors="coerce").fillna(0) == 1).sum()
            c.drawString(left, y, f"{c0}: {cnt}")
            y -= 14
    else:
        c.drawString(left, y, "Race flag columns not present in file.")
        y -= 14

    c.showPage()
    c.save()
    return buf.getvalue()


def _safe_path_part(s: str) -> str:
    s = str(s or "").strip()
    # avoid breaking ZIP paths
    s = s.replace("/", "-").replace("\\", "-").replace(":", "-").replace("|", "-")
    s = " ".join(s.split())
    return s.replace(" ", "_")


def _section_folder_label(section_number: str, period: str | None = None, room: str | None = None) -> str:
    sec = str(section_number or "").strip() or "Unknown"
    label = f"Sec_{sec}"

    p = str(period or "").strip()
    if p and p.lower() not in ("nan", "none"):
        label = f"P{p}_{label}"

    r = str(room or "").strip()
    if r and r.lower() not in ("nan", "none"):
        label = f"{label}_{r}"

    return _safe_path_part(label)


def add_teacher_assessment_to_zip(
    z: zipfile.ZipFile,
    teacher: str,
    subject: str,
    assessment: str,
    combined_pdf: bytes,
    summary_pdf: bytes,
    individual_pdfs: dict,
    section_folder: str | None = None,
):
    parts = [teacher, subject, assessment]
    if section_folder:
        parts.append(section_folder)

    base_folder = "/".join(_safe_path_part(p) for p in parts if p is not None and str(p).strip() != "")

    z.writestr(f"{base_folder}/Student_Pages.pdf", combined_pdf)
    z.writestr(f"{base_folder}/Summary.pdf", summary_pdf)

    for fname, b in individual_pdfs.items():
        z.writestr(f"{base_folder}/Students/{_safe_path_part(fname)}", b)


# -----------------------------
# Teacher mapping
# -----------------------------
def build_student_teacher_map(roster_df: pd.DataFrame, section_df: pd.DataFrame, crosswalk_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Build a student->teacher mapping used for per-teacher exports.

    Returns unique rows with columns:
        StudentIdentifier, TeacherName, SubjectNorm, SectionNumber
        plus optional: Period, Room, CourseName (if present in SectionMap)

    Notes:
    - If roster already has StudentIdentifier, crosswalk is optional.
    - If roster only has StudentID, crosswalk is required to map to StudentIdentifier.
    - To avoid section number collisions across schools, the join prefers:
        (SchoolCode, SectionNumber) if present in BOTH roster & sectionmap and non-empty;
        otherwise (SchoolNameNorm, SectionNumber) if present and non-empty;
        otherwise falls back to SectionNumber only.
    """
    if roster_df is None or section_df is None:
        raise ValueError("Roster and SectionMap are required to build teacher grouping.")

    roster = roster_df.copy()
    section = section_df.copy()

    # Ensure roster has StudentIdentifier
    if "StudentIdentifier" not in roster.columns or roster["StudentIdentifier"].isna().all():
        if crosswalk_df is None:
            raise ValueError("Roster does not include StudentIdentifier; please upload Crosswalk CSV.")
        if "StudentID" not in roster.columns:
            raise ValueError("Roster is missing StudentID, so Crosswalk cannot be applied.")
        roster = roster.merge(crosswalk_df, on="StudentID", how="left")
        if "StudentIdentifier" not in roster.columns or roster["StudentIdentifier"].isna().all():
            raise ValueError("Crosswalk merge did not produce StudentIdentifier (check StudentID keys).")

    # Ensure normalized school names exist if SchoolName exists
    if "SchoolNameNorm" not in roster.columns and "SchoolName" in roster.columns:
        roster["SchoolNameNorm"] = roster["SchoolName"].map(_norm_school_name)
    if "SchoolNameNorm" not in section.columns and "SchoolName" in section.columns:
        section["SchoolNameNorm"] = section["SchoolName"].map(_norm_school_name)

    # Choose join key to avoid section collisions across schools
    join_keys = ["SectionNumber"]
    if "SchoolCode" in roster.columns and "SchoolCode" in section.columns:
        if roster["SchoolCode"].fillna("").astype(str).str.strip().ne("").any() and section["SchoolCode"].fillna("").astype(str).str.strip().ne("").any():
            join_keys = ["SchoolCode", "SectionNumber"]
    if join_keys == ["SectionNumber"]:
        if "SchoolNameNorm" in roster.columns and "SchoolNameNorm" in section.columns:
            if roster["SchoolNameNorm"].fillna("").astype(str).str.strip().ne("").any() and section["SchoolNameNorm"].fillna("").astype(str).str.strip().ne("").any():
                join_keys = ["SchoolNameNorm", "SectionNumber"]

    # Ensure required fields exist in sectionmap
    if "TeacherName" not in section.columns:
        raise ValueError("SectionMap is missing TeacherName.")
    if "SubjectNorm" not in section.columns:
        raise ValueError("SectionMap is missing Subject (normalized). Make sure Subject exists in SectionMap CSV.")

    # Include optional fields for section folder labeling
    sec_cols = join_keys + ["TeacherName", "SubjectNorm"]
    for opt in ("Period", "Room", "CourseName"):
        if opt in section.columns and opt not in sec_cols:
            sec_cols.append(opt)

    merged = roster.merge(section[sec_cols], on=join_keys, how="left")

    out_cols = ["StudentIdentifier", "TeacherName", "SubjectNorm", "SectionNumber"]
    for opt in ("Period", "Room", "CourseName"):
        if opt in merged.columns and opt not in out_cols:
            out_cols.append(opt)

    out = (
        merged[out_cols]
        .dropna(subset=["StudentIdentifier", "TeacherName", "SubjectNorm", "SectionNumber"])
        .drop_duplicates()
    )

    out["StudentIdentifier"] = out["StudentIdentifier"].astype(str)
    out["TeacherName"] = out["TeacherName"].astype(str)
    out["SubjectNorm"] = out["SubjectNorm"].astype(str)
    out["SectionNumber"] = out["SectionNumber"].astype(str)

    return out

def filter_teacher_map_by_subject(teacher_map: pd.DataFrame, subject_choice: str) -> pd.DataFrame:
    # Include "Both" for elementary homeroom/core sections
    if str(subject_choice).upper() == "MATH":
        return teacher_map[teacher_map["SubjectNorm"].str.upper().isin(["MATH", "BOTH"])].copy()
    return teacher_map[teacher_map["SubjectNorm"].str.upper().isin(["ELA", "BOTH"])].copy()


# -----------------------------
# Exclusions / Data quality report
# -----------------------------
def build_exclusion_report(
    results_assess: pd.DataFrame,
    roster_df: pd.DataFrame,
    section_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame | None,
    subject_choice: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Builds a student-level report of who is excluded from outputs due to missing links:
      Results -> (Crosswalk) -> Roster -> SectionMap (Teacher/Subject)

    Returns:
      exclusions_df: one row per student with a primary_reason + helpful details
      summary: dict of counts
    """
    if results_assess.empty:
        return pd.DataFrame(), {"results_students": 0, "included": 0, "excluded": 0}

    # Unique students from results for this assessment
    base_cols = ["StudentIdentifier", "FirstName", "LastName", "SchoolName", "GradeLevelWhenAssessed"]
    base_cols = [c for c in base_cols if c in results_assess.columns]
    res_students = results_assess[base_cols].drop_duplicates().copy()
    res_students["StudentIdentifier"] = res_students["StudentIdentifier"].astype(str)

    # Crosswalk (optional)
    if crosswalk_df is not None and not crosswalk_df.empty:
        xw = crosswalk_df[["StudentIdentifier", "StudentID"]].drop_duplicates().copy()
        xw["StudentIdentifier"] = xw["StudentIdentifier"].astype(str)
        xw["StudentID"] = xw["StudentID"].astype(str)
    else:
        xw = pd.DataFrame(columns=["StudentIdentifier", "StudentID"])

    # Roster normalized columns (should already be normalized)
    roster = roster_df[["StudentIdentifier", "StudentID", "SectionNumber"]].copy()
    roster["StudentIdentifier"] = roster["StudentIdentifier"].astype(str)
    roster["StudentID"] = roster["StudentID"].astype(str)
    roster["SectionNumber"] = roster["SectionNumber"].astype(str)

    # Map results -> roster via StudentIdentifier (direct)
    m_sid = res_students.merge(roster, on="StudentIdentifier", how="left")

    # Map results -> roster via crosswalk StudentID (fallback)
    if not xw.empty:
        m_xw = res_students.merge(xw, on="StudentIdentifier", how="left", suffixes=("", "_xw"))
        m_xw = m_xw.merge(roster.drop(columns=["StudentIdentifier"]), on="StudentID", how="left")
    else:
        m_xw = res_students.copy()
        m_xw["StudentID"] = pd.NA
        m_xw["SectionNumber"] = pd.NA

    # Combine mapping rows (students can have multiple sections)
    m = pd.concat(
        [
            m_sid[res_students.columns.tolist() + ["StudentID", "SectionNumber"]],
            m_xw[res_students.columns.tolist() + ["StudentID", "SectionNumber"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    # Join to SectionMap
    sec = section_df[["SectionNumber", "TeacherName", "SubjectNorm"]].copy()
    sec["SectionNumber"] = sec["SectionNumber"].astype(str)

    m2 = m.merge(sec, on="SectionNumber", how="left")

    # Subject filter set (include BOTH)
    if str(subject_choice).upper() == "MATH":
        ok_subjects = {"MATH", "BOTH"}
    else:
        ok_subjects = {"ELA", "BOTH"}

    # Aggregate to student-level
    def _agg_student(g: pd.DataFrame) -> pd.Series:
        sid = g["StudentIdentifier"].iloc[0]
        has_xw = sid in set(xw["StudentIdentifier"].astype(str).unique()) if not xw.empty else False

        # sections from roster matches
        sections = sorted(set([s for s in g["SectionNumber"].dropna().astype(str).tolist() if s not in ("nan", "NaN")]))
        has_roster = len(sections) > 0

        # sectionmap matches
        # A section "matches" if it exists in sectionmap and has a TeacherName + SubjectNorm
        section_rows = g.dropna(subset=["SectionNumber"]).copy()
        has_any_sectionmap = section_rows["TeacherName"].notna().any() and section_rows["SubjectNorm"].notna().any()

        # subject matches
        subject_matches = section_rows["SubjectNorm"].astype(str).str.upper().isin(ok_subjects).any() if not section_rows.empty else False

        # missing sections in sectionmap
        missing_in_sectionmap = sorted(set(section_rows.loc[section_rows["TeacherName"].isna(), "SectionNumber"].astype(str).tolist()))

        # primary reason
        if not has_roster:
            if xw.empty:
                reason = "No roster match (and no crosswalk uploaded)"
            elif not has_xw:
                reason = "Missing in crosswalk"
            else:
                reason = "Crosswalk StudentID not found in roster"
        elif not has_any_sectionmap:
            reason = "SectionNumber not in SectionMap (or missing Teacher/Subject)"
        elif not subject_matches:
            reason = f"Section subject not {subject_choice} (only other subject(s) found)"
        else:
            reason = "Included"

        return pd.Series(
            {
                "PrimaryReason": reason,
                "InCrosswalk": has_xw,
                "RosterSections": ", ".join(sections[:20]) + ("…" if len(sections) > 20 else ""),
                "MissingSectionsInSectionMap": ", ".join(missing_in_sectionmap[:20]) + ("…" if len(missing_in_sectionmap) > 20 else ""),
            }
        )

    student_report = m2.groupby("StudentIdentifier", as_index=False).apply(_agg_student)

    # Included vs excluded
    exclusions = student_report[student_report["PrimaryReason"] != "Included"].copy()

    # Attach names/school/grade back (first non-null)
    meta = res_students.groupby("StudentIdentifier", as_index=False).first()
    out = meta.merge(exclusions, on="StudentIdentifier", how="inner")

    summary = {
        "results_students": int(res_students["StudentIdentifier"].nunique()),
        "included": int(student_report[student_report["PrimaryReason"] == "Included"]["StudentIdentifier"].nunique()),
        "excluded": int(exclusions["StudentIdentifier"].nunique()),
    }
    return out.sort_values(["PrimaryReason", "LastName", "FirstName"], na_position="last"), summary


# -----------------------------
# UI
# -----------------------------
st.title("IAB/FIAB Progress Reporting (v0.2)")

st.sidebar.header("Uploads")
results_file = st.sidebar.file_uploader("Results CSV (required)", type=["csv"], key="results")

multi_mode = st.sidebar.checkbox("Multi-teacher mode (use stored roster/section map)", value=True)

# Reference files (Roster / Crosswalk / SectionMap)
roster_raw = crosswalk_raw = section_raw = None
ref_status = None

if multi_mode:
    st.sidebar.subheader("Reference files")
    override_ref = st.sidebar.checkbox(
        "Override reference files (upload manually)",
        value=False,
        help="Recommended OFF. Turn ON only for troubleshooting or one-off runs."
    )

    if not override_ref:
        roster_raw, crosswalk_raw, section_raw, ref_status = _try_load_reference_raw()
        if ref_status and ref_status.get("source"):
            st.sidebar.caption(f"Loaded from: **{ref_status['source'].upper()}**")
        if ref_status:
            for k, v in ref_status.get("details", {}).items():
                st.sidebar.caption(f"{k}: {v}")

    # If override is on OR auto-load failed for required files, show upload widgets
    if override_ref or (roster_raw is None) or (section_raw is None):
        st.sidebar.caption("Upload reference files (required for multi-teacher mode)")
        roster_file = st.sidebar.file_uploader("Roster CSV (Student ↔ Section)", type=["csv"], key="roster")
        crosswalk_file = st.sidebar.file_uploader("Crosswalk CSV (StudentIdentifier ↔ StudentID)", type=["csv"], key="crosswalk")
        section_file = st.sidebar.file_uploader("SectionMap CSV (Section ↔ Teacher ↔ Subject)", type=["csv"], key="section")

        roster_raw = roster_file.getvalue() if roster_file else roster_raw
        crosswalk_raw = crosswalk_file.getvalue() if crosswalk_file else crosswalk_raw
        section_raw = section_file.getvalue() if section_file else section_raw

if not results_file:
    st.info("Upload a Results CSV to begin.")
    st.stop()

try:
    results_df = normalize_results_cached(results_file.getvalue())
except Exception as e:
    st.error(f"Could not read/normalize Results CSV: {e}")
    st.stop()

# Sidebar: growth thresholds
st.sidebar.header("Growth Highlight Thresholds")
green_min = st.sidebar.number_input("Green if growth ≥", value=GROWTH_THRESHOLDS_DEFAULT["green_min"], step=1)
yellow_min = st.sidebar.number_input("Yellow if growth ≥", value=GROWTH_THRESHOLDS_DEFAULT["yellow_min"], step=1)
red_max = st.sidebar.number_input("Red if growth ≤", value=GROWTH_THRESHOLDS_DEFAULT["red_max"], step=1)
thresholds = {"green_min": int(green_min), "yellow_min": int(yellow_min), "red_max": int(red_max)}

# Windows
w1_auto, w2_auto = detect_two_windows(results_df)
st.subheader("Detected Windows (override if needed)")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Baseline Window**")
    w1_start = st.date_input("Baseline start", value=w1_auto.start, key="w1s")
    w1_end = st.date_input("Baseline end", value=w1_auto.end, key="w1e")
with c2:
    st.markdown("**Follow-up Window**")
    w2_start = st.date_input("Follow-up start", value=w2_auto.start, key="w2s")
    w2_end = st.date_input("Follow-up end", value=w2_auto.end, key="w2e")

w1 = Window("Baseline", w1_start, w1_end)
w2 = Window("Follow-up", w2_start, w2_end)


# Assessment selector
assessments = sorted(results_df["AssessmentName"].dropna().unique().tolist())

assessment_name = st.selectbox("Assessment (preview)", assessments)
run_all_assessments = st.checkbox("Generate ALL assessments on export", value=False)

# Preview uses the selected assessment
results_assess = results_df[results_df["AssessmentName"] == assessment_name].copy()

# Teacher mapping (optional)
teacher_map = None
subject_choice = None
teacher_choice = None
all_teachers = False

if multi_mode and roster_raw and section_raw:
    try:
        roster_df = normalize_roster_cached(roster_raw)
        section_df = normalize_sectionmap_cached(section_raw)
        crosswalk_df = normalize_crosswalk_cached(crosswalk_raw) if crosswalk_raw else None
        teacher_map = build_student_teacher_map(roster_df, section_df, crosswalk_df)

        # Validation: sections in roster missing from SectionMap (school-aware)
        # Prefer SchoolCode join if present; else SchoolNameNorm; else SectionNumber only.
        use_schoolcode = (
            ("SchoolCode" in roster_df.columns) and ("SchoolCode" in section_df.columns)
            and roster_df["SchoolCode"].fillna("").astype(str).str.strip().ne("").any()
            and section_df["SchoolCode"].fillna("").astype(str).str.strip().ne("").any()
        )
        use_schoolname = (
            ("SchoolNameNorm" in roster_df.columns) and ("SchoolNameNorm" in section_df.columns)
            and roster_df["SchoolNameNorm"].fillna("").astype(str).str.strip().ne("").any()
            and section_df["SchoolNameNorm"].fillna("").astype(str).str.strip().ne("").any()
        )

        if use_schoolcode:
            roster_keys = set(zip(
                roster_df["SchoolCode"].fillna("").astype(str).str.strip(),
                roster_df["SectionNumber"].fillna("").astype(str).str.strip()
            ))
            mapped_keys = set(zip(
                section_df["SchoolCode"].fillna("").astype(str).str.strip(),
                section_df["SectionNumber"].fillna("").astype(str).str.strip()
            ))
            missing_keys = sorted(list(roster_keys - mapped_keys))
            if missing_keys:
                st.warning(
                    f"{len(missing_keys)} roster section(s) are not present in the SectionMap (by SchoolCode+SectionNumber). "
                    "Students in those sections will be excluded until SectionMap is updated."
                )
                st.dataframe(pd.DataFrame(missing_keys[:50], columns=["SchoolCode", "SectionNumber"]))
        elif use_schoolname:
            roster_keys = set(zip(
                roster_df["SchoolNameNorm"].fillna("").astype(str).str.strip(),
                roster_df["SectionNumber"].fillna("").astype(str).str.strip()
            ))
            mapped_keys = set(zip(
                section_df["SchoolNameNorm"].fillna("").astype(str).str.strip(),
                section_df["SectionNumber"].fillna("").astype(str).str.strip()
            ))
            missing_keys = sorted(list(roster_keys - mapped_keys))
            if missing_keys:
                st.warning(
                    f"{len(missing_keys)} roster section(s) are not present in the SectionMap (by SchoolName+SectionNumber). "
                    "Students in those sections will be excluded until SectionMap is updated."
                )
                # show pretty school name by joining back (optional)
                st.dataframe(pd.DataFrame(missing_keys[:50], columns=["SchoolNameNorm", "SectionNumber"]))
        else:
            roster_sections = set(roster_df["SectionNumber"].dropna().astype(str).unique().tolist()) if "SectionNumber" in roster_df.columns else set()
            mapped_sections = set(section_df["SectionNumber"].dropna().astype(str).unique().tolist()) if "SectionNumber" in section_df.columns else set()
            missing_sections = sorted(list(roster_sections - mapped_sections))
            if missing_sections:
                st.warning(
                    f"{len(missing_sections)} section(s) in the roster are not present in the SectionMap. "
                    "Students in those sections will not be included until SectionMap is updated."
                )
                st.dataframe(pd.DataFrame({'Missing SectionNumber': missing_sections[:50]}))


        st.sidebar.header("Grouping")
        subject_choice = st.sidebar.selectbox("Subject", ["Math", "ELA"], index=0)

        # Filter map by subject
        teacher_map_sub = filter_teacher_map_by_subject(teacher_map, subject_choice)

        teachers = sorted(teacher_map_sub["TeacherName"].unique().tolist())
        if not teachers:
            st.warning("No teachers found for the selected subject (check SectionMap Subject column).")
        teacher_choice = st.sidebar.selectbox("Teacher", ["(All teachers)"] + teachers, index=0)

        all_teachers = (teacher_choice == "(All teachers)")


        # -----------------------------
        # Exclusions report (students excluded due to missing mappings)
        # -----------------------------
        with st.expander("Data Exclusions (students missing crosswalk/roster/section map)", expanded=False):
            try:
                excl_df, excl_summary = build_exclusion_report(
                    results_assess,
                    roster_df=roster_df,
                    section_df=section_df,
                    crosswalk_df=crosswalk_df,
                    subject_choice=subject_choice,
                )
                st.write(
                    f"Students in results for this assessment: **{excl_summary['results_students']}**  \n"
                    f"Included in outputs (for {subject_choice}): **{excl_summary['included']}**  \n"
                    f"Excluded due to missing links: **{excl_summary['excluded']}**"
                )

                if excl_df.empty:
                    st.success("No excluded students detected for this assessment/subject.")
                else:
                    st.caption("Tip: download this CSV and use it as your to-do list for filling Crosswalk/Roster/SectionMap.")
                    st.dataframe(excl_df, use_container_width=True)
                    st.download_button(
                        "Download exclusions CSV",
                        data=excl_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"excluded_students_{subject_choice}_{assessment_name}.csv".replace(" ", "_"),
                        mime="text/csv",
                    )
            except Exception as ex:
                st.warning(f"Could not generate exclusions report: {ex}")


        # Warnings about missing section mappings
        missing_teacher_rows = teacher_map[teacher_map["TeacherName"].isna()].shape[0] if teacher_map is not None else 0

    except Exception as e:
        st.error(f"Could not build teacher grouping from roster/section/crosswalk: {e}")
        teacher_map = None

# Build preview dataframe (depends on mode)
if teacher_map is None or (not multi_mode) or (not roster_raw) or (not section_raw):
    # single-file mode
    growth_df = build_growth_table(results_assess, w1, w2)
    st.caption("Single-file mode: generating one packet from the uploaded Results CSV.")
else:
    teacher_map_sub = filter_teacher_map_by_subject(teacher_map, subject_choice)
    if all_teachers:
        # preview shows combined across selected subject teachers (dedupe within teacher not needed for preview)
        student_ids = teacher_map_sub["StudentIdentifier"].unique().tolist()
        growth_df = build_growth_table(results_assess[results_assess["StudentIdentifier"].isin(student_ids)], w1, w2)
        st.caption(f"Multi-teacher mode: previewing ALL teachers for {subject_choice}. Export will create a folder per teacher.")
    else:
        teacher_students = teacher_map_sub[teacher_map_sub["TeacherName"] == teacher_choice]["StudentIdentifier"].unique().tolist()
        growth_df = build_growth_table(results_assess[results_assess["StudentIdentifier"].isin(teacher_students)], w1, w2)
        st.caption(f"Multi-teacher mode: previewing {teacher_choice} ({subject_choice}).")

# Preview table
st.subheader("Preview (one row per student for this assessment)")
preview_cols = [
    "LastName", "FirstName", "StudentIdentifier",
    "BaselineDate", "BaselineScore",
    "FollowupDate", "FollowupScore",
    "Growth",
    "BaselineCategory", "FollowupCategory",
]
preview_cols = [c for c in preview_cols if c in growth_df.columns]
st.dataframe(growth_df[preview_cols].sort_values(["LastName", "FirstName"], na_position="last"), use_container_width=True)

# Export
st.subheader("Export")

split_by_section = st.checkbox(
    "Split exports by section (adds folders inside each assessment)",
    value=True,
    help="Creates: Teacher/Subject/Assessment/P#_Sec_####_<Room>/ ...",
)

if st.button("Generate ZIP (student one-pagers + summary)"):
    assessments_to_run = assessments if run_all_assessments else [assessment_name]

    # Determine whether multi-teacher grouping is ready
    multi_ready = (
        multi_mode
        and (teacher_map is not None)
        and (roster_raw is not None)
        and (section_raw is not None)
        and (subject_choice is not None)
    )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:

        if not multi_ready:
            # Single-file mode: use sidebar labels for folder naming
            teacher_label = st.sidebar.text_input("TeacherName (folder label)", value="Teacher", key="teacherlabel")
            subject_label = st.sidebar.selectbox("Subject label", ["Math", "ELA"], index=0, key="subjectlabel")

            prog = st.progress(0.0, text="Generating assessment packets…")
            for i, a in enumerate(assessments_to_run):
                a_df = results_df[results_df["AssessmentName"] == a].copy()
                if a_df.empty:
                    prog.progress((i + 1) / max(len(assessments_to_run), 1), text=f"Generating assessment packets… ({i+1}/{len(assessments_to_run)})")
                    continue

                a_growth = build_growth_table(a_df, w1, w2)
                if a_growth.empty:
                    prog.progress((i + 1) / max(len(assessments_to_run), 1), text=f"Generating assessment packets… ({i+1}/{len(assessments_to_run)})")
                    continue

                combined_pdf, individual = make_student_packets(
                    a_growth, w1, w2, thresholds, teacher_label=teacher_label, subject_label=subject_label
                )
                summary_pdf = make_summary_pdf(
                    a_growth, a, w1, w2, teacher_label=teacher_label, subject_label=subject_label
                )

                # (single-file mode has no roster/sections, so keep current structure)
                add_teacher_assessment_to_zip(z, teacher_label, subject_label, a, combined_pdf, summary_pdf, individual)

                prog.progress((i + 1) / max(len(assessments_to_run), 1), text=f"Generating assessment packets… ({i+1}/{len(assessments_to_run)})")

        else:
            # Multi-teacher mode:
            teacher_map_sub = filter_teacher_map_by_subject(teacher_map, subject_choice)

            join_cols = [c for c in ["StudentIdentifier", "TeacherName", "SectionNumber", "Period", "Room", "CourseName"] if c in teacher_map_sub.columns]
            teacher_map_join = teacher_map_sub[join_cols].drop_duplicates()

            # Helper: compute total tasks for progress bar
            total_tasks = 0
            for a in assessments_to_run:
                a_ids = results_df.loc[results_df["AssessmentName"] == a, ["StudentIdentifier"]].drop_duplicates()
                if a_ids.empty:
                    continue

                present = a_ids.merge(teacher_map_join, on="StudentIdentifier", how="inner")
                if present.empty:
                    continue

                if split_by_section:
                    present_groups = present[["TeacherName", "SectionNumber"]].drop_duplicates().shape[0]
                else:
                    present_groups = present["TeacherName"].nunique()

                total_tasks += int(present_groups)

            total_tasks = max(total_tasks, 1)
            done = 0
            prog = st.progress(0.0, text="Generating teacher packets…")

            drop_cols_base = [c for c in ["TeacherName", "SectionNumber", "Period", "Room", "CourseName"] if c in teacher_map_join.columns]

            if all_teachers:
                for a in assessments_to_run:
                    a_df = results_df[results_df["AssessmentName"] == a].copy()
                    if a_df.empty:
                        continue

                    a_growth = build_growth_table(a_df, w1, w2)
                    if a_growth.empty:
                        continue

                    assigned = a_growth.merge(teacher_map_join, on="StudentIdentifier", how="inner")
                    if assigned.empty:
                        continue

                    if split_by_section:
                        # Teacher + Section folders
                        for (t, sec), gdf in assigned.groupby(["TeacherName", "SectionNumber"], sort=True):
                            period = None
                            room = None
                            if "Period" in gdf.columns and gdf["Period"].notna().any():
                                period = gdf["Period"].dropna().astype(str).iloc[0]
                            if "Room" in gdf.columns and gdf["Room"].notna().any():
                                room = gdf["Room"].dropna().astype(str).iloc[0]

                            sec_folder = _section_folder_label(sec, period=period, room=room)

                            t_growth = (
                                gdf.drop(columns=[c for c in drop_cols_base if c in gdf.columns])
                                .drop_duplicates(subset=["StudentIdentifier", "AssessmentName"], keep="first")
                                .copy()
                            )
                            if t_growth.empty:
                                continue

                            combined_pdf, individual = make_student_packets(
                                t_growth, w1, w2, thresholds, teacher_label=t, subject_label=subject_choice
                            )
                            summary_pdf = make_summary_pdf(
                                t_growth, a, w1, w2, teacher_label=t, subject_label=subject_choice
                            )
                            add_teacher_assessment_to_zip(
                                z, t, subject_choice, a, combined_pdf, summary_pdf, individual, section_folder=sec_folder
                            )

                            done += 1
                            prog.progress(done / total_tasks, text=f"Generating teacher packets… ({done}/{total_tasks})")

                    else:
                        # Original teacher-only folders
                        for t, t_df in assigned.groupby("TeacherName", sort=True):
                            t_growth = (
                                t_df.drop(columns=[c for c in drop_cols_base if c in t_df.columns])
                                .drop_duplicates(subset=["StudentIdentifier", "AssessmentName"], keep="first")
                                .copy()
                            )
                            if t_growth.empty:
                                continue

                            combined_pdf, individual = make_student_packets(
                                t_growth, w1, w2, thresholds, teacher_label=t, subject_label=subject_choice
                            )
                            summary_pdf = make_summary_pdf(
                                t_growth, a, w1, w2, teacher_label=t, subject_label=subject_choice
                            )
                            add_teacher_assessment_to_zip(z, t, subject_choice, a, combined_pdf, summary_pdf, individual)

                            done += 1
                            prog.progress(done / total_tasks, text=f"Generating teacher packets… ({done}/{total_tasks})")

            else:
                # Single teacher export
                tm_t = teacher_map_join[teacher_map_join["TeacherName"] == teacher_choice].copy()

                for a in assessments_to_run:
                    a_df = results_df[results_df["AssessmentName"] == a].copy()
                    if a_df.empty:
                        continue

                    a_growth = build_growth_table(a_df, w1, w2)

                    assigned = a_growth.merge(tm_t, on="StudentIdentifier", how="inner")
                    if assigned.empty:
                        continue

                    if split_by_section:
                        for sec, gdf in assigned.groupby("SectionNumber", sort=True):
                            period = None
                            room = None
                            if "Period" in gdf.columns and gdf["Period"].notna().any():
                                period = gdf["Period"].dropna().astype(str).iloc[0]
                            if "Room" in gdf.columns and gdf["Room"].notna().any():
                                room = gdf["Room"].dropna().astype(str).iloc[0]

                            sec_folder = _section_folder_label(sec, period=period, room=room)

                            t_growth = (
                                gdf.drop(columns=[c for c in drop_cols_base if c in gdf.columns])
                                .drop_duplicates(subset=["StudentIdentifier", "AssessmentName"], keep="first")
                                .copy()
                            )
                            if t_growth.empty:
                                continue

                            combined_pdf, individual = make_student_packets(
                                t_growth, w1, w2, thresholds, teacher_label=teacher_choice, subject_label=subject_choice
                            )
                            summary_pdf = make_summary_pdf(
                                t_growth, a, w1, w2, teacher_label=teacher_choice, subject_label=subject_choice
                            )
                            add_teacher_assessment_to_zip(
                                z, teacher_choice, subject_choice, a, combined_pdf, summary_pdf, individual, section_folder=sec_folder
                            )

                            done += 1
                            prog.progress(done / total_tasks, text=f"Generating teacher packets… ({done}/{total_tasks})")

                    else:
                        t_growth = (
                            assigned.drop(columns=[c for c in drop_cols_base if c in assigned.columns])
                            .drop_duplicates(subset=["StudentIdentifier", "AssessmentName"], keep="first")
                            .copy()
                        )
                        if t_growth.empty:
                            continue

                        combined_pdf, individual = make_student_packets(
                            t_growth, w1, w2, thresholds, teacher_label=teacher_choice, subject_label=subject_choice
                        )
                        summary_pdf = make_summary_pdf(
                            t_growth, a, w1, w2, teacher_label=teacher_choice, subject_label=subject_choice
                        )
                        add_teacher_assessment_to_zip(z, teacher_choice, subject_choice, a, combined_pdf, summary_pdf, individual)

                        done += 1
                        prog.progress(done / total_tasks, text=f"Generating teacher packets… ({done}/{total_tasks})")

    st.success("ZIP created.")
    st.download_button(
        "Download ZIP",
        data=zip_buf.getvalue(),
        file_name=f"IAB_FIAB_{('ALL' if run_all_assessments else assessment_name)}.zip".replace(" ", "_"),
        mime="application/zip",
    )
