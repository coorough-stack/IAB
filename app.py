import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from calendar import monthrange

import pandas as pd
import streamlit as st
from dateutil import parser as dtparser
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
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

    df["StudentIdentifier"] = df["StudentIdentifier"].astype(str)
    df["StudentID"] = df["StudentID"].astype(str)
    df["SectionNumber"] = df["SectionNumber"].astype(str)

    return df[need]


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
    width, height = letter
    left = 0.75 * inch
    top = height - 0.75 * inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, top, "IAB/FIAB Progress Report")

    c.setFont("Helvetica", 11)
    c.drawString(left, top - 22, f"Student: {row.get('LastName','')}, {row.get('FirstName','')}")
    c.drawString(left, top - 38, f"StudentIdentifier: {row.get('StudentIdentifier','')}")
    if teacher_label:
        c.drawString(left, top - 54, f"Teacher: {teacher_label}    Subject: {subject_label}")
        y_shift = 0
    else:
        y_shift = -16

    c.drawString(left, top - 54 + y_shift, f"Assessment: {row.get('AssessmentName','')}")
    c.drawString(left, top - 70 + y_shift, f"SchoolYear: {row.get('SchoolYear','')}    Grade: {row.get('GradeLevelWhenAssessed','')}")

    # Baseline
    y = top - 110 + y_shift
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, f"Baseline Window ({w1.start.strftime('%b %d, %Y')} – {w1.end.strftime('%b %d, %Y')})")
    c.setFont("Helvetica", 11)
    c.drawString(left, y - 18, f"Date: {fmt_date(row.get('BaselineDate', pd.NaT))}")
    c.drawString(left, y - 34, f"Score: {row.get('BaselineScore','N/A')}")
    c.drawString(left, y - 50, f"Error Band: {row.get('BaselineBandMin','')} – {row.get('BaselineBandMax','')}")
    c.drawString(left, y - 66, f"Reporting Category: {row.get('BaselineCategory','')}")
    c.drawString(left, y - 82, f"Status: {row.get('BaselineStatus','')}")

    # Follow-up
    y2 = y - 125
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y2, f"Follow-up Window ({w2.start.strftime('%b %d, %Y')} – {w2.end.strftime('%b %d, %Y')})")
    c.setFont("Helvetica", 11)
    c.drawString(left, y2 - 18, f"Date: {fmt_date(row.get('FollowupDate', pd.NaT))}")
    c.drawString(left, y2 - 34, f"Score: {row.get('FollowupScore','N/A')}")
    c.drawString(left, y2 - 50, f"Error Band: {row.get('FollowupBandMin','')} – {row.get('FollowupBandMax','')}")
    c.drawString(left, y2 - 66, f"Reporting Category: {row.get('FollowupCategory','')}")
    c.drawString(left, y2 - 82, f"Status: {row.get('FollowupStatus','')}")

    # Growth
    y3 = y2 - 125
    g = row.get("Growth", None)
    gtxt = "N/A" if pd.isna(g) else f"{int(g):+d}"
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(growth_color(g, thresholds))
    c.drawString(left, y3, f"Growth (Follow-up − Baseline): {gtxt}")
    c.setFillColor(colors.black)

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(left, 0.6 * inch, "Note: Growth compares the same assessment taken in two windows during the school year.")


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
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    left = 0.75 * inch
    top = height - 0.75 * inch

    # Page 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, top, "Teacher/Class Summary")
    c.setFont("Helvetica", 11)
    if teacher_label:
        c.drawString(left, top - 22, f"Teacher: {teacher_label}    Subject: {subject_label}")
        y0 = top - 38
    else:
        y0 = top - 22

    c.drawString(left, y0, f"Assessment: {assessment_name}")
    c.drawString(left, y0 - 16, f"Baseline: {w1.start.strftime('%b %d, %Y')} – {w1.end.strftime('%b %d, %Y')}")
    c.drawString(left, y0 - 32, f"Follow-up: {w2.start.strftime('%b %d, %Y')} – {w2.end.strftime('%b %d, %Y')}")

    total = len(df)
    both = df["Growth"].notna().sum()
    baseline_only = df["BaselineScore"].notna().sum() - both
    follow_only = df["FollowupScore"].notna().sum() - both

    c.drawString(left, y0 - 60, f"Students in report: {total}")
    c.drawString(left, y0 - 76, f"With both attempts: {both}")
    c.drawString(left, y0 - 92, f"Baseline only: {baseline_only}    Follow-up only: {follow_only}")

    g = df["Growth"].dropna()
    y = y0 - 126
    if not g.empty:
        c.drawString(left, y, f"Mean growth: {g.mean():.1f}    Median growth: {g.median():.1f}")
        y -= 20
        buckets = [
            ("<= -20", (g <= -20).sum()),
            ("-19 to -1", ((g >= -19) & (g <= -1)).sum()),
            ("0", (g == 0).sum()),
            ("1 to 19", ((g >= 1) & (g <= 19)).sum()),
            (">= 20", (g >= 20).sum()),
        ]
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Growth Distribution")
        c.setFont("Helvetica", 11)
        y -= 18
        for label, cnt in buckets:
            c.drawString(left, y, f"{label}: {cnt}")
            y -= 16
    else:
        c.drawString(left, y, "No growth computed (missing baseline or follow-up scores).")

    c.showPage()

    # Page 2 - Demographics
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, top, "Demographics Summary (Growth)")
    c.setFont("Helvetica", 11)
    if teacher_label:
        c.drawString(left, top - 22, f"Teacher: {teacher_label}    Subject: {subject_label}")
        y0 = top - 38
    else:
        y0 = top - 22
    c.drawString(left, y0, f"Assessment: {assessment_name}")

    y = y0 - 30
    # ELAS
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "EnglishLanguageAcquisitionStatus")
    c.setFont("Helvetica", 11)
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
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "HispanicOrLatinoEthnicity")
    c.setFont("Helvetica", 11)
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
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Race Flags (counts)")
    c.setFont("Helvetica", 11)
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


def add_teacher_assessment_to_zip(z: zipfile.ZipFile, teacher: str, subject: str, assessment: str, combined_pdf: bytes, summary_pdf: bytes, individual_pdfs: dict):
    base_folder = f"{teacher}/{subject}/{assessment}".replace(" ", "_")
    z.writestr(f"{base_folder}/Student_Pages.pdf", combined_pdf)
    z.writestr(f"{base_folder}/Summary.pdf", summary_pdf)
    for fname, b in individual_pdfs.items():
        z.writestr(f"{base_folder}/Students/{fname}", b)


# -----------------------------
# Teacher mapping
# -----------------------------
def build_student_teacher_map(roster_df: pd.DataFrame, section_df: pd.DataFrame, crosswalk_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Returns unique rows: StudentIdentifier, TeacherName, SubjectNorm

    If roster already has StudentIdentifier, crosswalk is optional.
    If roster only has StudentID, crosswalk is required to map to StudentIdentifier.
    """
    roster = roster_df.copy()
    section = section_df.copy()

    # Ensure roster has StudentIdentifier
    if "StudentIdentifier" not in roster.columns or roster["StudentIdentifier"].isna().all():
        if crosswalk_df is None:
            raise ValueError("Roster does not include StudentIdentifier; please upload Crosswalk CSV.")
        roster = roster.merge(crosswalk_df, on="StudentID", how="left")

    m = roster.merge(section[["SectionNumber", "TeacherName", "SubjectNorm"]], on="SectionNumber", how="left")
    out = m[["StudentIdentifier", "TeacherName", "SubjectNorm"]].dropna(subset=["StudentIdentifier", "TeacherName", "SubjectNorm"]).drop_duplicates()
    out["TeacherName"] = out["TeacherName"].astype(str)
    out["SubjectNorm"] = out["SubjectNorm"].astype(str)
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

multi_mode = st.sidebar.checkbox("Multi-teacher mode (upload roster + section map)", value=True)

roster_file = st.sidebar.file_uploader("Roster CSV (Student ↔ Section)", type=["csv"], key="roster") if multi_mode else None
crosswalk_file = st.sidebar.file_uploader("Crosswalk CSV (StudentIdentifier ↔ StudentID)", type=["csv"], key="crosswalk") if multi_mode else None
section_file = st.sidebar.file_uploader("SectionMap CSV (Section ↔ Teacher ↔ Subject)", type=["csv"], key="section") if multi_mode else None

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

if multi_mode and roster_file and section_file:
    try:
        roster_df = normalize_roster_cached(roster_file.getvalue())
        section_df = normalize_sectionmap_cached(section_file.getvalue())
        crosswalk_df = normalize_crosswalk_cached(crosswalk_file.getvalue()) if crosswalk_file else None
        teacher_map = build_student_teacher_map(roster_df, section_df, crosswalk_df)

        # Validation: sections in roster missing from SectionMap
        roster_sections = set(roster_df["SectionNumber"].dropna().astype(str).unique().tolist()) if "SectionNumber" in roster_df.columns else set()
        mapped_sections = set(section_df["SectionNumber"].dropna().astype(str).unique().tolist()) if "SectionNumber" in section_df.columns else set()
        missing_sections = sorted(list(roster_sections - mapped_sections))
        if missing_sections:
            st.warning(
                f"{len(missing_sections)} section(s) in the roster are not present in the SectionMap. "
                "Students in those sections will not be included until SectionMap is updated."
            )
            st.dataframe(pd.DataFrame({'Missing SectionNumber': missing_sections}))

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
if teacher_map is None or not multi_mode or not roster_file or not section_file:
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


if st.button("Generate ZIP (student one-pagers + summary)"):
    assessments_to_run = assessments if run_all_assessments else [assessment_name]

    # Determine whether multi-teacher grouping is ready
    multi_ready = (
        multi_mode
        and (teacher_map is not None)
        and (roster_file is not None)
        and (section_file is not None)
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
                add_teacher_assessment_to_zip(z, teacher_label, subject_label, a, combined_pdf, summary_pdf, individual)

                prog.progress((i + 1) / max(len(assessments_to_run), 1), text=f"Generating assessment packets… ({i+1}/{len(assessments_to_run)})")

        else:
            # Multi-teacher mode (performance-optimized):
            # For each assessment, build the growth table ONCE, then assign teachers via join.
            teacher_map_sub = filter_teacher_map_by_subject(teacher_map, subject_choice)
            teacher_map_join = teacher_map_sub[["StudentIdentifier", "TeacherName"]].drop_duplicates()

            # Helper: compute how many teacher+assessment combos actually exist (for accurate progress)
            total_tasks = 0
            for a in assessments_to_run:
                a_ids = results_df.loc[results_df["AssessmentName"] == a, ["StudentIdentifier"]].drop_duplicates()
                if a_ids.empty:
                    continue
                present_teachers = a_ids.merge(teacher_map_join, on="StudentIdentifier", how="inner")["TeacherName"].nunique()
                total_tasks += present_teachers
            total_tasks = max(total_tasks, 1)

            done = 0
            prog = st.progress(0.0, text="Generating teacher packets…")

            if all_teachers:
                for a in assessments_to_run:
                    a_df = results_df[results_df["AssessmentName"] == a].copy()
                    if a_df.empty:
                        continue

                    a_growth = build_growth_table(a_df, w1, w2)
                    if a_growth.empty:
                        continue

                    # Assign teachers (duplicates students into each teacher's report if the student maps to multiple teachers)
                    assigned = a_growth.merge(teacher_map_join, on="StudentIdentifier", how="inner")

                    for t, t_df in assigned.groupby("TeacherName", sort=True):
                        t_growth = t_df.drop(columns=["TeacherName"]).copy()

                        # Guard: avoid empty packets
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
                # Single teacher: still do assessment-first growth build, then filter
                teacher_students = teacher_map_sub[teacher_map_sub["TeacherName"] == teacher_choice]["StudentIdentifier"].unique().tolist()

                prog = st.progress(0.0, text="Generating assessment packets…")
                for i, a in enumerate(assessments_to_run):
                    a_df = results_df[results_df["AssessmentName"] == a].copy()
                    if a_df.empty:
                        prog.progress((i + 1) / max(len(assessments_to_run), 1), text=f"Generating assessment packets… ({i+1}/{len(assessments_to_run)})")
                        continue

                    a_growth = build_growth_table(a_df, w1, w2)
                    t_growth = a_growth[a_growth["StudentIdentifier"].isin(teacher_students)].copy()

                    if t_growth.empty:
                        prog.progress((i + 1) / max(len(assessments_to_run), 1), text=f"Generating assessment packets… ({i+1}/{len(assessments_to_run)})")
                        continue

                    combined_pdf, individual = make_student_packets(
                        t_growth, w1, w2, thresholds, teacher_label=teacher_choice, subject_label=subject_choice
                    )
                    summary_pdf = make_summary_pdf(
                        t_growth, a, w1, w2, teacher_label=teacher_choice, subject_label=subject_choice
                    )
                    add_teacher_assessment_to_zip(z, teacher_choice, subject_choice, a, combined_pdf, summary_pdf, individual)

                    prog.progress((i + 1) / max(len(assessments_to_run), 1), text=f"Generating assessment packets… ({i+1}/{len(assessments_to_run)})")

    st.success("ZIP created.")
    st.download_button(
        "Download ZIP",
        data=zip_buf.getvalue(),
        file_name=f"IAB_FIAB_{('ALL' if run_all_assessments else assessment_name)}.zip".replace(" ", "_"),
        mime="application/zip",
    )
