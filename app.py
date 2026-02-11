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


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="IAB/FIAB Progress Reports", layout="wide")

GROWTH_THRESHOLDS = {
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

# Accept both your current export headers and the “future template” headers
COLUMN_ALIASES = {
    "Error Band Min": "ScaleScoreErrorBandMin",
    "Error Band Max": "ScaleScoreErrorBandMax",
    "LastOrSurname": "LastName",
    # If later your template uses ScaleScoreErrorBandMin/Max, this keeps it stable.
}


@dataclass
class Window:
    label: str
    start: date
    end: date


# -----------------------------
# Helpers: parsing + windows
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # rename known aliases to stable internal names
    for old, new in COLUMN_ALIASES.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    return df


def parse_submit_datetime(series: pd.Series) -> pd.Series:
    # Your file looks like "Sep 29, 2025" or "Feb 9, 2026"
    def _parse(x):
        if pd.isna(x) or str(x).strip() == "":
            return pd.NaT
        return dtparser.parse(str(x))
    return series.apply(_parse)


def month_window(year: int, month: int) -> Window:
    last_day = monthrange(year, month)[1]
    return Window(
        label=f"{date(year, month, 1).strftime('%b %Y')}",
        start=date(year, month, 1),
        end=date(year, month, last_day),
    )


def detect_two_windows(dt: pd.Series) -> tuple[Window, Window]:
    """
    Auto-detect two dominant (year, month) clusters.
    If unclear, fallback to Oct and Feb based on the data's school year span.
    """
    dts = dt.dropna()
    if dts.empty:
        # fallback to current year Oct/Feb if nothing
        y = date.today().year
        return month_window(y, 10), month_window(y + 1 if date.today().month > 2 else y, 2)

    ym = dts.dt.to_period("M").astype(str)
    counts = ym.value_counts()

    if len(counts) >= 2:
        top2 = counts.index[:2].tolist()
        # Sort by chronological order
        top2_dt = [datetime.strptime(x, "%Y-%m") for x in top2]
        top2_dt.sort()
        w1 = month_window(top2_dt[0].year, top2_dt[0].month)
        w2 = month_window(top2_dt[1].year, top2_dt[1].month)
        return w1, w2

    # Fallback: use October and February around the school year in the data
    min_dt, max_dt = dts.min(), dts.max()
    # School year typically starts in late summer/fall; pick October of min_dt.year
    oct_year = min_dt.year
    feb_year = oct_year if 2 >= min_dt.month else oct_year + 1
    return month_window(oct_year, 10), month_window(feb_year, 2)


def ensure_date(x) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return dtparser.parse(str(x)).date()


# -----------------------------
# Core logic: pick attempts + growth
# -----------------------------
def pick_latest_in_window(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """
    For each student+assessment, pick the latest attempt in the date window.
    """
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
    """
    Returns one row per StudentIdentifier + AssessmentName with baseline/follow-up and growth.
    """
    base = pick_latest_in_window(df, w1).copy()
    foll = pick_latest_in_window(df, w2).copy()

    # Columns we carry forward
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
        c_y = f"{col}_y"
        if c_y in merged.columns:
            merged[col] = merged[col].fillna(merged[c_y])
            merged.drop(columns=[c_y], inplace=True)

    # Demographic cols might be duplicated; clean similarly
    for col in DEMOGRAPHIC_COLS:
        c_y = f"{col}_y"
        if col in merged.columns and c_y in merged.columns:
            merged[col] = merged[col].fillna(merged[c_y])
            merged.drop(columns=[c_y], inplace=True)

    # Compute growth
    merged["Growth"] = pd.to_numeric(merged["FollowupScore"], errors="coerce") - pd.to_numeric(merged["BaselineScore"], errors="coerce")

    return merged


# -----------------------------
# PDF generation
# -----------------------------
def growth_color(growth: float):
    if pd.isna(growth):
        return colors.black
    if growth >= GROWTH_THRESHOLDS["green_min"]:
        return colors.green
    if growth >= GROWTH_THRESHOLDS["yellow_min"]:
        return colors.darkgoldenrod
    if growth <= GROWTH_THRESHOLDS["red_max"]:
        return colors.red
    return colors.black


def draw_student_one_pager(c: canvas.Canvas, row: pd.Series, w1: Window, w2: Window):
    width, height = letter
    left = 0.75 * inch
    top = height - 0.75 * inch

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, top, "IAB/FIAB Progress Report")

    c.setFont("Helvetica", 11)
    c.drawString(left, top - 22, f"Student: {row.get('LastName','')}, {row.get('FirstName','')}")
    c.drawString(left, top - 38, f"StudentIdentifier: {row.get('StudentIdentifier','')}")
    c.drawString(left, top - 54, f"Assessment: {row.get('AssessmentName','')}")
    c.drawString(left, top - 70, f"SchoolYear: {row.get('SchoolYear','')}    Subject: {row.get('Subject','')}    Grade: {row.get('GradeLevelWhenAssessed','')}")

    # Baseline box
    y = top - 110
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, f"Baseline Window ({w1.start.strftime('%b %d, %Y')} – {w1.end.strftime('%b %d, %Y')})")
    c.setFont("Helvetica", 11)
    bd = row.get("BaselineDate", pd.NaT)
    bd_str = fmt_date(bd)    
    c.drawString(left, y - 18, f"Date: {bd_str}")
    c.drawString(left, y - 34, f"Score: {row.get('BaselineScore','N/A')}")
    c.drawString(left, y - 50, f"Error Band: {row.get('BaselineBandMin','')} – {row.get('BaselineBandMax','')}")
    c.drawString(left, y - 66, f"Reporting Category: {row.get('BaselineCategory','')}")
    c.drawString(left, y - 82, f"Status: {row.get('BaselineStatus','')}")

    # Follow-up box
    y2 = y - 125
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y2, f"Follow-up Window ({w2.start.strftime('%b %d, %Y')} – {w2.end.strftime('%b %d, %Y')})")
    c.setFont("Helvetica", 11)
    fd = row.get("FollowupDate", pd.NaT)
    fd_str = fmt_date(fd)
    c.drawString(left, y2 - 18, f"Date: {fd_str}")
    c.drawString(left, y2 - 34, f"Score: {row.get('FollowupScore','N/A')}")
    c.drawString(left, y2 - 50, f"Error Band: {row.get('FollowupBandMin','')} – {row.get('FollowupBandMax','')}")
    c.drawString(left, y2 - 66, f"Reporting Category: {row.get('FollowupCategory','')}")
    c.drawString(left, y2 - 82, f"Status: {row.get('FollowupStatus','')}")

    # Growth
    y3 = y2 - 125
    c.setFont("Helvetica-Bold", 14)
    g = row.get("Growth", None)
    gtxt = "N/A" if pd.isna(g) else f"{int(g):+d}"
    c.setFillColor(growth_color(g))
    c.drawString(left, y3, f"Growth (Follow-up − Baseline): {gtxt}")
    c.setFillColor(colors.black)

    # Footer note
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(left, 0.6 * inch, "Note: Growth compares the same assessment taken in two windows during the school year.")


def make_student_packets(growth_df: pd.DataFrame, w1: Window, w2: Window) -> tuple[bytes, dict]:
    """
    Returns:
      - combined_pdf_bytes
      - individual_pdfs: {filename: pdf_bytes}
    """
    # Sort for consistent ordering
    out = growth_df.sort_values(["LastName", "FirstName", "StudentIdentifier"]).copy()

    # Combined PDF
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for _, row in out.iterrows():
        draw_student_one_pager(c, row, w1, w2)
        c.showPage()
    c.save()
    combined_bytes = buf.getvalue()

    # Individual PDFs
    individual = {}
    for _, row in out.iterrows():
        b = io.BytesIO()
        cc = canvas.Canvas(b, pagesize=letter)
        draw_student_one_pager(cc, row, w1, w2)
        cc.save()
        last = str(row.get("LastName","")).strip() or "Last"
        first = str(row.get("FirstName","")).strip() or "First"
        sid = str(row.get("StudentIdentifier","")).strip() or "ID"
        fname = f"{last}_{first}_{sid}.pdf".replace(" ", "_")
        individual[fname] = b.getvalue()

    return combined_bytes, individual


def make_summary_pdf(growth_df: pd.DataFrame, assessment_name: str, w1: Window, w2: Window) -> bytes:
    """
    2-page summary:
      Page 1: no demographics
      Page 2: demographics breakdown
    """
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
    c.drawString(left, top - 22, f"Assessment: {assessment_name}")
    c.drawString(left, top - 38, f"Baseline: {w1.start.strftime('%b %d, %Y')} – {w1.end.strftime('%b %d, %Y')}")
    c.drawString(left, top - 54, f"Follow-up: {w2.start.strftime('%b %d, %Y')} – {w2.end.strftime('%b %d, %Y')}")

    total = len(df)
    both = df["Growth"].notna().sum()
    baseline_only = df["BaselineScore"].notna().sum() - both
    follow_only = df["FollowupScore"].notna().sum() - both

    c.drawString(left, top - 84, f"Students in file: {total}")
    c.drawString(left, top - 100, f"With both attempts: {both}")
    c.drawString(left, top - 116, f"Baseline only: {baseline_only}    Follow-up only: {follow_only}")

    g = df["Growth"].dropna()
    y = top - 150
    if not g.empty:
        c.drawString(left, y, f"Mean growth: {g.mean():.1f}    Median growth: {g.median():.1f}")
        y -= 20
        # simple buckets
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
    c.drawString(left, top - 22, f"Assessment: {assessment_name}")

    y = top - 55
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "EnglishLanguageAcquisitionStatus")
    c.setFont("Helvetica", 11)
    y -= 16

    if "EnglishLanguageAcquisitionStatus" in df.columns:
        for val, sub in df.groupby("EnglishLanguageAcquisitionStatus"):
            gg = sub["Growth"].dropna()
            line = f"{val}: n={len(sub)}, both={gg.size}, mean={gg.mean():.1f}" if gg.size else f"{val}: n={len(sub)}, both=0"
            c.drawString(left, y, line)
            y -= 14
    else:
        c.drawString(left, y, "Column not present in file.")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "HispanicOrLatinoEthnicity")
    c.setFont("Helvetica", 11)
    y -= 16
    if "HispanicOrLatinoEthnicity" in df.columns:
        for val, sub in df.groupby("HispanicOrLatinoEthnicity"):
            gg = sub["Growth"].dropna()
            line = f"{val}: n={len(sub)}, both={gg.size}, mean={gg.mean():.1f}" if gg.size else f"{val}: n={len(sub)}, both=0"
            c.drawString(left, y, line)
            y -= 14
    else:
        c.drawString(left, y, "Column not present in file.")
        y -= 14

    y -= 10
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


def make_zip_package(
    teacher_name: str,
    subject: str,
    assessment_name: str,
    combined_students_pdf: bytes,
    individual_pdfs: dict,
    summary_pdf: bytes,
) -> bytes:
    buf = io.BytesIO()
    base_folder = f"{teacher_name}/{subject}/{assessment_name}".replace(" ", "_")

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{base_folder}/Student_Pages.pdf", combined_students_pdf)
        z.writestr(f"{base_folder}/Summary.pdf", summary_pdf)
        for fname, pdf_bytes in individual_pdfs.items():
            z.writestr(f"{base_folder}/Students/{fname}", pdf_bytes)

    return buf.getvalue()


# -----------------------------
# UI
# -----------------------------
st.title("IAB/FIAB Progress Reporting (v0.1)")

st.sidebar.header("Packet Info")
teacher_name = st.sidebar.text_input("TeacherName (for folder naming)", value="Teacher")
subject_override = st.sidebar.selectbox("Subject (for folder naming)", ["Math", "ELA"], index=0)

st.sidebar.header("Upload")
uploaded = st.sidebar.file_uploader("Upload Results CSV", type=["csv"])

if not uploaded:
    st.info("Upload a Results CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)
df = normalize_columns(df)

required = ["StudentIdentifier", "SubmitDateTime", "AssessmentName", "ScaleScore"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Parse dates
df["SubmitDateTimeParsed"] = parse_submit_datetime(df["SubmitDateTime"])

# Ensure name columns exist (your file has FirstName + LastOrSurname -> mapped to LastName)
if "FirstName" not in df.columns:
    df["FirstName"] = ""
if "LastName" not in df.columns:
    df["LastName"] = ""

# Detect default windows
w1_auto, w2_auto = detect_two_windows(df["SubmitDateTimeParsed"])

st.subheader("Detected Windows (override if needed)")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Baseline Window**")
    w1_start = st.date_input("Baseline start", value=w1_auto.start, key="w1s")
    w1_end = st.date_input("Baseline end", value=w1_auto.end, key="w1e")

with col2:
    st.markdown("**Follow-up Window**")
    w2_start = st.date_input("Follow-up start", value=w2_auto.start, key="w2s")
    w2_end = st.date_input("Follow-up end", value=w2_auto.end, key="w2e")

# Fallback option info (as requested)
st.caption("Fallback (if auto-detect fails in future files): Baseline = October, Follow-up = February.")

w1 = Window("Baseline", ensure_date(w1_start), ensure_date(w1_end))
w2 = Window("Follow-up", ensure_date(w2_start), ensure_date(w2_end))

# Assessment selection
assessments = sorted(df["AssessmentName"].dropna().unique().tolist())
assessment_name = st.selectbox("Assessment", assessments)

df_assess = df[df["AssessmentName"] == assessment_name].copy()

growth_df = build_growth_table(df_assess, w1, w2)

st.subheader("Preview (one row per student for this assessment)")
preview_cols = [
    "LastName", "FirstName", "StudentIdentifier",
    "BaselineDate", "BaselineScore",
    "FollowupDate", "FollowupScore",
    "Growth",
    "BaselineCategory", "FollowupCategory",
]
preview_cols = [c for c in preview_cols if c in growth_df.columns]
st.dataframe(growth_df[preview_cols].sort_values(["LastName","FirstName"]), use_container_width=True)

st.subheader("Export")
if st.button("Generate ZIP (student one-pagers + summary)"):
    combined_pdf, individual = make_student_packets(growth_df, w1, w2)
    summary_pdf = make_summary_pdf(growth_df, assessment_name, w1, w2)
    zip_bytes = make_zip_package(
        teacher_name=teacher_name or "Teacher",
        subject=subject_override,
        assessment_name=assessment_name,
        combined_students_pdf=combined_pdf,
        individual_pdfs=individual,
        summary_pdf=summary_pdf,
    )

    st.success("ZIP created.")
    st.download_button(
        "Download ZIP",
        data=zip_bytes,
        file_name=f"{teacher_name}_{subject_override}_{assessment_name}.zip".replace(" ", "_"),
        mime="application/zip",
    )

def fmt_date(x) -> str:
    # Handles pandas NaT safely
    if pd.isna(x):
        return "N/A"
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.strftime("%b %d, %Y")
    return str(x)
