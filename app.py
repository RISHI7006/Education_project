"""EduPro - modern student dashboard inspired by the supplied UI reference.

The ML/segmentation/recommendation engine remains in edupro_core.py.  This file
adds a polished, interactive Streamlit dashboard around that engine.
"""
from datetime import date, timedelta
import calendar

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import edupro_core as ec

st.set_page_config(page_title="EduPro Academy", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# -----------------------------------------------------------------------------
# Theme / layout
# -----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --primary:#5146f5; --primary2:#6257ff; --ink:#252531; --muted:#8f909d; --line:#eeeef5; --bg:#f8f8fc; }
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: radial-gradient(circle at 75% 0%, #fdfdff 0, #f8f8fc 48%, #f7f7fb 100%); color:var(--ink); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"] { visibility:hidden; height:0; }
.block-container { padding: 2rem 2.8rem 2.5rem 2.2rem; max-width: 1600px; }
section[data-testid="stSidebar"] { background:#fff; border-right:1px solid #eeeef4; }
section[data-testid="stSidebar"] > div { padding: 1.25rem 1rem 1rem; }
/* buttons: clean white surface, black/blue text */
.stButton > button,
section[data-testid="stSidebar"] .stButton > button {
    background:#ffffff !important;
    color:#171923 !important;
    border:1px solid #dfe3ee !important;
    border-radius:11px !important;
    min-height:43px;
    font-weight:600;
    box-shadow:0 2px 8px rgba(30,45,80,.04);
}
.stButton > button:hover,
section[data-testid="stSidebar"] .stButton > button:hover {
    background:#ffffff !important;
    color:#3f46f5 !important;
    border-color:#5146f5 !important;
    box-shadow:0 5px 14px rgba(81,70,245,.10);
}
/* active sidebar item remains white, with blue text */
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background:#ffffff !important;
    color:#3f46f5 !important;
    border:1.5px solid #5146f5 !important;
    box-shadow:0 5px 14px rgba(81,70,245,.10);
}
/* Inputs and selects */
input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"] > div {
    background:#ffffff !important;
    color:#171923 !important;
    border-color:#dfe3ee !important;
}
input::placeholder, textarea::placeholder { color:#8d94a6 !important; opacity:1 !important; }
label, [data-testid="stWidgetLabel"] p { color:#171923 !important; }
/* cards */
.card { background:#fff; border:1px solid var(--line); border-radius:18px; padding:20px 20px; box-shadow:0 5px 22px rgba(35,34,61,.035); }
.hero-title { font-size:29px; font-weight:800; letter-spacing:-.8px; margin:0; }
.subtitle { color:#9697a3; margin-top:4px; font-size:13px; }
.section-title { font-size:20px; font-weight:800; margin:8px 0 14px; }
.stat-card { min-height:119px; border-radius:16px; padding:15px 16px; border:1px solid rgba(0,0,0,.02); }
.stat-label { font-size:11px; font-weight:700; color:#555766; }
.stat-value { font-size:31px; font-weight:800; margin:9px 0 7px; }
.stat-line { width:58px; height:3px; border-radius:4px; background:var(--primary); }
.stat-coral { background:#fff2ef; } .stat-green { background:#effaf6; } .stat-blue { background:#eff6fd; } .stat-purple { background:#f2f0ff; }
.profile-name { font-size:16px; font-weight:800; text-align:center; margin-top:8px; }
.profile-role { color:#999aa6; font-size:11px; text-align:center; }
.avatar { width:78px; height:78px; border-radius:50%; border:2px solid var(--primary); display:flex; align-items:center; justify-content:center; margin:auto; background:#19191f; color:#fff; font-size:28px; font-weight:800; }
.premium { background:#f1f0ff; border-radius:16px; padding:18px 14px 14px; text-align:center; }
.premium-art { font-size:54px; line-height:1; margin:3px 0 10px; }
.premium h4 { margin:4px 0; font-size:15px; } .premium p { color:#9697a3; font-size:10px; line-height:1.5; }
.premium .stButton > button { background:#fff !important; color:#3f46f5 !important; border:1px solid #5146f5 !important; }
.toolbar { display:flex; align-items:center; gap:10px; }
.small-muted { color:#a0a1ac; font-size:11px; }
.badge { display:inline-block; padding:5px 9px; border-radius:20px; background:#f0efff; color:var(--primary); font-size:10px; font-weight:700; }
.event { border-radius:8px; padding:8px 10px; color:white; font-size:11px; font-weight:700; margin:4px 0; }
.course-card { background:#fff; border:1px solid var(--line); border-radius:16px; padding:16px; height:100%; }
.course-icon { width:40px; height:40px; border-radius:12px; background:#f0efff; display:flex; align-items:center; justify-content:center; font-size:20px; }
.course-title { font-weight:800; margin:12px 0 6px; font-size:15px; }
.course-meta { color:#92939f; font-size:11px; }
.kpi { font-size:26px; font-weight:800; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:15px; padding:12px 14px; }
div[data-testid="stMetricLabel"] { color:#858692; font-size:11px; }
div[data-testid="stMetricValue"] { font-size:25px; }
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; }
/* remove Streamlit tab underline because navigation is custom */
button[data-baseweb="tab"] { display:none !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data…")
def load_from_disk():
    return ec.load_data("data")


@st.cache_data(show_spinner="Reading uploaded files…")
def load_from_upload(u, c, t):
    rd = lambda f: pd.read_csv(f) if f is not None else None
    return ec.load_data("__none__", users_df=rd(u), courses_df=rd(c), tx_df=rd(t))


@st.cache_resource(show_spinner="Building learner intelligence…")
def pipeline(users, courses, tx, k, min_enroll):
    return ec.run_pipeline(users, courses, tx, k=k, min_enroll=min_enroll)


@st.cache_data(show_spinner="Running evaluation…")
def evaluation(users, courses, tx, k, min_enroll, max_users):
    res = ec.evaluate(users, courses, tx, k, min_enroll=min_enroll, max_users=max_users)
    return res, res.attrs.get("n_eval_users", 0)


try:
    data = load_from_disk()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

users, courses, tx = data["users"], data["courses"], data["tx"]

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "selected_uid" not in st.session_state:
    st.session_state.selected_uid = users.UserID.iloc[0]
if "chat" not in st.session_state:
    st.session_state.chat = []

# Keep controls available through Settings but default to sensible values.
k_choice = st.session_state.get("k_choice", "Auto (best silhouette)")
min_enroll = st.session_state.get("min_enroll", 2)
R = pipeline(users, courses, tx, None if str(k_choice).startswith("Auto") else int(k_choice), min_enroll)
feat, rec = R["feat"], R["recommender"]
segments = sorted(feat.Segment.unique())
if st.session_state.selected_uid not in feat.index:
    st.session_state.selected_uid = feat.index[0]
uid = st.session_state.selected_uid
row = feat.loc[uid]
user_row = users.loc[users["UserID"] == uid].iloc[0] if not users.loc[users["UserID"] == uid].empty else None
user_email = str(user_row["Email"]) if user_row is not None and "Email" in users.columns else "Not available"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def navigate(page):
    st.session_state.page = page


def icon_for_category(cat):
    return {"Data Science":"📊", "Web Development":"💻", "Business":"💼", "Finance":"💰",
            "Design":"🎨", "Digital Marketing":"📣", "Cybersecurity":"🔐", "Personal Development":"🌱"}.get(str(cat), "📚")


def safe_mean(series, default=0):
    x = pd.to_numeric(series, errors="coerce").dropna()
    return float(x.mean()) if len(x) else default


def page_header(title, subtitle=""):
    st.markdown(f'<div class="hero-title">{title}</div><div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)


def card_metric(label, value, cls="stat-purple", icon=""):
    st.markdown(f'''<div class="stat-card {cls}"><div class="stat-label">{icon} {label}</div><div class="stat-value">{value}</div><div class="stat-line"></div></div>''', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar: reference-style navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='font-size:18px;font-weight:800;margin:4px 8px 22px;'>◼ <span style='margin-left:8px'>Academy</span></div>", unsafe_allow_html=True)
    nav = [
        ("▦", "Dashboard"), ("▱", "Courses"), ("◌", "Chat"),
        ("▥", "Grades"), ("□", "Schedule"), ("⚙", "Settings")
    ]
    for ico, label in nav:
        if st.button(f"{ico}   {label}", key=f"nav_{label}", use_container_width=True,
                     type="primary" if st.session_state.page == label else "secondary"):
            navigate(label); st.rerun()

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='premium'><div class='premium-art'>👩🏻‍💻📚</div><h4>Get Premium</h4><p>Buy premium and get access to new courses and learning tools.</p></div>", unsafe_allow_html=True)
    if st.button("Subscribe", use_container_width=True, key="subscribe"):
        st.session_state.premium = True
        st.toast("Premium mode enabled for this demo.")

    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
    st.caption(f"{len(users):,} learners · {len(courses):,} courses")

# -----------------------------------------------------------------------------
# Top bar
# -----------------------------------------------------------------------------
left, search_col, notif_col, profile_col = st.columns([3.5, 2.2, .45, 1.15])
with left:
    page_header(f"Hello {row.UserName.split()[0].title()} 👋", "Let’s learn something new today!")
with search_col:
    search = st.text_input("", placeholder="⌕  Search", label_visibility="collapsed", key="global_search")
with notif_col:
    if st.button("🔔", key="notification"):
        st.toast("You have 3 new learning updates.")
with profile_col:
    if st.button("Profile  ✎", key="profile_top", use_container_width=True):
        navigate("Settings"); st.rerun()

if search.strip():
    q = search.strip().lower()
    sr = courses[courses.CourseName.astype(str).str.lower().str.contains(q, na=False) |
                 courses.CourseCategory.astype(str).str.lower().str.contains(q, na=False)].head(8)
    if not sr.empty:
        st.markdown("<div class='card'><b>Search results</b></div>", unsafe_allow_html=True)
        for _, r in sr.iterrows():
            a, b = st.columns([5,1])
            a.write(f"{icon_for_category(r.CourseCategory)}  **{r.CourseName}** · {r.CourseCategory} · {r.CourseLevel}")
            if b.button("Open", key=f"search_{r.CourseID}"):
                st.session_state.course_focus = r.CourseID
                navigate("Courses"); st.rerun()

# -----------------------------------------------------------------------------
# DASHBOARD - visual structure closely follows supplied image
# -----------------------------------------------------------------------------
if st.session_state.page == "Dashboard":
    st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)
    # Dynamic dashboard numbers
    course_in_progress = int(np.ceil(len(courses) * .08))
    course_completed = int(max(1, round(len(tx) * .65)))
    certificates = int(max(1, round(len(tx) * .18)))
    support = int(max(1, round(len(users) * .03)))
    s1, s2, s3, s4 = st.columns(4)
    with s1: card_metric("Course in Progress", course_in_progress, "stat-coral", "▣")
    with s2: card_metric("Course Completed", course_completed, "stat-green", "▣")
    with s3: card_metric("Certificates Earned", certificates, "stat-blue", "▤")
    with s4: card_metric("Community Support", support, "stat-purple", "☁")

    main, right = st.columns([2.25, .95], gap="large")
    with main:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:18px;font-weight:800'>Actively Hours</div><div class='small-muted'>Weekly learning activity</div>", unsafe_allow_html=True)
            d = pd.to_datetime(tx.TransactionDate, errors="coerce").dropna()
            weekday = d.dt.day_name().value_counts().reindex(["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"], fill_value=0)
            vals = weekday.values.astype(float)
            if vals.max() > 0: vals = np.maximum(1, vals / vals.max() * 7.2)
            fig = go.Figure(go.Bar(x=["S","M","T","W","T","F","S"], y=vals,
                                   marker_color="#5146f5", width=.38, hovertemplate="%{y:.1f} hrs<extra></extra>"))
            fig.update_layout(height=245, margin=dict(l=5,r=5,t=8,b=5), yaxis=dict(range=[0,8], title="8h", showgrid=False), xaxis=dict(showgrid=False), paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#6e6f7a"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            st.markdown("<div style='display:flex;justify-content:space-between;color:#90919c;font-size:11px'><span>Time spent<br><b style='font-size:18px;color:#222'>28h</b> <span style='color:#4cb58a'>85%</span></span><span>Lessons taken<br><b style='font-size:18px;color:#222'>60</b> <span style='color:#4cb58a'>79%</span></span><span>Exam passed<br><b style='font-size:18px;color:#222'>10</b> <span style='color:#4cb58a'>100%</span></span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:18px;font-weight:800'>Performance</div><div class='small-muted'>Monthly learning trend</div>", unsafe_allow_html=True)
            monthly = pd.to_datetime(tx.TransactionDate, errors="coerce").dt.to_period("M").value_counts().sort_index()
            if monthly.empty: monthly = pd.Series([1,2,3], index=pd.period_range("2026-01", periods=3, freq="M"))
            x = [str(v) for v in monthly.index]
            y = monthly.values
            y2 = pd.Series(y).rolling(2, min_periods=1).mean().values
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y2, mode="lines", line=dict(color="#b8e1ee", width=3), name="Trend"))
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#5146f5", width=2.5), name="Activity"))
            fig.update_layout(height=210, margin=dict(l=0,r=0,t=5,b=5), xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), paper_bgcolor="white", plot_bgcolor="white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            st.markdown("<div class='kpi'>40% <span style='font-size:11px;font-weight:500;color:#999'>Your productivity is higher compared to last month</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>My Assignments</div>", unsafe_allow_html=True)
        hist = tx[tx.UserID == uid].merge(courses, on="CourseID").sort_values("TransactionDate", ascending=False)
        assignments = []
        for i, (_, r) in enumerate(hist.head(3).iterrows()):
            assignments.append((r.CourseName, r.TransactionDate, min(200, int(round(float(r.CourseRating or 4)*40))), "Completed"))
        while len(assignments) < 3:
            assignments.append((f"{['Typography','Inclusive design','Drawing'][len(assignments)]} test", date.today()+timedelta(days=len(assignments)+1), None, "Upcoming"))
        rows = []
        for name, dt, grade, status in assignments:
            rows.append({"TASK":name, "DATE":pd.to_datetime(dt).strftime("%d %b, %I:%M %p"), "GRADE":f"{grade}/200" if grade else "-- /200", "UPDATE":status})
        adf = pd.DataFrame(rows)
        st.dataframe(adf, use_container_width=True, hide_index=True, column_config={"UPDATE":st.column_config.TextColumn(width="small")})

    with right:
        # profile card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='avatar'>{row.UserName[:2].upper()}</div><div class='profile-name'>{row.UserName.title()}</div><div class='profile-role'>College Student · {user_email}</div>", unsafe_allow_html=True)
        st.markdown("<hr style='border:0;border-top:1px solid #f0f0f4'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:800;text-align:center'>Selected Learner</div>", unsafe_allow_html=True)
        options = feat.index.tolist()
        selected = st.selectbox("", options, index=options.index(uid), format_func=lambda u:f"{u} · {feat.loc[u,'UserName']}", label_visibility="collapsed")
        if selected != uid:
            st.session_state.selected_uid = selected; st.rerun()
        st.markdown(f"<div style='text-align:center;margin:8px 0'><span class='badge'>{row.Segment}</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
        today = date.today()
        st.markdown(f"<div style='display:flex;justify-content:space-between;font-weight:800'><span>‹</span><span>{today.strftime('%B %Y')}</span><span>›</span></div>", unsafe_allow_html=True)
        days = [today + timedelta(days=i-2) for i in range(6)]
        cols = st.columns(6)
        for c, day in zip(cols, days):
            active = day == today
            c.markdown(f"<div style='text-align:center;padding:7px 1px;border-radius:18px;background:{'#5146f5' if active else '#fff'};color:{'white' if active else '#777'}'><small>{day.strftime('%a')}</small><br><b>{day.day}</b></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title' style='margin-top:22px'>Upcoming Events</div>", unsafe_allow_html=True)
        events = [("09:30", "Team Meetup", "#ef7f58"), ("10:30", "Illustration", "#111111"), ("11:30", "Research", "#1971c2"), ("12:30", "Presentation", "#ef8f78"), ("13:30", "Report", "#4dbb91")]
        for tm, name, clr in events:
            st.markdown(f"<div class='event' style='background:{clr}'>{tm}  ·  {name}</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# COURSES
# -----------------------------------------------------------------------------
elif st.session_state.page == "Courses":
    page_header("Courses", "Explore, filter and start a personalized learning path.")
    f1, f2, f3 = st.columns([2,1,1])
    with f1: q = st.text_input("Search courses", placeholder="Search course name or category")
    with f2: level = st.selectbox("Level", ["All"] + ec.LEVELS)
    with f3: category = st.selectbox("Category", ["All"] + sorted(courses.CourseCategory.unique().tolist()))
    view = courses.copy()
    if q: view = view[view.CourseName.str.contains(q, case=False, na=False) | view.CourseCategory.str.contains(q, case=False, na=False)]
    if level != "All": view = view[view.CourseLevel == level]
    if category != "All": view = view[view.CourseCategory == category]
    st.caption(f"Showing {len(view):,} courses")
    for start in range(0, min(len(view), 24), 3):
        cols = st.columns(3)
        for col, (_, cr) in zip(cols, view.iloc[start:start+3].iterrows()):
            with col:
                st.markdown(f"<div class='course-card'><div class='course-icon'>{icon_for_category(cr.CourseCategory)}</div><div class='course-title'>{cr.CourseName}</div><div class='course-meta'>{cr.CourseCategory} · {cr.CourseLevel} · ⭐ {float(cr.CourseRating):.1f}</div><div style='margin-top:13px'><span class='badge'>{cr.CourseType}</span> <span style='float:right;font-weight:800'>{'Free' if float(cr.CoursePrice)==0 else '₹'+format(float(cr.CoursePrice),',.0f')}</span></div></div>", unsafe_allow_html=True)
                if st.button("Start / View", key=f"course_{cr.CourseID}", use_container_width=True):
                    st.session_state.course_focus = cr.CourseID
                    st.toast(f"Opened {cr.CourseName}")

# -----------------------------------------------------------------------------
# CHAT
# -----------------------------------------------------------------------------
elif st.session_state.page == "Chat":
    page_header("Chat", "Ask EduPro about courses, progress and recommendations.")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    for role, msg in st.session_state.chat:
        if role == "user": st.chat_message("user").write(msg)
        else: st.chat_message("assistant").write(msg)
    prompt = st.chat_input("Ask something about your learning...")
    if prompt:
        st.session_state.chat.append(("user", prompt))
        p = prompt.lower()
        if "recommend" in p or "course" in p:
            recs = rec.recommend(uid, n=5)
            answer = "Here are some courses from your personalized recommendation engine:\n\n" + "\n".join([f"• {x.Course} — {x.Level} — ⭐ {x.Rating}" for _, x in recs.iterrows()])
        elif "progress" in p or "grade" in p:
            answer = f"You have {int(row.total_courses)} course enrolments, an average enrolled rating of {float(row.avg_rating):.2f}, and your current segment is **{row.Segment}**."
        else:
            answer = "I can help with course recommendations, your progress, learner segments, grades, or schedule."
        st.session_state.chat.append(("assistant", answer)); st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# GRADES
# -----------------------------------------------------------------------------
elif st.session_state.page == "Grades":
    page_header("Grades", "Track assessment performance and completed learning activity.")
    g1,g2,g3,g4 = st.columns(4)
    with g1: st.metric("Average rating", f"{float(row.avg_rating):.2f}/5")
    with g2: st.metric("Courses", int(row.total_courses))
    with g3: st.metric("Categories", int(row.n_categories))
    with g4: st.metric("Segment", str(row.Segment))
    hist = tx[tx.UserID == uid].merge(courses, on="CourseID").sort_values("TransactionDate")
    if hist.empty:
        st.info("No completed learning records for this learner yet.")
    else:
        gd = hist[["TransactionDate","CourseName","CourseCategory","CourseLevel","CourseRating","Amount"]].copy()
        gd["Score"] = (pd.to_numeric(gd.CourseRating, errors="coerce") / 5 * 100).round(0).astype(int)
        st.dataframe(gd, use_container_width=True, hide_index=True)
        fig = px.line(gd, x="TransactionDate", y="Score", markers=True, title="Learning score trend")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# SCHEDULE
# -----------------------------------------------------------------------------
elif st.session_state.page == "Schedule":
    page_header("Schedule", "Plan your week around upcoming learning activities.")
    today = date.today()
    start = today - timedelta(days=today.weekday())
    days = [start + timedelta(days=i) for i in range(7)]
    events = {0:["Team Meetup"], 1:["Illustration"], 2:["Research"], 3:["Presentation"], 4:["Report"], 5:[], 6:[]}
    cols = st.columns(7)
    for i, (c, day) in enumerate(zip(cols, days)):
        with c:
            st.markdown(f"<div class='card' style='min-height:230px'><div style='font-weight:800'>{day.strftime('%a')}</div><div class='small-muted'>{day.strftime('%d %b')}</div><hr style='border:0;border-top:1px solid #f0f0f4'>" + ''.join([f"<div class='event' style='background:#5146f5'>{e}</div>" for e in events[i]]) + "</div>", unsafe_allow_html=True)
    st.markdown("### Add learning session")
    a,b,c = st.columns(3)
    session_name = a.text_input("Session name", "Study session")
    session_day = b.date_input("Date", today)
    session_time = c.time_input("Time")
    if st.button("Add to schedule", type="primary"):
        st.session_state.setdefault("custom_events", []).append((session_day, session_time, session_name))
        st.success(f"Added {session_name} on {session_day.strftime('%d %b')} at {session_time.strftime('%I:%M %p')}.")
    if st.session_state.get("custom_events"):
        st.markdown("### Your added sessions")
        st.dataframe(pd.DataFrame(st.session_state.custom_events, columns=["Date","Time","Session"]), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------
elif st.session_state.page == "Settings":
    page_header("Settings", "Customize the dashboard and learner intelligence engine.")
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Profile")
        st.text_input("Name", row.UserName.title())
        st.text_input("Email", user_email)
        st.selectbox("Learning goal", ["Career growth", "Academic learning", "Skill building", "Exploration"])
        st.checkbox("Learning reminders", value=True)
        st.checkbox("Weekly progress email", value=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Segmentation")
        kc = st.selectbox("Number of segments", ["Auto (best silhouette)"] + list(range(3,9)), index=0 if str(k_choice).startswith("Auto") else int(k_choice)-2)
        me = st.slider("Minimum enrolments", 1, 5, int(min_enroll))
        if st.button("Apply ML settings", type="primary"):
            st.session_state.k_choice = kc
            st.session_state.min_enroll = me
            st.cache_resource.clear()
            st.success("Settings applied. Reloading learner intelligence…")
            st.rerun()
        st.markdown("### Data source")
        st.info("This dashboard currently uses the bundled Users.csv. Courses and Transactions are generated when they are not supplied, exactly as supported by the existing EduPro engine.")
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Hidden/secondary ML views from the original project remain accessible from
# the dashboard through compact action buttons.
# -----------------------------------------------------------------------------
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
a,b,c = st.columns([1,1,4])
with a:
    if st.button("📊 Analytics", use_container_width=True):
        st.session_state.page = "Analytics"; st.rerun()
with b:
    if st.button("🎯 My recommendations", use_container_width=True):
        st.session_state.page = "Recommendations"; st.rerun()

if st.session_state.page == "Analytics":
    page_header("Analytics", "Full learner segmentation, explorer, comparison and evaluation tools.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Learners", f"{len(feat):,}"); c2.metric("Courses", f"{len(courses):,}"); c3.metric("Enrolments", f"{len(tx):,}"); c4.metric("Silhouette", f"{R['silhouette']:.3f}")
    at1, at2, at3, at4 = st.tabs(["👤 Learner Explorer", "🧩 Cluster Dashboard", "⚖️ Segment Comparison", "✅ Evaluation"])

    with at1:
        st.subheader(f"{row.UserName.title()} ({uid})")
        st.markdown(f"**Segment:** `{row.Segment}` · Age {int(row.Age)} · {row.Gender}")
        m1,m2,m3,m4,m5,m6=st.columns(6)
        m1.metric("Courses", int(row.total_courses)); m2.metric("Categories", int(row.n_categories)); m3.metric("Avg spend", f"{row.avg_spend:,.0f}")
        m4.metric("Avg rating", f"{row.avg_rating:.2f}"); m5.metric("Preferred category", row.preferred_category); m6.metric("Preferred level", row.preferred_level)
        hist = tx[tx.UserID == uid].merge(courses, on="CourseID").sort_values("TransactionDate")
        if not hist.empty:
            a,b=st.columns([3,2])
            a.dataframe(hist[["TransactionDate","CourseName","CourseCategory","CourseLevel","CourseRating","Amount"]], use_container_width=True, hide_index=True)
            b.plotly_chart(px.sunburst(hist, path=["CourseCategory","CourseLevel"], title="Learning mix"), use_container_width=True)
        cols=ec.NUM_FEATURES
        z=(feat[feat.Segment==row.Segment][cols].mean()-feat[cols].mean())/feat[cols].std()
        zl=(row[cols].astype(float)-feat[cols].mean())/feat[cols].std()
        fig=go.Figure([go.Bar(name="This learner",x=[ec.FEATURE_LABELS[c] for c in cols],y=zl.values),go.Bar(name=f"{row.Segment} avg",x=[ec.FEATURE_LABELS[c] for c in cols],y=z.values)])
        fig.update_layout(barmode="group",height=380,yaxis_title="z-score")
        st.plotly_chart(fig,use_container_width=True)

    with at2:
        a,b=st.columns([3,2])
        samp=feat.sample(min(3000,len(feat)),random_state=1)
        a.plotly_chart(px.scatter(samp,x="PC1",y="PC2",color="Segment",opacity=.65,hover_name="UserName",title=f"Learner map · {R['pca_var'].sum():.0%} variance explained"),use_container_width=True)
        b.plotly_chart(px.bar(feat.Segment.value_counts().reset_index(),x="Segment",y="count",color="Segment",title="Segment sizes"),use_container_width=True)
        prof=ec.segment_profile(feat)
        zz=(prof[ec.NUM_FEATURES]-feat[ec.NUM_FEATURES].mean())/feat[ec.NUM_FEATURES].std(); zz.columns=[ec.FEATURE_LABELS[c] for c in zz.columns]
        st.plotly_chart(px.imshow(zz,color_continuous_scale="RdBu_r",zmin=-2,zmax=2,aspect="auto",text_auto=".2f",title="Segment fingerprints"),use_container_width=True)
        sw=R["sweep"]
        a,b=st.columns(2)
        f1=px.line(sw,x="k",y="inertia",markers=True,title="Elbow method (inertia)"); f1.add_vline(x=R["k"],line_dash="dash")
        f2=px.line(sw,x="k",y="silhouette",markers=True,title="Silhouette score by k"); f2.add_vline(x=R["k"],line_dash="dash")
        a.plotly_chart(f1,use_container_width=True); b.plotly_chart(f2,use_container_width=True)
        h=R["hier"]; q1,q2,q3=st.columns(3)
        q1.metric("K-Means silhouette",f"{h['kmeans_silhouette_same_sample']:.3f}"); q2.metric("Ward silhouette",f"{h['hier_silhouette']:.3f}"); q3.metric("Agreement (ARI)",f"{h['adjusted_rand_index']:.3f}")
        st.dataframe(prof.round(2),use_container_width=True)

    with at3:
        sel=st.multiselect("Segments to compare",segments,default=segments[:min(3,len(segments))])
        if len(sel)<2: st.info("Pick at least two segments.")
        else:
            prof=ec.segment_profile(feat).loc[sel]
            st.dataframe(prof.round(2).T,use_container_width=True)
            cols=ec.NUM_FEATURES[:-1]; mn,mx=feat[cols].min(),feat[cols].max(); norm=(prof[cols]-mn)/(mx-mn)
            fig=go.Figure()
            for ss in sel: fig.add_trace(go.Scatterpolar(r=norm.loc[ss].values,theta=[ec.FEATURE_LABELS[c] for c in cols],fill="toself",name=ss))
            fig.update_layout(title="Behavioural radar (min-max scaled)",height=480)
            a,b=st.columns(2); a.plotly_chart(fig,use_container_width=True)
            t=tx.merge(courses,on="CourseID").merge(feat[["Segment"]],left_on="UserID",right_index=True); t=t[t.Segment.isin(sel)]
            mix=t.groupby(["Segment","CourseCategory"]).size().reset_index(name="n"); mix["share"]=mix.n/mix.groupby("Segment").n.transform("sum")
            b.plotly_chart(px.bar(mix,x="Segment",y="share",color="CourseCategory",title="Category mix of enrolments",barmode="stack"),use_container_width=True)
            lvl=t.groupby(["Segment","CourseLevel"]).size().reset_index(name="n"); lvl["share"]=lvl.n/lvl.groupby("Segment").n.transform("sum")
            a,b=st.columns(2); a.plotly_chart(px.bar(lvl,x="Segment",y="share",color="CourseLevel",barmode="group",category_orders={"CourseLevel":ec.LEVELS},title="Level mix"),use_container_width=True)
            top=(t.groupby(["Segment","CourseName"]).size().reset_index(name="enrolments").sort_values(["Segment","enrolments"],ascending=[True,False]).groupby("Segment").head(5))
            b.markdown("**Top 5 courses per segment**"); b.dataframe(top,use_container_width=True,hide_index=True)

    with at4:
        st.markdown("**Cluster quality**")
        q1,q2,q3=st.columns(3)
        q1.metric("Silhouette (K-Means)",f"{R['silhouette']:.3f}"); q2.metric("Avg intra-cluster similarity",f"{R['intra'].mean():.3f}"); q3.metric("Ward ↔ K-Means ARI",f"{R['hier']['adjusted_rand_index']:.3f}")
        st.bar_chart(R["intra"].rename("Intra-cluster cosine similarity"))
        n_ev=st.slider("Learners to evaluate",200,2000,800,100)
        if st.button("Run hold-out evaluation",type="primary"):
            res,n_used=evaluation(users,courses,tx,R["k"],min_enroll,n_ev)
            st.caption(f"Each learner's latest enrolment is hidden and models are refit. Evaluated on {n_used:,} learners.")
            st.dataframe(res.round(4),use_container_width=True,hide_index=True)
            m=res.melt(id_vars="Model",value_vars=["HitRate@5","HitRate@10"],var_name="Metric",value_name="Value")
            st.plotly_chart(px.bar(m,x="Model",y="Value",color="Metric",barmode="group",title="Hit rate by model"),use_container_width=True)

if st.session_state.page == "Recommendations":
    page_header("My Recommendations", "Hybrid content + segment popularity + similar learner recommendations.")
    f1,f2,f3=st.columns(3)
    lv=f1.multiselect("Level", ec.LEVELS); ca=f2.multiselect("Category", sorted(courses.CourseCategory.unique())); n=f3.slider("How many",3,20,8)
    recs=rec.recommend(uid,n=n,levels=lv or None,categories=ca or None)
    if recs.empty: st.info("No courses match your filters.")
    else:
        for _,r in recs.iterrows():
            st.markdown(f"<div class='course-card' style='margin-bottom:10px'><b>{icon_for_category(r.Category)} {r.Course}</b><br><span class='course-meta'>{r.Category} · {r.Level} · {r.Type} · ⭐ {r.Rating}</span><br><span class='small-muted'>{r.Why}</span></div>", unsafe_allow_html=True)
        st.download_button("Download recommendations (CSV)", recs.to_csv(index=False), f"recs_{uid}.csv", mime="text/csv")
