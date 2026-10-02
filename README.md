# EduPro Academy Dashboard

A Streamlit learner-intelligence project with a redesigned dashboard closely matching the supplied Academy UI reference.

## Run on Windows

Open CMD/PowerShell inside this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or double-click `run_app.bat` after installing the requirements.

## Dashboard UI

- Academy-style left navigation: Dashboard, Courses, Chat, Grades, Schedule, Settings
- Header search, notification and profile actions
- Overview KPI cards
- Active-hours chart and performance trend
- My Assignments table
- Profile / learner selector
- Calendar and upcoming events
- Premium card and subscribe interaction
- Responsive course catalogue with filters and Start/View actions
- Learner chat using the recommendation engine
- Grades and score trend
- Weekly schedule with add-session interaction
- Settings for clustering parameters

## Original ML features retained

The existing `edupro_core.py` engine is still used for:

- learner feature engineering
- K-Means segmentation and silhouette selection
- PCA learner map
- hierarchical/Ward validation
- segment fingerprints
- hybrid course recommendations
- segment comparison
- leave-last-out recommendation evaluation
- recommendation CSV download

The original bundled `Users.csv` remains in `data/`. If Courses/Transactions files are not supplied, the existing core engine generates demo course/transaction data.
