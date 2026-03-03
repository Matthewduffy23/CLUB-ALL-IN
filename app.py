# team_squad_app.py — Combined Team Profile + Squad Depth Chart
# pip install streamlit pandas numpy matplotlib scikit-learn requests
# streamlit run team_squad_app.py

import io, os, re, math, unicodedata
from pathlib import Path
from datetime import date

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import requests

st.set_page_config(page_title="TEAM HQ + SQUAD", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED CONSTANTS / HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Column normalisation ──────────────────────────────────────────────────────
COL_MAP = {
    "League":            ["league"],
    "Team":              ["team"],
    "Matches":           ["matches"],
    "Wins":              ["wins"],
    "Draws":             ["draws"],
    "Losses":            ["losses"],
    "Points":            ["points"],
    "Expected Points":   ["expected points","xpoints","x points","expected_points"],
    "Goals For":         ["goals for","goals scored","goals_for"],
    "Goals Against":     ["goals against","goals conceded","goals_against"],
    "Goal Difference":   ["goal difference","goal diff","goal_difference"],
    "Avg Age":           ["avg age","average age","avg_age"],
    "Possession %":      ["possession %","possession","possession_pct"],
    "Goals p90":         ["goals p90","goals per 90","goals_p90"],
    "xG p90":            ["xg p90","xg per 90","xg_p90"],
    "Shots p90":         ["shots p90","shots per 90","shots_p90"],
    "Shot Accuracy %":   ["shot accuracy %","shooting accuracy %","shot_accuracy_pct"],
    "Crosses p90":       ["crosses p90","crosses per 90","crosses_p90"],
    "Cross Accuracy %":  ["cross accuracy %","crossing accuracy %","cross_accuracy_pct"],
    "Dribbles p90":      ["dribbles p90","dribbles per 90","dribbles_p90"],
    "Touches in Box p90":["touches in box p90","touches in box per 90","touches_in_box_p90"],
    "Shots Against p90": ["shots against p90","shots vs p90","shots_against_p90"],
    "Defensive Duels p90":["defensive duels p90","defensive_duels_p90"],
    "Defensive Duels Won %":["defensive duels won %","defensive_duels_won_pct","def duels won %"],
    "Aerial Duels p90": ["aerial duels p90","aerial_duels_p90"],
    "Aerial Duels Won %":["aerial duels won %","aerial_duels_won_pct"],
    "PPDA":              ["ppda"],
    "Passes p90":        ["passes p90","passes per 90","passes_p90"],
    "Pass Accuracy %":   ["pass accuracy %","passing accuracy %","pass_accuracy_pct","accurate passes %"],
    "Through Passes p90":["through passes p90","through_passes_p90"],
    "Passes to Final Third p90":["passes to final third p90","passes_to_final_third_p90","passes to final 3rd p90"],
    "Passes to Final Third Acc %":["passes to final third acc %","passes_to_final_third_acc_pct"],
    "Long Passes p90":   ["long passes p90","long_passes_p90"],
    "Long Pass Accuracy %":["long pass accuracy %","long_pass_accuracy_pct"],
    "Progressive Passes p90":["progressive passes p90","progressive_passes_p90"],
    "Progressive Runs p90":["progressive runs p90","progressive_runs_p90"],
    "xG Against p90":    ["xg against p90","xga p90","xg_against_p90","xg against"],
    "Goals Against p90": ["goals against p90","goals conceded p90","goals_against_p90"],
}

NUMERIC_COLS = [c for c in COL_MAP.keys() if c not in ("League","Team")]
INVERT_METRICS = {"xG Against p90","Goals Against p90","Shots Against p90","PPDA","Goals Against"}

METRIC_LABELS = {
    "Crosses p90":"Crosses","Cross Accuracy %":"Crossing Accuracy %",
    "Goals p90":"Goals Scored","xG p90":"xG","Shots p90":"Shots",
    "Shot Accuracy %":"Shooting %","Touches in Box p90":"Touches in Box",
    "Aerial Duels Won %":"Aerial Duel Success %","Goals Against p90":"Goals Against",
    "xG Against p90":"xG Against","Defensive Duels p90":"Defensive Duels",
    "Defensive Duels Won %":"Defensive Duel Win %","Shots Against p90":"Shots Against",
    "PPDA":"PPDA","Aerial Duels p90":"Aerial Duels","Dribbles p90":"Dribbles",
    "Passes p90":"Passes","Pass Accuracy %":"Passing Accuracy %",
    "Long Passes p90":"Long Passes","Long Pass Accuracy %":"Long Passing %",
    "Possession %":"Possession","Passes to Final Third p90":"Passes to Final 3rd",
    "Progressive Passes p90":"Progressive Passes","Progressive Runs p90":"Progressive Runs",
    "Expected Points":"xPoints","Points":"Points","Goals For":"Goals For",
    "Goals Against":"Goals Against (Total)","Matches":"Matches","Avg Age":"Avg Age",
}
def mlabel(col): return METRIC_LABELS.get(col, col)

# ── League / region helpers ───────────────────────────────────────────────────
LEAGUE_STRENGTHS = {
    "England 1":100.00,"Spain 1":87.84,"Germany 1":87.45,"Italy 1":85.88,"France 1":83.14,
    "England 2":75.10,"Belgium 1":74.51,"Brazil 1":74.31,"Portugal 1":72.94,"Argentina 1":71.37,
    "USA 1":70.00,"Denmark 1":70.78,"Poland 1":69.61,"Turkey 1":69.02,"Netherlands 1":69.02,
    "Croatia 1":68.43,"Germany 2":68.04,"Japan 1":67.84,"Switzerland 1":67.45,"Spain 2":67.06,
    "Norway 1":66.67,"Mexico 1":66.47,"Sweden 1":66.27,"Colombia 1":65.88,"Czech 1":65.29,
    "Ecuador 1":65.29,"Greece 1":64.12,"Italy 2":63.53,"Hungary 1":63.53,"Austria 1":63.33,
    "Morocco 1":63.14,"Korea 1":62.75,"France 2":64.00,"England 3":61.96,"Romania 1":61.76,
    "Scotland 1":61.76,"Uruguay 1":60.39,"Chile 1":59.80,"Israel 1":58.43,"Slovenia 1":57.45,
    "Slovakia 1":56.47,"Germany 3":54.51,"Ukraine 1":54.31,"Portugal 2":53.14,
    "Serbia 1":52.16,"England 4":50.78,"Ireland 1":50.59,"Russia 1":62.41,
    "France 3":49.61,"Scotland 2":38.63,"England 5":33.33,"England 6":16.08,
}
COUNTRY_TO_REGION = {
    "England":"Europe","Spain":"Europe","Germany":"Europe","Italy":"Europe","France":"Europe",
    "Belgium":"Europe","Portugal":"Europe","Netherlands":"Europe","Croatia":"Europe",
    "Switzerland":"Europe","Norway":"Europe","Sweden":"Europe","Czech":"Europe",
    "Greece":"Europe","Austria":"Europe","Hungary":"Europe","Romania":"Europe",
    "Scotland":"Europe","Slovenia":"Europe","Slovakia":"Europe","Ukraine":"Europe",
    "Bulgaria":"Europe","Serbia":"Europe","Albania":"Europe","Bosnia":"Europe",
    "Kosovo":"Europe","Ireland":"Europe","Finland":"Europe","Armenia":"Europe",
    "Georgia":"Europe","Poland":"Europe","Iceland":"Europe","Latvia":"Europe",
    "Montenegro":"Europe","Denmark":"Europe","Estonia":"Europe",
    "Northern Ireland":"Europe","Wales":"Europe","Russia":"Europe",
    "Kazakhstan":"Europe","Israel":"Europe","Turkey":"Asia","Australia":"Oceania",
    "Brazil":"South America","Argentina":"South America","Colombia":"South America",
    "Ecuador":"South America","Uruguay":"South America","Chile":"South America",
    "USA":"North America","Mexico":"North America","Japan":"Asia","Korea":"Asia",
    "China":"Asia","Azerbaijan":"Asia","Morocco":"Africa","Tunisia":"Africa",
    "South Africa":"Africa",
}
def league_country(lg): return re.sub(r"\s*\d+\s*$","",str(lg)).strip().rstrip(".")
def league_region(lg): return COUNTRY_TO_REGION.get(league_country(lg),"Other")

COUNTRY_TO_CC = {
    "england":"eng","scotland":"sct","wales":"wls","northern ireland":"gb",
    "australia":"au","austria":"at","belgium":"be","bulgaria":"bg","croatia":"hr",
    "czech":"cz","denmark":"dk","france":"fr","germany":"de","hungary":"hu",
    "italy":"it","netherlands":"nl","norway":"no","poland":"pl","portugal":"pt",
    "romania":"ro","spain":"es","sweden":"se","switzerland":"ch","turkey":"tr",
    "ukraine":"ua","ireland":"ie",
}
TWEMOJI_SPECIAL={"eng":"1f3f4-e0067-e0062-e0065-e006e-e0067-e007f",
                  "sct":"1f3f4-e0067-e0062-e0073-e0063-e0074-e007f",
                  "wls":"1f3f4-e0067-e0062-e0077-e006c-e0073-e007f"}

def _norm(s): return unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode().strip().lower()

def flag_html(league_name):
    country=league_country(league_name); n=_norm(country)
    cc=COUNTRY_TO_CC.get(n,"")
    if not cc: return ""
    if cc in TWEMOJI_SPECIAL: code=TWEMOJI_SPECIAL[cc]
    else:
        if len(cc)!=2: return ""
        base=0x1F1E6
        code=f"{base+(ord(cc[0].upper())-65):x}-{base+(ord(cc[1].upper())-65):x}"
    src=f"https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/{code}.svg"
    return f"<img src='{src}' style='height:18px;vertical-align:middle;margin-right:4px;'>"

# ── Colour helpers ────────────────────────────────────────────────────────────
def rating_color(v):
    v=float(v)
    if v>=85: return "#2E6114"
    if v>=75: return "#5C9E2E"
    if v>=66: return "#7FBC41"
    if v>=54: return "#A7D763"
    if v>=44: return "#F6D645"
    if v>=25: return "#D77A2E"
    return "#C63733"

def fmt2(n):
    try: return f"{max(0,min(99,int(float(n)))):02d}"
    except: return "00"

# ═══════════════════════════════════════════════════════════════════════════════
# TEAM STATS CSV LOADER & NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _read_team_csv(data:bytes)->pd.DataFrame: return pd.read_csv(io.BytesIO(data))
@st.cache_data(show_spinner=False)
def _read_team_path(path:str)->pd.DataFrame: return pd.read_csv(path)

def normalise_cols(df:pd.DataFrame)->pd.DataFrame:
    rename={}; existing_lower={c.lower().strip():c for c in df.columns}
    for canonical,aliases in COL_MAP.items():
        if canonical in df.columns: continue
        for alias in aliases:
            if alias in existing_lower: rename[existing_lower[alias]]=canonical; break
    return df.rename(columns=rename)

def pct_rank(series:pd.Series,invert:bool=False)->pd.Series:
    r=series.rank(pct=True)*100
    return 100-r if invert else r

# ═══════════════════════════════════════════════════════════════════════════════
# SQUAD DEPTH — Role Buckets, Formation, Helpers (from squad app)
# ═══════════════════════════════════════════════════════════════════════════════
ROLE_BUCKETS = {
    "GK":{"Shot Stopper GK":{"metrics":{"Prevented goals per 90":3,"Save rate, %":1}},
          "Ball Playing GK":{"metrics":{"Passes per 90":1,"Accurate passes, %":3,"Accurate long passes, %":2}},
          "Sweeper GK":{"metrics":{"Exits per 90":1}}},
    "CB":{"Ball Playing CB":{"metrics":{"Passes per 90":2,"Accurate passes, %":2,"Forward passes per 90":2,
              "Accurate forward passes, %":2,"Progressive passes per 90":2,"Progressive runs per 90":1.5,
              "Dribbles per 90":1.5,"Accurate long passes, %":1,"Passes to final third per 90":1.5}},
          "Wide CB":{"metrics":{"Defensive duels per 90":1.5,"Defensive duels won, %":2,
              "Dribbles per 90":2,"Forward passes per 90":1,"Progressive passes per 90":1,"Progressive runs per 90":2}},
          "Box Defender":{"metrics":{"Aerial duels per 90":1,"Aerial duels won, %":3,
              "PAdj Interceptions":2,"Shots blocked per 90":1,"Defensive duels won, %":4}}},
    "FB":{"Build Up FB":{"metrics":{"Passes per 90":2,"Accurate passes, %":1.5,"Forward passes per 90":2,
              "Accurate forward passes, %":2,"Progressive passes per 90":2.5,"Progressive runs per 90":2,
              "Dribbles per 90":2,"Passes to final third per 90":2,"xA per 90":1}},
          "Attacking FB":{"metrics":{"Crosses per 90":2,"Dribbles per 90":3.5,"Accelerations per 90":1,
              "Successful dribbles, %":1,"Touches in box per 90":2,"Progressive runs per 90":3,
              "Passes to penalty area per 90":2,"xA per 90":3}},
          "Defensive FB":{"metrics":{"Aerial duels per 90":1,"Aerial duels won, %":1.5,
              "Defensive duels per 90":2,"PAdj Interceptions":3,"Shots blocked per 90":1,"Defensive duels won, %":3.5}}},
    "CM":{"Deep Playmaker CM":{"metrics":{"Passes per 90":1,"Accurate passes, %":1,"Forward passes per 90":2,
              "Accurate forward passes, %":1.5,"Progressive passes per 90":3,"Passes to final third per 90":2.5,
              "Accurate long passes, %":1}},
          "Advanced Playmaker CM":{"metrics":{"Deep completions per 90":1.5,"Smart passes per 90":2,
              "xA per 90":4,"Passes to penalty area per 90":2}},
          "Defensive CM":{"metrics":{"Defensive duels per 90":4,"Defensive duels won, %":4,
              "PAdj Interceptions":3,"Aerial duels per 90":0.5,"Aerial duels won, %":1}},
          "Ball Carrying CM":{"metrics":{"Dribbles per 90":4,"Successful dribbles, %":2,
              "Progressive runs per 90":3,"Accelerations per 90":3}}},
    "ATT":{"Playmaker ATT":{"metrics":{"Passes per 90":2,"xA per 90":3,"Key passes per 90":1,
               "Deep completions per 90":1.5,"Smart passes per 90":1.5,"Passes to penalty area per 90":2}},
           "Goal Threat ATT":{"metrics":{"xG per 90":3,"Non-penalty goals per 90":3,"Shots per 90":2,"Touches in box per 90":2}},
           "Ball Carrier ATT":{"metrics":{"Dribbles per 90":4,"Successful dribbles, %":2,
               "Progressive runs per 90":3,"Accelerations per 90":3}}},
    "CF":{"Target Man CF":{"metrics":{"Aerial duels per 90":3,"Aerial duels won, %":5}},
          "Goal Threat CF":{"metrics":{"Non-penalty goals per 90":3,"Shots per 90":1.5,"xG per 90":3,
              "Touches in box per 90":1,"Shots on target, %":0.5}},
          "Link Up CF":{"metrics":{"Passes per 90":2,"Passes to penalty area per 90":1.5,
              "Deep completions per 90":1,"Smart passes per 90":1.5,"Accurate passes, %":1.5,
              "Key passes per 90":1,"Dribbles per 90":2,"Successful dribbles, %":1,
              "Progressive runs per 90":2,"xA per 90":3}}},
}
ROLE_KEY_MAP={
    "GK":"GK","CB":"CB","LCB":"CB","RCB":"CB","LB":"FB","RB":"FB","LWB":"FB","RWB":"FB",
    "DMF":"CM","LDMF":"CM","RDMF":"CM","LCMF":"CM","RCMF":"CM",
    "AMF":"ATT","LAMF":"ATT","LW":"ATT","LWF":"ATT","RAMF":"ATT","RW":"ATT","RWF":"ATT","CF":"CF",
}
POS_POOL_MAP={"GK":["GK"],"CB":["CB","LCB","RCB"],"FB":["LB","RB","LWB","RWB"],
              "CM":["DMF","LDMF","RDMF","LCMF","RCMF"],
              "ATT":["AMF","LAMF","RAMF","LW","LWF","RW","RWF"],"CF":["CF"]}
CANONICAL={
    "GK":"GK","CB":"CB","LCB":"LCB","RCB":"RCB","LB":"LB","LWB":"LWB","RB":"RB","RWB":"RWB",
    "DMF":"DM","LDMF":"DM","RDMF":"DM","LCMF":"CM","RCMF":"CM",
    "AMF":"AM","LAMF":"LW","LW":"LW","LWF":"LW","RAMF":"RW","RW":"RW","RWF":"RW","CF":"ST",
}
SIDE_PREF={"RCB":"R","RCMF":"R","RDMF":"R","RB":"R","RWB":"R","RW":"R","RWF":"R","RAMF":"R",
            "LCB":"L","LCMF":"L","LDMF":"L","LB":"L","LWB":"L","LW":"L","LWF":"L","LAMF":"L"}
PITCH_ORDER=["GK","LCB","CB","RCB","LB","RB","LWB","RWB","CM","DM","AM","LW","RW","ST"]

def _tok(pos): return str(pos).split(",")[0].strip().upper()
def _canon(pos): return CANONICAL.get(_tok(pos),"CM")
def _side(pos): return SIDE_PREF.get(_tok(pos),"N")
def _role_key(pos): return ROLE_KEY_MAP.get(_tok(pos),"ATT")
def _all_toks(pos): return [t.strip().upper() for t in str(pos).split(",") if t.strip()]
def _multi_role(pos): return len(_all_toks(pos))>=4

FORMATIONS={
    "4-2-3-1":[
        {"id":"ST","label":"ST","x":50,"y":14,"accepts":["ST"],"side":"N"},
        {"id":"LW","label":"LW","x":13,"y":30,"accepts":["LW"],"side":"L","native_toks":["LW","LWF","LAMF"]},
        {"id":"AM","label":"AM","x":50,"y":32,"accepts":["AM"],"side":"N","priority_toks":["AMF"],"native_toks":["AMF"]},
        {"id":"RW","label":"RW","x":87,"y":30,"accepts":["RW"],"side":"R","native_toks":["RW","RWF","RAMF"]},
        {"id":"DM","label":"DM","x":35,"y":51,"accepts":["DM"],"side":"L"},
        {"id":"CM","label":"CM","x":65,"y":51,"accepts":["CM"],"side":"R"},
        {"id":"LB","label":"LB","x":12,"y":66,"accepts":["LB","LWB"],"side":"L","wb_only":True},
        {"id":"CB1","label":"CB","x":32,"y":72,"accepts":["CB","LCB","RCB"],"side":"L"},
        {"id":"CB2","label":"CB","x":68,"y":72,"accepts":["CB","LCB","RCB"],"side":"R"},
        {"id":"RB","label":"RB","x":88,"y":66,"accepts":["RB","RWB"],"side":"R","wb_only":True},
        {"id":"GK","label":"GK","x":50,"y":89,"accepts":["GK"],"side":"N"},
    ],
    "4-3-3":[
        {"id":"ST","label":"ST","x":50,"y":14,"accepts":["ST"],"side":"N"},
        {"id":"LW","label":"LW","x":14,"y":21,"accepts":["LW"],"side":"L","native_toks":["LW","LWF","LAMF"]},
        {"id":"RW","label":"RW","x":86,"y":21,"accepts":["RW"],"side":"R","native_toks":["RW","RWF","RAMF"]},
        {"id":"CM","label":"CM","x":22,"y":41,"accepts":["CM"],"side":"L"},
        {"id":"DM","label":"DM","x":50,"y":49,"accepts":["DM"],"side":"N"},
        {"id":"AM","label":"AM","x":78,"y":41,"accepts":["AM"],"side":"R"},
        {"id":"LB","label":"LB","x":12,"y":66,"accepts":["LB","LWB"],"side":"L","wb_only":True},
        {"id":"CB1","label":"CB","x":32,"y":72,"accepts":["CB","LCB","RCB"],"side":"L"},
        {"id":"CB2","label":"CB","x":68,"y":72,"accepts":["CB","LCB","RCB"],"side":"R"},
        {"id":"RB","label":"RB","x":88,"y":66,"accepts":["RB","RWB"],"side":"R","wb_only":True},
        {"id":"GK","label":"GK","x":50,"y":89,"accepts":["GK"],"side":"N"},
    ],
    "4-4-2":[
        {"id":"ST1","label":"ST","x":35,"y":14,"accepts":["ST"],"side":"L"},
        {"id":"ST2","label":"ST","x":65,"y":14,"accepts":["ST"],"side":"R"},
        {"id":"LW","label":"LW","x":9,"y":39,"accepts":["LW"],"side":"L","native_toks":["LW","LWF","LAMF"]},
        {"id":"CM1","label":"CM","x":34,"y":43,"accepts":["CM"],"side":"L"},
        {"id":"CM2","label":"CM","x":66,"y":43,"accepts":["CM"],"side":"R"},
        {"id":"RW","label":"RW","x":91,"y":39,"accepts":["RW"],"side":"R","native_toks":["RW","RWF","RAMF"]},
        {"id":"LB","label":"LB","x":12,"y":66,"accepts":["LB","LWB"],"side":"L","wb_only":True},
        {"id":"CB1","label":"CB","x":32,"y":72,"accepts":["CB","LCB","RCB"],"side":"L"},
        {"id":"CB2","label":"CB","x":68,"y":72,"accepts":["CB","LCB","RCB"],"side":"R"},
        {"id":"RB","label":"RB","x":88,"y":66,"accepts":["RB","RWB"],"side":"R","wb_only":True},
        {"id":"GK","label":"GK","x":50,"y":89,"accepts":["GK"],"side":"N"},
    ],
    "3-5-2":[
        {"id":"ST1","label":"ST","x":35,"y":14,"accepts":["ST"],"side":"L"},
        {"id":"ST2","label":"ST","x":65,"y":14,"accepts":["ST"],"side":"R"},
        {"id":"LWB","label":"LWB","x":13,"y":37,"accepts":["LWB","LB"],"side":"L","wb_only":True},
        {"id":"AM","label":"AM","x":30,"y":41,"accepts":["AM"],"side":"L"},
        {"id":"DM","label":"DM","x":50,"y":48,"accepts":["DM"],"side":"N"},
        {"id":"CM","label":"CM","x":70,"y":41,"accepts":["CM"],"side":"R"},
        {"id":"RWB","label":"RWB","x":87,"y":37,"accepts":["RWB","RB"],"side":"R","wb_only":True},
        {"id":"LCB","label":"LCB","x":25,"y":67,"accepts":["LCB","CB"],"side":"L"},
        {"id":"CB","label":"CB","x":50,"y":71,"accepts":["CB","LCB","RCB"],"side":"N"},
        {"id":"RCB","label":"RCB","x":75,"y":67,"accepts":["RCB","CB"],"side":"R"},
        {"id":"GK","label":"GK","x":50,"y":88,"accepts":["GK"],"side":"N"},
    ],
    "3-4-3":[
        {"id":"LW","label":"LW","x":14,"y":21,"accepts":["LW"],"side":"L","native_toks":["LW","LWF","LAMF"]},
        {"id":"ST","label":"ST","x":50,"y":14,"accepts":["ST"],"side":"N"},
        {"id":"RW","label":"RW","x":86,"y":21,"accepts":["RW"],"side":"R","native_toks":["RW","RWF","RAMF"]},
        {"id":"LWB","label":"LWB","x":13,"y":45,"accepts":["LWB","LB"],"side":"L","wb_only":True},
        {"id":"CM","label":"CM","x":38,"y":43,"accepts":["CM"],"side":"L"},
        {"id":"DM","label":"DM","x":62,"y":43,"accepts":["DM"],"side":"R"},
        {"id":"RWB","label":"RWB","x":87,"y":45,"accepts":["RWB","RB"],"side":"R","wb_only":True},
        {"id":"LCB","label":"LCB","x":25,"y":67,"accepts":["LCB","CB"],"side":"L"},
        {"id":"CB","label":"CB","x":50,"y":71,"accepts":["CB","LCB","RCB"],"side":"N"},
        {"id":"RCB","label":"RCB","x":75,"y":67,"accepts":["RCB","CB"],"side":"R"},
        {"id":"GK","label":"GK","x":50,"y":88,"accepts":["GK"],"side":"N"},
    ],
    "4-1-4-1":[
        {"id":"ST","label":"ST","x":50,"y":14,"accepts":["ST"],"side":"N"},
        {"id":"LW","label":"LW","x":9,"y":31,"accepts":["LW"],"side":"L","native_toks":["LW","LWF","LAMF"]},
        {"id":"AM","label":"AM","x":30,"y":38,"accepts":["AM"],"side":"L","priority_toks":["AMF"],"native_toks":["AMF"]},
        {"id":"DM","label":"DM","x":50,"y":41,"accepts":["DM"],"side":"N"},
        {"id":"CM","label":"CM","x":70,"y":38,"accepts":["CM"],"side":"R"},
        {"id":"RW","label":"RW","x":91,"y":31,"accepts":["RW"],"side":"R","native_toks":["RW","RWF","RAMF"]},
        {"id":"LB","label":"LB","x":12,"y":66,"accepts":["LB","LWB"],"side":"L","wb_only":True},
        {"id":"CB1","label":"CB","x":32,"y":72,"accepts":["CB","LCB","RCB"],"side":"L"},
        {"id":"CB2","label":"CB","x":68,"y":72,"accepts":["CB","LCB","RCB"],"side":"R"},
        {"id":"RB","label":"RB","x":88,"y":66,"accepts":["RB","RWB"],"side":"R","wb_only":True},
        {"id":"GK","label":"GK","x":50,"y":89,"accepts":["GK"],"side":"N"},
    ],
}

FALLBACK_CANON={"DMF":["DM","CM"],"LDMF":["DM","CM"],"RDMF":["DM","CM"],
    "LCMF":["CM","DM"],"RCMF":["CM","DM"],
    "AMF":["AM","CM","LW","RW"],"LAMF":["LW","AM","RW"],"RAMF":["RW","AM","LW"],
    "LW":["LW","AM"],"RW":["RW","AM"],"LWF":["LW","AM"],"RWF":["RW","AM"],
    "CF":["ST"],"GK":["GK"],"CB":["CB","LCB","RCB"],"LCB":["LCB","CB"],"RCB":["RCB","CB"],
    "LB":["LB","LWB"],"RB":["RB","RWB"],"LWB":["LWB","LB"],"RWB":["RWB","RB"]}

def contract_years(s):
    s=str(s or "").strip()
    if s in ("","nan","NaT"): return -1
    m=re.search(r"(20\d{2})",s)
    return max(0,int(m.group(1))-date.today().year) if m else -1

def is_loan(p):
    for k in ("On loan","On Loan","on_loan","Loan","loan","On loan?"):
        if k in p and str(p[k]).strip().lower() in ("yes","y","true","1","on loan"):
            return True
    return False

def is_loaned_out(p): return str(p.get("Loaned Out","")).strip().lower() in ("yes","y","true","1")
def is_youth(p): return str(p.get("Youth Player","")).strip().lower() in ("yes","y","true","1")

def player_css_color(yrs,loan,loaned_out=False,youth=False):
    if loaned_out: return "#eab308"
    if youth: return "#9ca3af"
    if loan: return "#22c55e"
    if yrs==0: return "#ef4444"
    if yrs==1: return "#f59e0b"
    return "#ffffff"

def score_to_color(v):
    if np.isnan(float(v if v is not None else np.nan)): return "#4b5563"
    v=max(0.0,min(100.0,float(v)))
    if v<=50:
        t=v/50; r=int(239+(234-239)*t); g=int(68+(179-68)*t); b=int(68+(8-68)*t)
    else:
        t=(v-50)/50; r=int(234+(34-234)*t); g=int(179+(197-179)*t); b=int(8+(94-8)*t)
    return f"rgb({r},{g},{b})"

@st.cache_data(show_spinner=False)
def compute_role_scores(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    skip={"Player","League","Team","Position","Age","Market value","Contract expires",
          "Matches played","Minutes played","Goals","Assists","xG","xA","Birth country","Foot","Height","_ftok","_key"}
    for c in df.columns:
        if c not in skip and not c.startswith("On ") and "loan" not in c.lower():
            df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
    for rk,pool_pos in POS_POOL_MAP.items():
        for role_name,spec in ROLE_BUCKETS.get(rk,{}).items():
            col_out=f"_rs_{role_name}"; df[col_out]=np.nan
            metrics=spec.get("metrics",{})
            for league in df["League"].unique():
                mask=(df["League"]==league)&(df["_ftok"].isin(pool_pos))&(df["Minutes played"]>=200)
                pool=df[mask]
                if pool.empty: continue
                pcts={}
                for met in metrics:
                    if met in pool.columns:
                        pcts[met]=pd.to_numeric(pool[met],errors="coerce").rank(pct=True,method="average")*100.0
                for idx in pool.index:
                    vals,wts=[],[]
                    for met,w in metrics.items():
                        if met in pcts and idx in pcts[met].index:
                            pv=pcts[met].loc[idx]
                            if not np.isnan(pv): vals.append(float(pv)); wts.append(float(w))
                    if vals: df.at[idx,col_out]=float(np.average(vals,weights=wts))
    return df

def assign_players(players,formation_key):
    slots=FORMATIONS.get(formation_key,FORMATIONS["4-2-3-1"])
    by_label={}
    for s in slots: by_label.setdefault(s["label"],[]).append(s)
    assigned=set(); slot_map={s["id"]:[] for s in slots}
    formation_labels=set(by_label.keys())

    def first_tok_fits(p,slot):
        tok=_tok(p.get("Position",""))
        if slot.get("wb_only"):
            return tok in {"LB","LWB","RB","RWB"} and CANONICAL.get(tok,"CM") in slot["accepts"]
        return CANONICAL.get(tok,"CM") in slot["accepts"]

    def primary_fits(p,slot): return first_tok_fits(p,slot)

    def has_any_primary_slot(p):
        tok=_tok(p.get("Position","")); canon=CANONICAL.get(tok,"CM")
        return canon in formation_labels

    def secondary_fits(p,slot):
        if slot.get("wb_only"): return False
        for t in _all_toks(p.get("Position",""))[1:]:
            if CANONICAL.get(t,"CM") in slot["accepts"]: return True
        return False

    def side_score(p,ss):
        ps=_side(p.get("Position",""))
        if ss=="N" or ps=="N": return 1
        return 0 if ps==ss else 2

    for label in PITCH_ORDER:
        if label not in by_label: continue
        slot_list=by_label[label]
        matched=[p for p in players if p["_key"] not in assigned and any(first_tok_fits(p,s) for s in slot_list)]
        if not matched:
            matched=[p for p in players if p["_key"] not in assigned
                     and not has_any_primary_slot(p)
                     and any(secondary_fits(p,s) for s in slot_list)]
        matched.sort(key=lambda p:-float(p.get("Minutes played") or 0))
        pt=set()
        for sl in slot_list: pt.update(sl.get("priority_toks",[]))
        if pt:
            matched.sort(key=lambda p:(0 if _tok(p.get("Position","")) in pt else 1,
                                       -float(p.get("Minutes played") or 0)))
        for p in matched: assigned.add(p["_key"])
        n=len(slot_list)
        if n==1:
            slot_map[slot_list[0]["id"]]=matched
        else:
            ordered=sorted(slot_list,key=lambda s:{"L":0,"N":1,"R":2}[s["side"]])
            starters=[]; used=set()
            for sl in ordered:
                best=None; best_sc=99
                for p in matched:
                    if id(p) in used: continue
                    sc=side_score(p,sl["side"])
                    if sc<best_sc: best_sc=sc; best=p
                if best: starters.append((sl["id"],best)); used.add(id(best))
            for sid,p in starters: slot_map[sid].append(p)
            depth_rem=[p for p in matched if id(p) not in used]
            for i,p in enumerate(depth_rem):
                slot_map[ordered[i%n]["id"]].append(p)

    # 4-back CB redistribution
    four_back={"4-2-3-1","4-3-3","4-4-2","4-1-4-1","3-4-3"}
    if formation_key in four_back:
        cb1_id=next((s["id"] for s in slots if s["id"]=="CB1"),None)
        cb2_id=next((s["id"] for s in slots if s["id"]=="CB2"),None)
        if cb1_id and cb2_id:
            all_cbs4=slot_map.get(cb1_id,[])+slot_map.get(cb2_id,[])
            all_cbs4.sort(key=lambda p:-float(p.get("Minutes played") or 0))
            lcb_p=[p for p in all_cbs4 if _tok(p.get("Position",""))=="LCB"]
            rcb_p=[p for p in all_cbs4 if _tok(p.get("Position",""))=="RCB"]
            cb_p=[p for p in all_cbs4 if _tok(p.get("Position",""))=="CB"]
            oth_p=[p for p in all_cbs4 if _tok(p.get("Position","")) not in {"CB","LCB","RCB"}]
            left=list(lcb_p); right=list(rcb_p)
            for i,p in enumerate(cb_p): (left if i%2==0 else right).append(p)
            for i,p in enumerate(oth_p): (left if i%2==0 else right).append(p)
            slot_map[cb1_id]=left; slot_map[cb2_id]=right

    # 3-back CB redistribution
    three_back={"3-5-2","3-4-3"}
    if formation_key in three_back:
        lcb_id=next((s["id"] for s in slots if s["id"]=="LCB"),None)
        cb_id=next((s["id"] for s in slots if s["id"]=="CB"),None)
        rcb_id=next((s["id"] for s in slots if s["id"]=="RCB"),None)
        if lcb_id and cb_id and rcb_id:
            all_cbs=[]
            for sid in (lcb_id,cb_id,rcb_id): all_cbs.extend(slot_map.get(sid,[]))
            all_cbs.sort(key=lambda p:-float(p.get("Minutes played") or 0))
            pure_cb=[p for p in all_cbs if _tok(p.get("Position",""))=="CB"]
            pure_lcb=[p for p in all_cbs if _tok(p.get("Position",""))=="LCB"]
            pure_rcb=[p for p in all_cbs if _tok(p.get("Position",""))=="RCB"]
            other=[p for p in all_cbs if _tok(p.get("Position","")) not in {"CB","LCB","RCB"}]
            if pure_cb:
                slot_map[lcb_id]=pure_lcb or other; slot_map[cb_id]=pure_cb; slot_map[rcb_id]=pure_rcb
            else:
                remaining=sorted([p for p in all_cbs if p not in pure_lcb],key=lambda p:-float(p.get("Minutes played") or 0))
                slot_map[lcb_id]=pure_lcb or []
                slot_map[cb_id]=[p for i,p in enumerate(remaining) if i%2==0]
                slot_map[rcb_id]=[p for i,p in enumerate(remaining) if i%2==1]

    # Fallback pass
    by_label_id={s["label"]:[] for s in slots}
    for s in slots: by_label_id[s["label"]].append(s["id"])
    remaining=[p for p in players if p["_key"] not in assigned]
    remaining.sort(key=lambda p:-float(p.get("Minutes played") or 0))
    for p in remaining:
        tok=_tok(p.get("Position","")); placed=False
        for try_label in FALLBACK_CANON.get(tok,[tok]):
            if try_label in by_label_id:
                best_sid=min(by_label_id[try_label],key=lambda sid:len(slot_map.get(sid,[])))
                slot_map.setdefault(best_sid,[]).append(p); assigned.add(p["_key"]); placed=True; break
        if not placed:
            best_sid=min((s["id"] for s in slots),key=lambda sid:len(slot_map.get(sid,[])))
            slot_map.setdefault(best_sid,[]).append(p); assigned.add(p["_key"])

    for sid,ps in slot_map.items():
        slot_def=next((s for s in slots if s["id"]==sid),None)
        for p in ps:
            p["_oop"]=not primary_fits(p,slot_def) if slot_def else False
            p["_primary_pos"]=_tok(p.get("Position",""))
            native=slot_def.get("native_toks") if slot_def else None
            p["_show_pos"]=(p["_oop"] or (native is not None and p["_primary_pos"] not in native))

    depth=[p for p in players if p["_key"] not in assigned]
    depth.sort(key=lambda p:-float(p.get("Minutes played") or 0))
    return slot_map,depth

# Role score HTML
def all_roles_html(player,df_sc,fs="8px"):
    if df_sc is None or df_sc.empty: return ""
    rows=df_sc[df_sc["Player"]==player.get("Player","")]
    if rows.empty: return ""
    row=rows.iloc[0]; rk=_role_key(player.get("Position",""))
    scores={}
    for rn in ROLE_BUCKETS.get(rk,{}):
        v=row.get(f"_rs_{rn}",np.nan)
        try:
            fv=float(v)
            if not np.isnan(fv): scores[rn]=fv
        except: pass
    if not scores: return ""
    best=max(scores,key=scores.get); lines=[]
    for rn,sc in sorted(scores.items(),key=lambda x:-x[1]):
        sc_col=score_to_color(sc); is_b=rn==best
        name_col=sc_col if is_b else "#7a8494"
        lines.append(f'<div style="display:flex;justify-content:space-between;gap:4px;font-size:{fs};line-height:1.4;">'
                     f'<span style="color:{name_col};font-weight:{"700" if is_b else "400"};">{rn}</span>'
                     f'<span style="color:{sc_col};font-weight:{"700" if is_b else "400"};min-width:22px;text-align:right;">{int(sc)}</span></div>')
    return f'<div style="margin-top:2px;">{"".join(lines)}</div>'

def best_role_html(player,df_sc,fs="8px"):
    if df_sc is None or df_sc.empty: return ""
    rows=df_sc[df_sc["Player"]==player.get("Player","")]
    if rows.empty: return ""
    row=rows.iloc[0]; rk=_role_key(player.get("Position",""))
    scores={}
    for rn in ROLE_BUCKETS.get(rk,{}):
        v=row.get(f"_rs_{rn}",np.nan)
        try:
            fv=float(v)
            if not np.isnan(fv): scores[rn]=fv
        except: pass
    if not scores: return ""
    best=max(scores,key=scores.get); sc=scores[best]; sc_col=score_to_color(sc)
    return (f'<div style="display:flex;justify-content:space-between;gap:4px;font-size:{fs};line-height:1.4;margin-top:2px;">'
            f'<span style="color:#7a8494;">{best}</span>'
            f'<span style="color:{sc_col};font-weight:700;min-width:22px;text-align:right;">{int(sc)}</span></div>')

# ── Squad pitch renderer ──────────────────────────────────────────────────────
PORTRAIT_SVG_LINES="""
  <rect  x="2"   y="2"     width="96" height="138" fill="none" stroke="#9ca3af" stroke-width="1.2" opacity=".18"/>
  <line  x1="2"  y1="71"   x2="98"   y2="71"      stroke="#9ca3af" stroke-width=".8"  opacity=".18"/>
  <circle cx="50" cy="71" r="10"                   fill="none" stroke="#9ca3af" stroke-width=".8"  opacity=".18"/>
  <circle cx="50" cy="71" r="1.2"                  fill="#9ca3af" opacity=".18"/>
  <rect  x="22"  y="2"     width="56" height="18"  fill="none" stroke="#9ca3af" stroke-width=".8"  opacity=".18"/>
  <rect  x="36"  y="2"     width="28" height="7"   fill="none" stroke="#9ca3af" stroke-width=".6"  opacity=".18"/>
  <circle cx="50" cy="14" r=".9"                   fill="#9ca3af" opacity=".18"/>
  <rect  x="22"  y="122"   width="56" height="18"  fill="none" stroke="#9ca3af" stroke-width=".8"  opacity=".18"/>
  <rect  x="36"  y="133"   width="28" height="7"   fill="none" stroke="#9ca3af" stroke-width=".6"  opacity=".18"/>
  <circle cx="50" cy="126" r=".9"                  fill="#9ca3af" opacity=".18"/>"""

def render_squad_pitch(team,league,formation,slots,slot_map,depth,df_sc,
                       show_mins=True,show_goals=True,show_assists=True,
                       show_roles=True,xi_only=False,best_role_only=False,
                       show_contracts=True):
    BG="#0a0f1c"; bsz="15px"; nsz="14px"; ssz="9px"; rsz="8px"

    def make_node(slot):
        ps_all=slot_map.get(slot["id"],[])
        ps=ps_all[:1] if xi_only else ps_all
        badge=(f'<div style="display:inline-block;padding:2px 8px;border:2px solid #ef4444;'
               f'color:#ef4444;font-size:{bsz};font-weight:900;letter-spacing:.1em;'
               f'margin-bottom:3px;background:rgba(10,15,28,.97);">{slot["label"]}</div>')
        rows=""
        for i,p in enumerate(ps):
            yrs=contract_years(p.get("Contract expires",""))
            yr_str=f"+{yrs}" if yrs>=0 else "+?"
            loan=is_loan(p); fw="800" if i==0 else "500"
            _lo=is_loaned_out(p); _yt=is_youth(p)
            col=player_css_color(yrs,loan,_lo,_yt)
            multi=" \U0001f501" if _multi_role(p.get("Position","")) else ""
            oop_s=f" ({p['_primary_pos']})" if p.get('_show_pos') else ''
            suffix=(f" L{oop_s}{multi}" if loan else f"{(yr_str if show_contracts else '')}{oop_s}{multi}")
            stat_parts=[]
            if show_mins: stat_parts.append(f"{int(float(p.get('Minutes played') or 0))}\u2032")
            if show_goals:
                g=float(p.get("Goals") or 0)
                if g>0: stat_parts.append(f"{int(g)}\u26bd")
            if show_assists:
                a=float(p.get("Assists") or 0)
                if a>0: stat_parts.append(f"{int(a)}\U0001f170")
            stat_html=(f'<div style="color:#fff;font-size:{ssz};line-height:1.2;opacity:.9;">{" ".join(stat_parts)}</div>'
                      ) if stat_parts else ""
            rs_html=(best_role_html(p,df_sc,rsz) if (show_roles and (best_role_only or i>0))
                     else all_roles_html(p,df_sc,rsz) if (i==0 and show_roles) else "")
            mt="margin-top:5px;" if i>0 else ""
            rows+=(f'<div style="color:{col};font-size:{nsz};line-height:1.45;font-weight:{fw};{mt}'
                   f'white-space:nowrap;text-shadow:0 0 8px rgba(0,0,0,1);">'
                   f'{p["Player"]} {suffix}</div>{stat_html}{rs_html}')
        if not ps: rows=f'<div style="color:#1f2937;font-size:{ssz};">&#8212;</div>'
        sx=float(slot.get("x",50)); mxw="115px" if (sx<20 or sx>80) else "none"
        return (f'<div style="position:absolute;left:{slot["x"]}%;top:{slot["y"]}%;'
                f'transform:translate(-50%,-50%);text-align:center;'
                f'min-width:80px;max-width:{mxw};z-index:10;">'
                f'{badge}<div>{rows}</div></div>')

    nodes="".join(make_node(s) for s in slots)
    portrait_svg=(f'<svg style="position:absolute;inset:0;width:100%;height:100%;'
                  f'pointer-events:none;z-index:1;" viewBox="0 0 100 142" preserveAspectRatio="none">'
                  +PORTRAIT_SVG_LINES+'</svg>')

    depth_html=""
    if not xi_only and depth:
        cards=""
        for p in depth:
            yrs=contract_years(p.get("Contract expires","")); yr_str=f"+{yrs}" if yrs>=0 else "+?"
            loan=is_loan(p); _lo=is_loaned_out(p); _yt=is_youth(p)
            col=player_css_color(yrs,loan,_lo,_yt)
            multi="\U0001f501" if _multi_role(p.get("Position","")) else ""
            pos_t=_tok(p.get("Position",""))
            br=best_role_html(p,df_sc,"8px") if show_roles else ""
            dep_yr="L" if loan else (f"+{yrs}" if yrs>=0 else "+?")
            cards+=(f'<div style="background:#0d1220;border:1px solid #1f2937;'
                    f'padding:5px 9px;min-width:100px;text-align:center;flex-shrink:0;">'
                    f'<div style="color:{col};font-size:11px;font-weight:700;">'
                    f'{p["Player"]} {dep_yr} {multi}</div>'
                    f'<div style="color:#6b7280;font-size:7px;">{pos_t}</div>{br}</div>')
        depth_html=(f'<div style="margin-top:10px;border-top:1px solid #1f2937;padding-top:8px;">'
                    f'<div style="font-size:9px;font-weight:800;letter-spacing:.18em;color:#6b7280;'
                    f'margin-bottom:6px;text-align:center;">DEPTH</div>'
                    f'<div style="display:flex;flex-wrap:wrap;gap:6px;justify-content:center;">{cards}</div></div>')

    legend=(f'<div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;'
            f'font-size:9px;font-weight:700;margin-top:6px;">'
            f'<span style="color:#fff;">Contracted</span>'
            f'<span style="color:#f59e0b;">Final Year</span>'
            f'<span style="color:#ef4444;">Out of Contract</span>'
            f'<span style="color:#22c55e;">On Loan</span>'
            f'<span style="color:#eab308;">Loaned Out</span>'
            f'<span style="color:#9ca3af;">Youth</span></div>')

    return (f'<div style="font-family:Montserrat,sans-serif;color:#fff;background:{BG};padding:0 4px 10px;">'
            f'<div style="position:relative;background:{BG};padding-bottom:142%;'
            f'overflow:hidden;border:1px solid #1a2540;">'
            f'{portrait_svg}{nodes}</div>'
            f'{depth_html}{legend}</div>')

# ── Badge loaders ─────────────────────────────────────────────────────────────
# Tries your repo filenames first, then common alternatives, then falls back silently.

_FOTMOB_URLS = {}
for _badges_mod in ("badges", "team_fotmob_urls", "fotmob_urls"):
    try:
        _m = __import__(_badges_mod)
        # badges.py may expose FOTMOB_TEAM_URLS or TEAM_URLS or a dict at module level
        _FOTMOB_URLS = (
            getattr(_m, "FOTMOB_TEAM_URLS", None)
            or getattr(_m, "TEAM_URLS", None)
            or getattr(_m, "BADGES", None)
            or {}
        )
        break
    except Exception:
        pass

_get_league_logo_url = lambda lg: ""
for _logos_mod in ("leaguelogos", "league_logo_urls", "league_logos"):
    try:
        _m = __import__(_logos_mod)
        if hasattr(_m, "get_league_logo_url"):
            _get_league_logo_url = _m.get_league_logo_url
        elif hasattr(_m, "LEAGUE_LOGO_URLS"):
            import unicodedata as _ud
            def _norm_key(s):
                s = _ud.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
                return " ".join(s.strip().lower().split())
            _LL = {_norm_key(k): str(v).strip() for k, v in _m.LEAGUE_LOGO_URLS.items() if str(v).strip()}
            _get_league_logo_url = lambda lg, _ll=_LL: _ll.get(_norm_key(lg), "")
        break
    except Exception:
        pass

@st.cache_data(show_spinner=False)
def load_remote_img(url):
    try:
        r=requests.get(url,timeout=8); r.raise_for_status()
        return plt.imread(io.BytesIO(r.content))
    except: return None

def fotmob_crest_url(team):
    raw = (_FOTMOB_URLS.get(team) or "").strip()
    if not raw: return ""
    # Already a direct image URL (e.g. from badges.py storing .png links directly)
    if raw.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".svg")):
        return raw
    # Fotmob team page URL — extract team ID
    m = re.search(r"/teams/(\d+)/", raw)
    if m: return f"https://images.fotmob.com/image_resources/logo/teamlogo/{m.group(1)}.png"
    # Fotmob image URL with teamlogo path already
    if "teamlogo" in raw or "fotmob.com" in raw:
        return raw
    return ""

@st.cache_data(show_spinner=False)
def get_team_badge(team):
    url=fotmob_crest_url(team)
    if url:
        img=load_remote_img(url)
        if img is not None: return img
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap');
html,body,[class*="css"]{font-family:'Montserrat',sans-serif!important;}
.stApp{background:#0a0f1c!important;}
section[data-testid="stSidebar"]{background:#060a14!important;border-right:1px solid #0d1220!important;}
section[data-testid="stSidebar"] *{color:#fff!important;}
section[data-testid="stSidebar"] input,section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea{background:#0d1424!important;border:1px solid #1e2d4a!important;color:#fff!important;}
.stSelectbox>div>div{background:#0d1424!important;border:1px solid #1e2d4a!important;}
div[data-baseweb="select"]*{background:#0d1424!important;color:#fff!important;}
div[data-baseweb="popover"]*{background:#0d1424!important;color:#fff!important;}
.stTextInput>div>div>input,.stNumberInput input{background:#0d1424!important;border:1px solid #1e2d4a!important;color:#fff!important;}
.stButton>button{background:#ffffff!important;color:#000000!important;font-weight:700!important;
  border:none!important;font-family:'Montserrat',sans-serif!important;border-radius:2px!important;}
.stDownloadButton>button{background:#ffffff!important;color:#000000!important;font-weight:700!important;
  border:none!important;font-family:'Montserrat',sans-serif!important;border-radius:2px!important;}
label{color:#6b7280!important;font-size:9px!important;letter-spacing:.12em!important;text-transform:uppercase!important;}
h1,h2,h3{color:#fff!important;font-family:'Montserrat',sans-serif!important;}
footer{display:none!important;}
.score-chip{display:inline-flex;align-items:center;justify-content:center;
  padding:4px 12px;border-radius:6px;font-weight:900;font-size:22px;min-width:52px;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚽ TEAM HQ + SQUAD")
    st.markdown("---")
    st.markdown("**TEAM STATS CSV**")

    # Auto-detect CSVs in working directory
    csv_candidates=sorted(Path.cwd().glob("*.csv"),key=lambda c:c.name)
    if csv_candidates:
        _csv_names=[c.name for c in csv_candidates]
        _world=[n for n in _csv_names if n.upper().startswith("WORLD")]
        _def=_csv_names.index(_world[0]) if _world else 0
        _team_csv_choice=st.selectbox("Team Stats CSV:",_csv_names,index=_def,key="sb_teamcsv")
        df_team_raw=_read_team_path(str(Path.cwd()/_team_csv_choice))
    else:
        _up_team=st.file_uploader("Upload Team Stats CSV",type=["csv"],key="sb_teamup")
        if _up_team is None:
            st.info("Upload team stats CSV to begin.")
            st.stop()
        df_team_raw=_read_team_csv(_up_team.getvalue())

    st.markdown("**PLAYER CSV (for squad)**")

    # ── Flexible player CSV column normalisation ──────────────────────────────
    # Wyscout exports use various column name styles. This maps them all to the
    # standard names the squad depth chart expects.
    PLAYER_COL_MAP = {
        # Standard name : list of possible raw names (lowercase, stripped)
        "Player":         ["player","full name","name","player name","player_name"],
        "Team":           ["team","club","team name","club name"],
        "League":         ["league","competition","league name"],
        "Position":       ["position","positions","primary position","pos","position(s)"],
        "Minutes played": ["minutes played","minutes","mins","min","minutes_played","mins played"],
        "Goals":          ["goals","goals scored","goal"],
        "Assists":        ["assists","assist"],
        "Age":            ["age","player age"],
        "xG":             ["xg","expected goals","xgoals"],
        "xA":             ["xa","expected assists","xassists"],
        "Market value":   ["market value","value","transfer value","market_value"],
        "Contract expires":["contract expires","contract expiry","contract","contract end",
                            "contract_expires","expiry date","expires"],
        "On loan":        ["on loan","loan","on_loan","loaned in"],
        "Loaned Out":     ["loaned out","loan out","loaned_out"],
        "Youth Player":   ["youth player","youth","academy","youth_player"],
        "Foot":           ["foot","preferred foot"],
        "Height":         ["height"],
        "Birth country":  ["birth country","nationality","country","birth_country"],
    }

    def normalise_player_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip()
        col_lower = {c.lower().strip(): c for c in df.columns}
        rename = {}
        for standard, aliases in PLAYER_COL_MAP.items():
            if standard in df.columns:
                continue  # already correct
            for alias in aliases:
                if alias in col_lower:
                    rename[col_lower[alias]] = standard
                    break
        df = df.rename(columns=rename)

        # Strip whitespace from key string columns
        for c in ["Player", "Team", "Position", "League"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

        # Coerce numeric columns
        for c in ["Minutes played", "Goals", "Assists", "Age", "xG", "xA"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        # Require Player and Position — raise a clear error if still missing
        missing = [c for c in ["Player", "Position"] if c not in df.columns]
        if missing:
            available = ", ".join(df.columns.tolist())
            raise KeyError(
                f"Could not find columns {missing} in player CSV. "
                f"Available columns: {available}"
            )

        # Ensure Team and League exist (can be empty if single-team upload)
        for c in ["Team", "League"]:
            if c not in df.columns:
                df[c] = ""

        df["_ftok"] = df["Position"].apply(_tok)
        df["_key"]  = df["Player"]
        return df

    @st.cache_data(show_spinner=False)
    def _load_player_path(path):
        df = pd.read_csv(path)
        return normalise_player_df(df)

    @st.cache_data(show_spinner=False)
    def _load_player_bytes(data):
        df = pd.read_csv(io.BytesIO(data))
        return normalise_player_df(df)

    _player_csvs=[c.name for c in csv_candidates]
    if _player_csvs:
        _def_p=0
        for i,n in enumerate(_player_csvs):
            if "player" in n.lower() or "squad" in n.lower() or "efl" in n.lower():
                _def_p=i; break
        _player_csv_choice=st.selectbox("Player CSV:",_player_csvs,index=_def_p,key="sb_playercsv")

        if st.session_state.get("_player_src")!=_player_csv_choice:
            st.session_state["_player_df"]=None
            st.session_state["_player_df_sc"]=None
            st.session_state["_player_src"]=_player_csv_choice
        if st.session_state.get("_player_df") is None:
            with st.spinner("Loading player data…"):
                try:
                    st.session_state["_player_df"]=_load_player_path(str(Path.cwd()/_player_csv_choice))
                except KeyError as e:
                    st.error(f"Player CSV column error: {e}")
                    st.stop()
            st.session_state["_player_df_sc"]=None
        if st.session_state.get("_player_df_sc") is None:
            with st.spinner("Computing role scores…"):
                st.session_state["_player_df_sc"]=compute_role_scores(st.session_state["_player_df"])
        df_players=st.session_state["_player_df"]
        df_players_sc=st.session_state["_player_df_sc"]
    else:
        _up_player=st.file_uploader("Upload Player CSV",type=["csv"],key="sb_playerup")
        if _up_player:
            try:
                df_players=_load_player_bytes(_up_player.getvalue())
            except KeyError as e:
                st.error(f"Player CSV column error: {e}")
                st.stop()
            if st.session_state.get("_player_df_sc") is None:
                with st.spinner("Computing role scores…"):
                    df_players_sc=compute_role_scores(df_players)
                    st.session_state["_player_df_sc"]=df_players_sc
            else:
                df_players_sc=st.session_state["_player_df_sc"]
        else:
            df_players=None; df_players_sc=None

    st.markdown("---")
    st.markdown("**DISPLAY OPTIONS**")
    show_mins   =st.toggle("Minutes played",True,key="sq_mins")
    show_goals  =st.toggle("Goals",True,key="sq_goals")
    show_assists=st.toggle("Assists",True,key="sq_assists")
    show_roles  =st.toggle("Role scores",True,key="sq_roles")
    best_role_only=st.toggle("Best role only",False,key="sq_bestonly")
    xi_only     =st.toggle("XI only",False,key="sq_xionly")
    show_contracts=st.toggle("Show contracts",True,key="sq_contracts")
    st.markdown("---")
    min_matches =st.slider("Min team matches played",0,80,5,key="sb_minmatches")

# ═══════════════════════════════════════════════════════════════════════════════
# NORMALISE TEAM DATA
# ═══════════════════════════════════════════════════════════════════════════════
df_team_raw=normalise_cols(df_team_raw)
for c in NUMERIC_COLS:
    if c in df_team_raw.columns:
        df_team_raw[c]=pd.to_numeric(df_team_raw[c],errors="coerce")
if "Matches" in df_team_raw.columns:
    df_team_raw=df_team_raw[pd.to_numeric(df_team_raw["Matches"],errors="coerce").fillna(0)>=min_matches]

# ── Percentile ranks ──────────────────────────────────────────────────────────
df_team=df_team_raw.copy()
def score_col(name): return f"_pct_{name}"

for col in NUMERIC_COLS:
    if col not in df_team.columns: continue
    inv=col in INVERT_METRICS
    df_team[f"_pct_{col}"]=df_team.groupby("League")[col].transform(lambda s,i=inv: pct_rank(s,i))

def compute_overall(row):
    ep=row.get(score_col("Expected Points"),np.nan)
    xg=row.get(score_col("xG p90"),np.nan)
    xga=row.get(score_col("xG Against p90"),np.nan)
    vals=[v for v in [ep,xg,xga] if pd.notna(v)]
    if not vals: return np.nan
    w=[0.5,0.25,0.25][:len(vals)]; tw=sum(w)
    return sum(v*ww for v,ww in zip(vals,w))/tw

def compute_attack(row):
    weights=[(row.get(score_col("xG p90"),np.nan),0.5),(row.get(score_col("Goals p90"),np.nan),0.3),
             (row.get(score_col("Shots p90"),np.nan),0.05),(row.get(score_col("Touches in Box p90"),np.nan),0.15)]
    vals=[(v,w) for v,w in weights if pd.notna(v)]
    if not vals: return np.nan
    tw=sum(w for _,w in vals); return sum(v*w for v,w in vals)/tw

def compute_defense(row):
    weights=[(row.get(score_col("xG Against p90"),np.nan),0.5),(row.get(score_col("Goals Against p90"),np.nan),0.3),
             (row.get(score_col("Shots Against p90"),np.nan),0.2)]
    vals=[(v,w) for v,w in weights if pd.notna(v)]
    if not vals: return np.nan
    tw=sum(w for _,w in vals); return sum(v*w for v,w in vals)/tw

def compute_possession(row):
    weights=[(row.get(score_col("Possession %"),np.nan),0.35),(row.get(score_col("Passes p90"),np.nan),0.30),
             (row.get(score_col("Pass Accuracy %"),np.nan),0.10),(row.get(score_col("Passes to Final Third p90"),np.nan),0.25)]
    vals=[(v,w) for v,w in weights if pd.notna(v)]
    if not vals: return np.nan
    tw=sum(w for _,w in vals); return sum(v*w for v,w in vals)/tw

df_team["OVR"]=df_team.apply(compute_overall,axis=1)
df_team["ATT"]=df_team.apply(compute_attack,axis=1)
df_team["DEF"]=df_team.apply(compute_defense,axis=1)
df_team["POS"]=df_team.apply(compute_possession,axis=1)

if df_team.empty:
    st.warning("No teams in data after filters."); st.stop()

# ── Team selector ─────────────────────────────────────────────────────────────
all_leagues=sorted(df_team["League"].dropna().unique()) if "League" in df_team.columns else []

with st.sidebar:
    st.markdown("---")
    if all_leagues:
        sel_league=st.selectbox("League",["All"]+all_leagues,key="sb_league")
    else:
        sel_league="All"

if sel_league=="All":
    _team_pool=df_team
else:
    _team_pool=df_team[df_team["League"]==sel_league]

team_options=sorted(_team_pool["Team"].dropna().unique()) if "Team" in _team_pool.columns else []

with st.sidebar:
    if not team_options:
        st.warning("No teams available."); st.stop()
    sel_team=st.selectbox("Select Team",team_options,key="sb_selteam")
    st.markdown("---")
    st.markdown("**SQUAD FORMATION**")
    formation=st.selectbox("Formation",list(FORMATIONS.keys()),key="sb_formation")

# ─────────────────────────────────────────────────────────────────────────────
# GET TEAM ROW
# ─────────────────────────────────────────────────────────────────────────────
team_df_rows=df_team[df_team["Team"]==sel_team]
if team_df_rows.empty: st.warning(f"'{sel_team}' not found."); st.stop()
t_row=team_df_rows.iloc[0]
t_league=str(t_row["League"]) if "League" in t_row.index else ""
t_country=league_country(t_league)
t_region=league_region(t_league)

ovr=t_row.get("OVR",np.nan); att=t_row.get("ATT",np.nan)
defv=t_row.get("DEF",np.nan); pos=t_row.get("POS",np.nan)

def _s(v):
    try: return int(round(float(v)))
    except: return "—"

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER — Team Name, League, Scores
# ═══════════════════════════════════════════════════════════════════════════════
import base64

flag=flag_html(t_league)
badge_img=get_team_badge(sel_team)

def _img_to_b64(img_array):
    buf=io.BytesIO(); plt.imsave(buf,img_array,format="png")
    return base64.b64encode(buf.getvalue()).decode()

# ── Club badge HTML ───────────────────────────────────────────────────────────
if badge_img is not None:
    badge_html_header=(f'<img src="data:image/png;base64,{_img_to_b64(badge_img)}" '
                       f'style="width:80px;height:80px;object-fit:contain;border-radius:8px;"/>')
else:
    _raw_url=fotmob_crest_url(sel_team)
    if _raw_url:
        badge_html_header=(f'<img src="{_raw_url}" '
                           f'style="width:80px;height:80px;object-fit:contain;border-radius:8px;" '
                           f'onerror="this.style.display=\'none\'"/>')
    else:
        badge_html_header=('<div style="width:80px;height:80px;background:#111827;border-radius:8px;'
                           'display:flex;align-items:center;justify-content:center;font-size:32px;">🏟️</div>')

# ── League logo HTML ──────────────────────────────────────────────────────────
_league_logo_url=_get_league_logo_url(t_league)
league_logo_html=(f'<img src="{_league_logo_url}" '
                  f'style="height:36px;width:36px;object-fit:contain;vertical-align:middle;margin-right:8px;" '
                  f'onerror="this.style.display=\'none\'"/>') if _league_logo_url else ""

# Build header HTML
def score_chip(label,val):
    bg=rating_color(val) if not (isinstance(val,float) and np.isnan(val)) else "#1a2035"
    fg="#000" if not (isinstance(val,float) and np.isnan(val)) and float(val)>=44 else "#fff"
    v_str=fmt2(val) if not (isinstance(val,float) and np.isnan(val)) else "—"
    return (f'<div style="display:flex;flex-direction:column;align-items:center;gap:2px;">'
            f'<div style="font-size:10px;color:#9ca3af;font-weight:700;letter-spacing:.1em;text-transform:uppercase;">{label}</div>'
            f'<div style="background:{bg};color:{fg};font-size:26px;font-weight:900;'
            f'padding:6px 16px;border-radius:8px;min-width:60px;text-align:center;">{v_str}</div>'
            f'</div>')

st.markdown(f"""
<div style="background:#0f1628;border:1px solid #1e2d4a;border-radius:16px;
            padding:24px 28px;margin-bottom:24px;display:flex;align-items:center;gap:24px;
            flex-wrap:wrap;">

  <!-- Club badge -->
  <div style="flex-shrink:0;">{badge_html_header}</div>

  <!-- Name + league meta -->
  <div style="flex:1;min-width:200px;">
    <div style="font-family:Montserrat,sans-serif;font-size:34px;font-weight:900;
                color:#fff;letter-spacing:.03em;line-height:1.1;">{sel_team.upper()}</div>
    <div style="margin-top:8px;display:flex;align-items:center;font-size:14px;
                color:#9ca3af;font-weight:600;gap:6px;flex-wrap:wrap;">
      {league_logo_html}{flag}
      <span>{t_league}</span>
      <span style="color:#374151;">&nbsp;·&nbsp;</span>
      <span>{t_country}</span>
      <span style="color:#374151;">&nbsp;·&nbsp;</span>
      <span>{t_region}</span>
    </div>
  </div>

  <!-- Score chips -->
  <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end;">
    {score_chip("Overall",ovr)}
    {score_chip("Attack",att)}
    {score_chip("Defense",defv)}
    {score_chip("Possession",pos)}
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE Y — POLAR RADAR
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="font-size:11px;font-weight:900;letter-spacing:.18em;color:#6b7280;'
            'text-transform:uppercase;margin-bottom:10px;">Squad Performance Radar</div>',
            unsafe_allow_html=True)

pool_y=df_team[df_team["League"]==t_league]
METRICS_Y=["xG p90","Goals p90","Touches in Box p90","xG Against p90","Goals Against p90",
           "PPDA","Possession %","Passes p90","Passes to Final Third p90","Long Passes p90",
           "Points","Expected Points"]
metrics_y=[m for m in METRICS_Y if m in df_team.columns]

def pct_y(col):
    inv=col in INVERT_METRICS
    if col not in pool_y.columns: return 50
    s=pd.to_numeric(pool_y[col],errors="coerce").dropna()
    v=float(t_row[col]) if pd.notna(t_row.get(col)) else np.nan
    if pd.isna(v) or s.empty: return 50
    p=(s<v).mean()*100+(s==v).mean()*50
    return float(np.clip((100-p) if inv else p,0,100))

pcts_y=[pct_y(m) for m in metrics_y]
labels_y=[mlabel(m) for m in metrics_y]
N_y=len(metrics_y)

color_scale_y=["#be2a3e","#e25f48","#f88f4d","#f4d166","#90b960","#4b9b5f","#22763f"]
cmap_y=LinearSegmentedColormap.from_list("csy",color_scale_y)
bar_colors_y=[cmap_y(p/100) for p in pcts_y]

angles_y=np.linspace(0,2*np.pi,N_y,endpoint=False)[::-1]
rot_shift_y=np.deg2rad(75)-angles_y[0]
rot_angles_y=[(a+rot_shift_y)%(2*np.pi) for a in angles_y]
bar_w_y=(2*np.pi/N_y)*0.85

col_radar, col_info = st.columns([1, 1])

with col_radar:
    fig_y=plt.figure(figsize=(7,6))
    fig_y.patch.set_facecolor("#0a0f1c")
    ax_y=fig_y.add_axes([0.05,0.05,0.9,0.85],polar=True)
    ax_y.set_facecolor("#0a0f1c"); ax_y.set_rlim(0,100)

    for i in range(N_y):
        ax_y.bar(rot_angles_y[i],100,width=bar_w_y,color="#444",edgecolor="none",zorder=0)
    for i,p in enumerate(pcts_y):
        ax_y.bar(rot_angles_y[i],p,width=bar_w_y,color=bar_colors_y[i],edgecolor="white",linewidth=1.5,zorder=2)
        if p>=20:
            lp=p-10 if p>=30 else p*0.7
            ax_y.text(rot_angles_y[i],lp,f"{int(round(p))}",ha='center',va='center',
                      fontsize=11,weight='bold',color='white',zorder=3)
    for i in range(N_y):
        sep=(rot_angles_y[i]-bar_w_y/2)%(2*np.pi)
        is_cross=any(np.isclose(sep,a,atol=0.01) for a in [0,np.pi/2,np.pi,3*np.pi/2])
        ax_y.plot([sep,sep],[0,100],color=(1,1,1,1.0) if is_cross else (1,1,1,0.25),
                  linewidth=1.8 if is_cross else 1,zorder=4)
    for rp in [90,75,50,25]:
        theta_ref=np.linspace(0,2*np.pi,500)
        ax_y.plot(theta_ref,[rp]*500,linestyle="dotted",lw=1.2,color="lightgrey",zorder=1)
    for i,lab in enumerate(labels_y):
        ax_y.text(rot_angles_y[i],145,lab.upper(),ha='center',va='center',
                  fontsize=9,weight='bold',color='white',zorder=5)
    ax_y.set_xticks([]); ax_y.set_yticks([])
    ax_y.spines['polar'].set_visible(False); ax_y.grid(False)

    st.pyplot(fig_y,use_container_width=True)
    buf_y=io.BytesIO()
    fig_y.savefig(buf_y,format="png",dpi=200,bbox_inches='tight',facecolor="#0a0f1c")
    st.download_button("⬇️ Download Radar",buf_y.getvalue(),
                       f"{sel_team.replace(' ','_')}_radar.png","image/png")
    plt.close(fig_y)

# ═══════════════════════════════════════════════════════════════════════════════
# STYLE / STRENGTHS / WEAKNESSES  (right column alongside radar)
# ═══════════════════════════════════════════════════════════════════════════════
def _op_pct(col,invert=False):
    if col not in pool_y.columns: return 0.0
    s=pd.to_numeric(pool_y[col],errors="coerce").dropna()
    v=float(t_row[col]) if pd.notna(t_row.get(col)) else np.nan
    if pd.isna(v) or s.empty: return 0.0
    p=(s<v).mean()*100+(s==v).mean()*50
    return float(np.clip((100-p) if invert else p,0,100))

STYLE_TEAM_MAP={
    "Crosses p90":{"style":"Create Chances via Crosses"},
    "Goals p90":{"style":"Attacking","sw":"Scoring Goals","sw_weak":"Scoring Goals"},
    "xG p90":{"sw":"Chance Creation","sw_weak":"Chance Creation"},
    "Shots p90":{"sw":"Shot Volume","sw_weak":"Shot Volume"},
    "Touches in Box p90":{"style":"Effective Attacking Sequences","sw":"Penalty Box Entries","sw_weak":"Penalty Box Entries"},
    "Goals Against p90":{"style":"Solid Defensive Structure","sw":"Preventing Goals","sw_weak":"Conceding Goals"},
    "xG Against p90":{"style":"Chance Prevention","sw":"Preventing Chances","sw_weak":"Conceding Chances"},
    "Aerial Duels p90":{"style":"High Balls"},
    "Aerial Duels Won %":{"sw":"Aerial Duels","sw_weak":"Aerial Duels"},
    "Defensive Duels p90":{"style":"Duel Heavy"},
    "Defensive Duels Won %":{"sw":"Defensive Duels","sw_weak":"Defensive Duels"},
    "Shots Against p90":{"sw":"Limiting Opposition Shots","sw_weak":"Conceding Many Shots"},
    "PPDA":{"style":"Press Intense Out of Possession","sw":"Pressing","sw_weak":"Pressing"},
    "Dribbles p90":{"style":"Break Lines via Carries"},
    "Possession %":{"style":"Control Games with the Ball","sw":"Game Control","sw_weak":"Game Control"},
    "Passes p90":{"style":"Build Up via Passing Sequences"},
    "Pass Accuracy %":{"sw":"Ball Retention","sw_weak":"Ball Retention"},
    "Long Passes p90":{"style":"Direct Build Up"},
    "Long Pass Accuracy %":{"style":"Calculated Vertical Build Up"},
    "Passes to Final Third p90":{"sw":"Final 3rd Entries","sw_weak":"Final 3rd Entries"},
    "Progressive Passes p90":{"sw":"Passing Progression","sw_weak":"Passing Progression"},
    "Progressive Runs p90":{"sw":"Ball Carriers","sw_weak":"Ball Carriers"},
}
HI,LO,STYLE_T=70,35,65
strengths,weaknesses,styles=[],[],[]
for m,cfg in STYLE_TEAM_MAP.items():
    if m not in df_team.columns: continue
    p=_op_pct(m,m in INVERT_METRICS)
    if cfg.get("sw") and p>=HI: strengths.append(cfg["sw"])
    if cfg.get("sw_weak") and p<=LO: weaknesses.append(cfg["sw_weak"])
    if cfg.get("style") and p>=STYLE_T: styles.append(cfg["style"])

def chips(items,bg,label):
    if not items:
        return f'<div style="font-size:12px;color:#6b7280;margin-bottom:4px;">None identified</div>'
    spans="".join(
        f'<span style="background:{bg};color:#fff;padding:4px 10px;border-radius:8px;'
        f'margin:0 5px 5px 0;display:inline-block;font-size:12px;font-weight:700;">{t}</span>'
        for t in list(dict.fromkeys(items))[:8])
    return f'<div style="margin-bottom:8px;">{spans}</div>'

# Key stats
_stats_html=""
for lbl,col,inv in [("xG",    "xG p90",False),("xGA","xG Against p90",True),
                    ("Poss",  "Possession %",False),("PPDA","PPDA",True),
                    ("Passes","Passes p90",False),("Pts", "Points",False)]:
    if col in t_row.index and pd.notna(t_row.get(col)):
        v=float(t_row[col])
        pct=_op_pct(col,inv)
        _stats_html+=(f'<div style="display:flex;justify-content:space-between;align-items:center;'
                      f'padding:5px 8px;border-bottom:1px solid #1e2d4a;">'
                      f'<span style="color:#9ca3af;font-size:12px;font-weight:600;">{lbl}</span>'
                      f'<span style="color:#fff;font-size:13px;font-weight:700;">{v:.2f}</span>'
                      f'<span style="background:{rating_color(pct)};color:#000;font-size:11px;font-weight:900;'
                      f'padding:2px 7px;border-radius:5px;">{int(pct)}</span></div>')

# League position
def _lg_pos(row,df_all,metric,asc=False):
    lg=row.get("League","")
    if not lg or metric not in df_all.columns: return None,None
    lg_df=df_all[df_all["League"]==lg].dropna(subset=[metric])
    n=len(lg_df)
    if n==0: return None,None
    sdf=lg_df.sort_values(metric,ascending=asc).reset_index(drop=True)
    team=row.get("Team","")
    m=sdf[sdf["Team"]==team]
    if m.empty: return None,n
    return m.index[0]+1,n

_pts_pos,_pts_n=_lg_pos(t_row,df_team,"Points",asc=False)
_xpts_pos,_xpts_n=_lg_pos(t_row,df_team,"Expected Points",asc=False)
_pts_str=f"{_pts_pos}/{_pts_n}" if _pts_pos is not None else "—"
_xpts_str=f"{_xpts_pos}/{_xpts_n}" if _xpts_pos is not None else "—"

with col_info:
    st.markdown(f"""
<div style="background:#0f1628;border:1px solid #1e2d4a;border-radius:12px;padding:18px 20px;height:100%;">
  <div style="font-size:11px;font-weight:900;letter-spacing:.15em;color:#6b7280;text-transform:uppercase;margin-bottom:12px;">
    Season Summary
  </div>
  <div style="font-size:13px;color:#9ca3af;margin-bottom:12px;">
    Matches: <b style="color:#fff;">{int(t_row['Matches']) if pd.notna(t_row.get('Matches')) else '—'}</b>
    &nbsp;·&nbsp; Pts: <b style="color:#fff;">{int(t_row['Points']) if pd.notna(t_row.get('Points')) else '—'}</b>
    ({_pts_str}) &nbsp;·&nbsp; xPts: <b style="color:#fff;">{float(t_row['Expected Points']):.1f if pd.notna(t_row.get('Expected Points')) else '—'}</b>
    ({_xpts_str})
  </div>
  {_stats_html}

  <div style="margin-top:14px;">
    <div style="font-size:10px;font-weight:900;letter-spacing:.14em;color:#6b7280;text-transform:uppercase;margin-bottom:6px;">Style</div>
    {chips(styles,"#3b82f6","Style")}
    <div style="font-size:10px;font-weight:900;letter-spacing:.14em;color:#6b7280;text-transform:uppercase;margin-bottom:6px;margin-top:8px;">Strengths</div>
    {chips(strengths,"#16a34a","Strengths")}
    <div style="font-size:10px;font-weight:900;letter-spacing:.14em;color:#6b7280;text-transform:uppercase;margin-bottom:6px;margin-top:8px;">Weaknesses</div>
    {chips(weaknesses,"#dc2626","Weaknesses")}
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SQUAD DEPTH CHART
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div style="font-size:11px;font-weight:900;letter-spacing:.18em;color:#6b7280;'
            'text-transform:uppercase;margin-bottom:10px;">Squad Depth</div>',
            unsafe_allow_html=True)

if df_players is None:
    st.info("Upload a player CSV in the sidebar to see the squad depth chart.")
else:
    # Find team in player data — try exact match, then fuzzy
    _team_players=df_players[df_players["Team"]==sel_team]
    if _team_players.empty:
        # Try partial match
        _tm=sel_team.lower()
        _match=df_players[df_players["Team"].str.lower().str.contains(_tm[:8],na=False)]
        if not _match.empty:
            _actual_team=_match["Team"].iloc[0]
            _team_players=df_players[df_players["Team"]==_actual_team]

    if _team_players.empty:
        st.info(f"No squad data found for **{sel_team}** in the player CSV. "
                f"Check that the Team name matches exactly.")
    else:
        with st.sidebar:
            st.markdown("---")
            sq_min_mins=st.slider("Min player minutes",0,3000,0,50,key="sq_minmins")

        _tp_filt=_team_players[_team_players["Minutes played"]>=sq_min_mins].copy()
        _tp_filt["_key"]=_tp_filt["Player"]
        players_list=_tp_filt.to_dict("records")

        # Build / cache slot assignment
        _cache_key=(sel_team,formation,sq_min_mins)
        if st.session_state.get("_squad_cache_key")!=_cache_key:
            _sm,_dep=assign_players(players_list,formation)
            st.session_state["_squad_slot_map"]=_sm
            st.session_state["_squad_depth"]=_dep
            st.session_state["_squad_cache_key"]=_cache_key

        slot_map_sq=st.session_state["_squad_slot_map"]
        depth_sq=st.session_state["_squad_depth"]
        slots_sq=FORMATIONS[formation]

        pitch_html=render_squad_pitch(
            sel_team,t_league,formation,slots_sq,slot_map_sq,depth_sq,df_players_sc,
            show_mins=show_mins,show_goals=show_goals,show_assists=show_assists,
            show_roles=show_roles,xi_only=xi_only,best_role_only=best_role_only,
            show_contracts=show_contracts
        )

        # Render pitch in a centered constrained column
        _pc1,_pc2,_pc3=st.columns([1,4,1])
        with _pc2:
            st.markdown(pitch_html,unsafe_allow_html=True)

        # Formation sub-selector
        with st.sidebar:
            if st.button("🔄 Rebuild Squad",key="sq_rebuild"):
                _sm,_dep=assign_players(players_list,formation)
                st.session_state["_squad_slot_map"]=_sm
                st.session_state["_squad_depth"]=_dep
                st.session_state["_squad_cache_key"]=_cache_key
                st.rerun()

        # Full squad table
        with st.expander("📋 Full Squad"):
            show_c=[c for c in ["Player","Position","Minutes played","Goals","Assists",
                                 "Market value","Contract expires","Age"] if c in _tp_filt.columns]
            st.dataframe(
                _tp_filt[show_c].sort_values("Minutes played",ascending=False).reset_index(drop=True),
                use_container_width=True
            )

st.markdown("---")
st.caption("TEAM HQ + SQUAD · Wyscout data · Percentile ranks computed within league pool")

        # Formation sub-selector
        with st.sidebar:
            if st.button("🔄 Rebuild Squad",key="sq_rebuild"):
                _sm,_dep=assign_players(players_list,formation)
                st.session_state["_squad_slot_map"]=_sm
                st.session_state["_squad_depth"]=_dep
                st.session_state["_squad_cache_key"]=_cache_key
                st.rerun()

        # Full squad table
        with st.expander("📋 Full Squad"):
            show_c=[c for c in ["Player","Position","Minutes played","Goals","Assists",
                                 "Market value","Contract expires","Age"] if c in _tp_filt.columns]
            st.dataframe(
                _tp_filt[show_c].sort_values("Minutes played",ascending=False).reset_index(drop=True),
                use_container_width=True
            )

st.markdown("---")
st.caption("TEAM HQ + SQUAD · Wyscout data · Percentile ranks computed within league pool")
