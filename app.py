# team_squad_app.py — Combined Team Profile + Squad Depth Chart + Pro Layout
# pip install streamlit pandas numpy matplotlib scikit-learn requests
# streamlit run team_squad_app.py

import io, os, re, math, unicodedata
from pathlib import Path
from datetime import date
from difflib import SequenceMatcher

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
                       show_contracts=True,show_positions=True):
    BG="#0a0f1c"; bsz="11px"; nsz="11px"; ssz="8px"; rsz="7px"

    def make_node(slot):
        ps_all=slot_map.get(slot["id"],[])
        ps=ps_all[:1] if xi_only else ps_all
        badge=(f'<div style="display:inline-block;padding:1px 6px;border:1.5px solid #ef4444;'
               f'color:#ef4444;font-size:{bsz};font-weight:900;letter-spacing:.08em;'
               f'margin-bottom:2px;background:rgba(10,15,28,.97);">{slot["label"]}</div>')
        rows=""
        for i,p in enumerate(ps):
            yrs=contract_years(p.get("Contract expires",""))
            yr_str=f"+{yrs}" if yrs>=0 else "+?"
            loan=is_loan(p); fw="800" if i==0 else "500"
            _lo=is_loaned_out(p); _yt=is_youth(p)
            col=player_css_color(yrs,loan,_lo,_yt)
            multi=" \U0001f501" if _multi_role(p.get("Position","")) else ""
            oop_s=f" ({p['_primary_pos']})" if (show_positions and p.get('_show_pos')) else ''
            suffix=(f" L{oop_s}{multi}" if loan else f"{(yr_str if show_contracts else '')}{oop_s}{multi}")
            stat_parts=[]
            if show_mins: stat_parts.append(f"{int(float(p.get('Minutes played') or 0))}\u2032")
            if show_goals:
                g=float(p.get("Goals") or 0)
                if g>0: stat_parts.append(f"{int(g)}\u26bd")
            if show_assists:
                a=float(p.get("Assists") or 0)
                if a>0: stat_parts.append(f"{int(a)}\U0001f170")
            stat_html=(f'<div style="color:#cbd5e1;font-size:{ssz};line-height:1.2;opacity:.85;">{" ".join(stat_parts)}</div>'
                      ) if stat_parts else ""
            rs_html=(best_role_html(p,df_sc,rsz) if (show_roles and (best_role_only or i>0))
                     else all_roles_html(p,df_sc,rsz) if (i==0 and show_roles) else "")
            mt="margin-top:4px;" if i>0 else ""
            rows+=(f'<div style="color:{col};font-size:{nsz};line-height:1.35;font-weight:{fw};{mt}'
                   f'white-space:nowrap;text-shadow:0 1px 4px rgba(0,0,0,.9);">'
                   f'{p["Player"]} {suffix}</div>{stat_html}{rs_html}')
        if not ps: rows=f'<div style="color:#1f2937;font-size:{ssz};">&#8212;</div>'
        sx=float(slot.get("x",50))
        mxw="105px" if (sx<18 or sx>82) else "130px"
        return (f'<div style="position:absolute;left:{slot["x"]}%;top:{slot["y"]}%;'
                f'transform:translate(-50%,-50%);text-align:center;'
                f'min-width:75px;max-width:{mxw};z-index:10;">'
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
            f'<div id="sq-pitch-field" style="position:relative;background:{BG};padding-bottom:142%;'
            f'overflow:hidden;border:1px solid #1a2540;">'
            f'{portrait_svg}{nodes}</div>'
            f'{depth_html}{legend}</div>')


FONT_URL="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap"

def make_sq_png_page(pitch_html:str, team:str, pitch_w:int=700)->str:
    BG="#0a0f1c"
    est_h=round(pitch_w*1.42)+180
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Saving PNG...</title>
<style>
@import url('{FONT_URL}');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:{BG};font-family:Montserrat,sans-serif;}}
#msg{{color:#fff;font-size:13px;text-align:center;padding:10px;letter-spacing:.12em;font-family:Montserrat,sans-serif;font-weight:700;}}
</style></head>
<body>
<div id="msg">GENERATING PNG — PLEASE WAIT...</div>
<div id="capture" style="width:{pitch_w}px;">{pitch_html}</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script>
document.fonts.ready.then(function(){{
  setTimeout(function(){{
    var el=document.getElementById('capture');
    var pf=el.querySelector('#sq-pitch-field');
    if(pf){{pf.style.paddingBottom='0';pf.style.height='{round(pitch_w*1.42)}px';}}
    html2canvas(el,{{
      backgroundColor:'{BG}',scale:2,useCORS:true,allowTaint:false,logging:false,
      width:el.offsetWidth,height:el.offsetHeight
    }}).then(function(canvas){{
      var a=document.createElement('a');
      a.download='{team.replace(" ","_")}_squad_depth.png';
      a.href=canvas.toDataURL('image/png');
      a.click();
      document.getElementById('msg').textContent='✓ PNG SAVED — YOU CAN CLOSE THIS TAB';
    }}).catch(function(e){{document.getElementById('msg').textContent='ERROR: '+e;}});
  }},1500);
}});
</script></body></html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# PRO LAYOUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
import unicodedata as _ud2
import re as _re2
from difflib import SequenceMatcher as _SM

# ── FotMob URL map ────────────────────────────────────────────────────────────
_FOTMOB_URLS = {}
for _badges_mod in ("badges", "team_fotmob_urls", "fotmob_urls"):
    try:
        _m = __import__(_badges_mod)
        _FOTMOB_URLS = (
            getattr(_m, "FOTMOB_TEAM_URLS", None)
            or getattr(_m, "TEAM_URLS", None)
            or getattr(_m, "BADGES", None)
            or {}
        )
        break
    except Exception:
        pass

# ── League logo URL helper ────────────────────────────────────────────────────
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

_POS_COLORS_PRO = {
    "CF":"#6EA8FF","LWF":"#6EA8FF","RWF":"#6EA8FF","LW":"#6EA8FF","RW":"#6EA8FF",
    "LAMF":"#6EA8FF","RAMF":"#6EA8FF","AMF":"#7FE28A",
    "LCMF":"#5FD37A","RCMF":"#5FD37A","CMF":"#5FD37A",
    "DMF":"#31B56B","LDMF":"#31B56B","RDMF":"#31B56B",
    "LWB":"#FFD34D","RWB":"#FFD34D","LB":"#FF9A3C","RB":"#FF9A3C",
    "CB":"#D1763A","LCB":"#D1763A","RCB":"#D1763A",
    "GK":"#B8A1FF",
}

def _pro_chip_color(p: str) -> str:
    return _POS_COLORS_PRO.get(str(p).strip().upper(), "#2d3550")

_TWEMOJI_SPECIAL = {
    "eng":"1f3f4-e0067-e0062-e0065-e006e-e0067-e007f",
    "sct":"1f3f4-e0067-e0062-e0073-e0063-e0074-e007f",
    "wls":"1f3f4-e0067-e0062-e0077-e006c-e0073-e007f",
}
_COUNTRY_TO_CC = {
    "united kingdom":"gb","great britain":"gb","northern ireland":"gb","england":"eng","scotland":"sct","wales":"wls",
    "ireland":"ie","republic of ireland":"ie","spain":"es","france":"fr","germany":"de","italy":"it","portugal":"pt",
    "netherlands":"nl","belgium":"be","austria":"at","switzerland":"ch","denmark":"dk","sweden":"se","norway":"no",
    "finland":"fi","iceland":"is","poland":"pl","czech republic":"cz","czechia":"cz","slovakia":"sk","slovenia":"si",
    "croatia":"hr","serbia":"rs","bosnia":"ba","montenegro":"me","kosovo":"xk","albania":"al","greece":"gr",
    "hungary":"hu","romania":"ro","bulgaria":"bg","russia":"ru","ukraine":"ua","georgia":"ge","kazakhstan":"kz",
    "azerbaijan":"az","armenia":"am","turkey":"tr","cyprus":"cy","luxembourg":"lu","estonia":"ee","latvia":"lv",
    "lithuania":"lt","moldova":"md","north macedonia":"mk","qatar":"qa","saudi arabia":"sa","uae":"ae",
    "united arab emirates":"ae","israel":"il","japan":"jp","south korea":"kr","korea republic":"kr","china":"cn",
    "brazil":"br","argentina":"ar","uruguay":"uy","chile":"cl","colombia":"co","peru":"pe","ecuador":"ec",
    "paraguay":"py","bolivia":"bo","mexico":"mx","canada":"ca","united states":"us","usa":"us",
    "australia":"au","new zealand":"nz",
    "algeria":"dz","cameroon":"cm","ghana":"gh","nigeria":"ng","senegal":"sn","morocco":"ma","egypt":"eg",
    "ivory coast":"ci","cote d'ivoire":"ci","côte d'ivoire":"ci","dr congo":"cd","drc":"cd","congo":"cg",
    "south africa":"za","kenya":"ke","mali":"ml","guinea":"gn","sierra leone":"sl","angola":"ao",
    "burkina faso":"bf","togo":"tg","gabon":"ga","benin":"bj","zambia":"zm","mozambique":"mz",
    "zimbabwe":"zw","tanzania":"tz","uganda":"ug","rwanda":"rw","liberia":"lr","gambia":"gm","namibia":"na",
    "cape verde":"cv","cabo verde":"cv","ethiopia":"et","sudan":"sd","tunisia":"tn","libya":"ly",
    "palestine":"ps","hong kong":"hk","curacao":"cw","curaçao":"cw",
}

def _norm_str(s: str) -> str:
    if not s: return ""
    return _ud2.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii").strip().lower()

def _cc_to_twemoji(cc):
    if not cc or len(cc) != 2: return None
    a, b = cc.upper()
    return f"{0x1F1E6+(ord(a)-65):04x}-{0x1F1E6+(ord(b)-65):04x}"

def _flag_html(country_name: str) -> str:
    if not country_name: return "<span class=\"chip\">—</span>"
    n = _norm_str(country_name)
    cc = _COUNTRY_TO_CC.get(n, "")
    if not cc: return "<span class=\"chip\">—</span>"
    if cc in _TWEMOJI_SPECIAL:
        src = f"https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/{_TWEMOJI_SPECIAL[cc]}.svg"
        return f"<span class=\"flagchip\"><img src=\"{src}\" alt=\"{country_name}\"></span>"
    code = _cc_to_twemoji(cc)
    if code:
        src = f"https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/{code}.svg"
        return f"<span class=\"flagchip\"><img src=\"{src}\" alt=\"{country_name}\"></span>"
    return f"<span class=\"chip\">{cc.upper()}</span>"

def _get_foot(row) -> str:
    for col in ("Foot", "Preferred foot", "Preferred Foot"):
        if col in row.index:
            val = row[col]
            try:
                if pd.isna(val): continue
            except Exception:
                pass
            s = str(val).strip()
            if s and s.lower() not in {"nan","none","null"}: return s
    return ""

# ═══════════════════════════════════════════════════════════════════════════════
# FIXED PHOTO RESOLUTION — robust slug + fuzzy matching (same as attacker app)
# ═══════════════════════════════════════════════════════════════════════════════

def _slug_name(s: str) -> str:
    """Accent-robust slug: handles ø, æ, å, ä, ö, ü, ß, ł, ç, ş, ğ, ı, đ, ð etc."""
    if not s: return ""
    s = str(s).strip().lower()
    for k, v in {
        "ø":"o","œ":"oe","æ":"ae","å":"a","ä":"a","ö":"o","ü":"u",
        "ß":"ss","ł":"l","đ":"d","ð":"d","þ":"th","ç":"c",
        "ş":"s","ğ":"g","ı":"i",
    }.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s)

def _slug_surname(player: str) -> str:
    p = (player or "").strip()
    if "," in p: return _slug_name(p.split(",", 1)[0].strip())
    parts = p.split()
    return _slug_name(parts[-1]) if parts else ""

def _fotmob_squad_cached(team_id: str) -> list:
    cache = st.session_state.setdefault("_fotmob_squad_cache2", {})
    if team_id in cache: return cache[team_id] or []
    squad = []
    try:
        url = f"https://www.fotmob.com/api/teams?id={team_id}"
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            data = r.json() or {}
            raw_squad = data.get("squad", None)
            if isinstance(raw_squad, list):
                for section in raw_squad:
                    members = section.get("members") or section.get("players") or []
                    if isinstance(members, list):
                        squad.extend([m for m in members if isinstance(m, dict)])
            elif isinstance(raw_squad, dict):
                for k in ("members", "players"):
                    members = raw_squad.get(k)
                    if isinstance(members, list):
                        squad.extend([m for m in members if isinstance(m, dict)])
                nested = raw_squad.get("squad")
                if isinstance(nested, list):
                    for section in nested:
                        members = section.get("members") or section.get("players") or []
                        if isinstance(members, list):
                            squad.extend([m for m in members if isinstance(m, dict)])
    except Exception:
        squad = []
    cache[team_id] = squad
    return squad

def _get_fotmob_url(team: str) -> str:
    return (_FOTMOB_URLS.get(team) or "").strip()

def resolve_player_photo(player, team, league, key_id, session_photo_map, global_overrides):
    """
    Resolve a FotMob player photo URL using robust accent-tolerant slug matching
    with exact surname, full-name contains, and fuzzy fallback (threshold 0.82).
    """
    if session_photo_map.get(key_id): return session_photo_map[key_id]
    if global_overrides.get(key_id): return global_overrides[key_id]

    team_url = _get_fotmob_url(team)
    tid_m = re.search(r"/teams/(\d+)/", team_url or "")
    if tid_m:
        squad = _fotmob_squad_cached(tid_m.group(1))
        target_surname = _slug_name(_slug_surname(player))
        target_full    = _slug_name(player)
        best_id = ""

        # 1) Exact surname match — prefer full-name hit within that set
        if target_surname:
            for m in squad:
                name = m.get("name") or m.get("playerName") or ""
                pid  = str(m.get("id") or m.get("playerId") or m.get("primaryId") or "")
                if not pid: continue
                if _slug_name(_slug_surname(name)) == target_surname:
                    best_id = pid
                    if target_full and target_full in _slug_name(name):
                        break  # perfect match, stop early

        # 2) Full slug of player name contained in squad member slug
        if not best_id and target_full:
            for m in squad:
                name = m.get("name") or m.get("playerName") or ""
                pid  = str(m.get("id") or m.get("playerId") or m.get("primaryId") or "")
                if pid and target_full in _slug_name(name):
                    best_id = pid; break

        # 3) Fuzzy surname fallback — lower threshold (0.82) catches lower-league names
        if not best_id and target_surname:
            bsc, bpid = 0.0, ""
            for m in squad:
                name = m.get("name") or m.get("playerName") or ""
                pid  = str(m.get("id") or m.get("playerId") or m.get("primaryId") or "")
                if not pid: continue
                sc = _SM(None, _slug_name(_slug_surname(name)), target_surname).ratio()
                if sc > bsc: bsc, bpid = sc, pid
            if bsc >= 0.82:
                best_id = bpid

        if best_id and str(best_id).isdigit():
            url = f"https://images.fotmob.com/image_resources/playerimages/{best_id}.png"
            session_photo_map[key_id] = url
            return url

    return "https://i.redd.it/43axcjdu59nd1.jpeg"


def load_remote_img(url):
    try:
        r=requests.get(url,timeout=8); r.raise_for_status()
        return plt.imread(io.BytesIO(r.content))
    except: return None

def fotmob_crest_url(team):
    raw = (_FOTMOB_URLS.get(team) or "").strip()
    if not raw: return ""
    if raw.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".svg")):
        return raw
    m = re.search(r"/teams/(\d+)/", raw)
    if m: return f"https://images.fotmob.com/image_resources/logo/teamlogo/{m.group(1)}.png"
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

    PLAYER_COL_MAP = {
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
                continue
            for alias in aliases:
                if alias in col_lower:
                    rename[col_lower[alias]] = standard
                    break
        df = df.rename(columns=rename)
        for c in ["Player", "Team", "Position", "League"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        for c in ["Minutes played", "Goals", "Assists", "Age", "xG", "xA"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        missing = [c for c in ["Player", "Position"] if c not in df.columns]
        if missing:
            available = ", ".join(df.columns.tolist())
            raise KeyError(
                f"Could not find columns {missing} in player CSV. "
                f"Available columns: {available}"
            )
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
    show_mins      =st.toggle("Minutes played",True,key="sq_mins")
    show_goals     =st.toggle("Goals",True,key="sq_goals")
    show_assists   =st.toggle("Assists",True,key="sq_assists")
    show_roles     =st.toggle("Role scores",True,key="sq_roles")
    best_role_only =st.toggle("Best role only",False,key="sq_bestonly")
    xi_only        =st.toggle("XI only",False,key="sq_xionly")
    show_contracts =st.toggle("Show contracts",True,key="sq_contracts")
    show_positions =st.toggle("Show positions",True,key="sq_showpos")
    sq_min_mins    =st.slider("Min player minutes",0,3000,0,50,key="sq_minmins")
    st.session_state["sq_minmins_val"]=sq_min_mins
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

import base64

flag=flag_html(t_league)
badge_img=get_team_badge(sel_team)

def _img_to_b64(img_array):
    buf=io.BytesIO(); plt.imsave(buf,img_array,format="png")
    return base64.b64encode(buf.getvalue()).decode()

@st.cache_data(show_spinner=False)
def _fetch_b64_url(url: str) -> str:
    if not url: return ""
    try:
        r = requests.get(url, timeout=8); r.raise_for_status()
        mime = r.headers.get("Content-Type","image/png").split(";")[0].strip()
        b64  = base64.b64encode(r.content).decode()
        return f"data:{mime};base64,{b64}"
    except: return ""

if badge_img is not None:
    badge_html_header=(f'<img src="data:image/png;base64,{_img_to_b64(badge_img)}" '
                       f'style="width:80px;height:80px;object-fit:contain;border-radius:8px;"/>')
else:
    _raw_url=fotmob_crest_url(sel_team)
    _badge_b64=_fetch_b64_url(_raw_url) if _raw_url else ""
    if _badge_b64:
        badge_html_header=(f'<img src="{_badge_b64}" '
                           f'style="width:80px;height:80px;object-fit:contain;border-radius:8px;"/>')
    else:
        badge_html_header=('<div style="width:80px;height:80px;background:#111827;border-radius:8px;'
                           'display:flex;align-items:center;justify-content:center;font-size:32px;">🏟️</div>')

_league_logo_url = _get_league_logo_url(t_league)
_league_logo_b64 = _fetch_b64_url(_league_logo_url) if _league_logo_url else ""
league_logo_html = (f'<img src="{_league_logo_b64}" '
                    f'style="height:32px;width:32px;object-fit:contain;vertical-align:middle;'
                    f'margin-right:6px;border-radius:3px;"/>') if _league_logo_b64 else ""

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
  <div style="flex-shrink:0;">{badge_html_header}</div>
  <div style="flex:1;min-width:200px;">
    <div style="font-family:Montserrat,sans-serif;font-size:34px;font-weight:900;
                color:#fff;letter-spacing:.03em;line-height:1.1;">{sel_team.upper()}</div>
    <div style="margin-top:8px;display:flex;align-items:center;font-size:14px;
                color:#9ca3af;font-weight:600;gap:6px;flex-wrap:wrap;">
      {league_logo_html}<span>{t_league}</span>
      <span style="color:#374151;">&nbsp;·&nbsp;</span>
      {flag}<span>{t_country}</span>
      <span style="color:#374151;">&nbsp;·&nbsp;</span>
      <span>{t_region}</span>
    </div>
  </div>
  <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end;">
    {score_chip("Overall",ovr)}
    {score_chip("Attack",att)}
    {score_chip("Defense",defv)}
    {score_chip("Possession",pos)}
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# POLAR RADAR
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

col_radar, col_info = st.columns([3, 2])

with col_radar:
    fig_y=plt.figure(figsize=(10,9))
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
        ax_y.text(rot_angles_y[i],155,lab.upper(),ha='center',va='center',
                  fontsize=10,weight='bold',color='white',zorder=5)
    ax_y.set_xticks([]); ax_y.set_yticks([])
    ax_y.spines['polar'].set_visible(False); ax_y.grid(False)

    st.pyplot(fig_y,use_container_width=True)
    buf_y=io.BytesIO()
    fig_y.savefig(buf_y,format="png",dpi=200,bbox_inches='tight',facecolor="#0a0f1c")
    st.download_button("⬇️ Download Radar",buf_y.getvalue(),
                       f"{sel_team.replace(' ','_')}_radar.png","image/png")
    plt.close(fig_y)

# ═══════════════════════════════════════════════════════════════════════════════
# STYLE / STRENGTHS / WEAKNESSES
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

_stats_rows=""
for lbl,col,inv in [("xG","xG p90",False),("xGA","xG Against p90",True),
                    ("Poss","Possession %",False),("PPDA","PPDA",True),
                    ("Passes","Passes p90",False),("Pts","Points",False)]:
    if col in t_row.index and pd.notna(t_row.get(col)):
        v=float(t_row[col])
        pct=_op_pct(col,inv)
        fg="#000" if pct>=44 else "#fff"
        _stats_rows+=(
            f'<tr style="border-bottom:1px solid #1e2d4a;">'
            f'<td style="color:#9ca3af;font-size:12px;font-weight:600;padding:6px 8px;white-space:nowrap;">{lbl}</td>'
            f'<td style="color:#fff;font-size:13px;font-weight:700;padding:6px 8px;text-align:right;">{v:.2f}</td>'
            f'<td style="padding:6px 4px;text-align:right;">'
            f'<span style="background:{rating_color(pct)};color:{fg};font-size:11px;font-weight:900;'
            f'padding:2px 8px;border-radius:5px;display:inline-block;min-width:30px;text-align:center;">'
            f'{int(pct)}</span></td></tr>')
_stats_html=(f'<table style="width:100%;border-collapse:collapse;">{_stats_rows}</table>'
             if _stats_rows else "")

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
    ({_pts_str}) &nbsp;·&nbsp; xPts: <b style="color:#fff;">{f"{float(t_row['Expected Points']):.1f}" if pd.notna(t_row.get('Expected Points')) else '—'}</b>
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
    _team_players=df_players[df_players["Team"]==sel_team]
    if _team_players.empty:
        _tm=sel_team.lower()
        _match=df_players[df_players["Team"].str.lower().str.contains(_tm[:8],na=False)]
        if not _match.empty:
            _actual_team=_match["Team"].iloc[0]
            _team_players=df_players[df_players["Team"]==_actual_team]

    if _team_players.empty:
        st.info(f"No squad data found for **{sel_team}** in the player CSV.")
    else:
        sq_min_mins=st.session_state.get("sq_minmins_val",0)
        _tp_filt=_team_players[_team_players["Minutes played"]>=sq_min_mins].copy()
        _tp_filt["_key"]=_tp_filt["Player"]
        players_list=_tp_filt.to_dict("records")

        _cache_key=(sel_team,formation,sq_min_mins)
        if st.session_state.get("_squad_cache_key")!=_cache_key:
            _sm,_dep=assign_players(players_list,formation)
            st.session_state["_squad_slot_map"]=_sm
            st.session_state["_squad_depth"]=_dep
            st.session_state["_squad_cache_key"]=_cache_key

        slot_map_sq=st.session_state["_squad_slot_map"]
        depth_sq=st.session_state["_squad_depth"]
        slots_sq=FORMATIONS[formation]

        pitch_html_sq=render_squad_pitch(
            sel_team,t_league,formation,slots_sq,slot_map_sq,depth_sq,df_players_sc,
            show_mins=show_mins,show_goals=show_goals,show_assists=show_assists,
            show_roles=show_roles,xi_only=xi_only,best_role_only=best_role_only,
            show_contracts=show_contracts,show_positions=show_positions
        )
        _pc1,_pc2,_pc3=st.columns([1,4,1])
        with _pc2:
            st.markdown(pitch_html_sq,unsafe_allow_html=True)
        _png_sq=make_sq_png_page(pitch_html_sq,sel_team,700)
        st.download_button("⬇️ Download Depth Chart",_png_sq.encode("utf-8"),
                           f"{sel_team.replace(' ','_')}_OPEN_TO_SAVE_PNG.html","text/html",
                           help="Download → open in Chrome/Edge → PNG auto-saves",
                           key="sq_dl")
        if st.button("🔄 Rebuild Squad", key="sq_rebuild"):
            _sm,_dep=assign_players(players_list,formation)
            st.session_state["_squad_slot_map"]=_sm
            st.session_state["_squad_depth"]=_dep
            st.session_state["_squad_cache_key"]=_cache_key
            st.rerun()

        with st.expander("📋 Full Squad"):
            show_c=[c for c in ["Player","Position","Minutes played","Goals","Assists",
                                 "Market value","Contract expires","Age"] if c in _tp_filt.columns]
            st.dataframe(
                _tp_filt[show_c].sort_values("Minutes played",ascending=False).reset_index(drop=True),
                use_container_width=True
            )

# ═══════════════════════════════════════════════════════════════════════════════
# PRO LAYOUT SECTION
# ═══════════════════════════════════════════════════════════════════════════════

PRO_LAYOUT_GROUPS = [
    ("CF / ST",  ["CF","LWF","RWF"]),
    ("AM / W",   ["AMF","LAMF","RAMF","LW","RW"]),
    ("CM",       ["LCMF","RCMF","CMF"]),
    ("DM",       ["DMF","LDMF","RDMF"]),
    ("FB",       ["LB","RB","LWB","RWB"]),
    ("CB",       ["CB","LCB","RCB"]),
    ("GK",       ["GK"]),
]

# ── All 5 CM/DM roles — top 3 shown dynamically by score ─────────────────────
_CM_ALL_ROLES = {
    "Deep Playmaker": {
        "col": "_rs_pro_deep_playmaker",
        "metrics": {
            "Passes per 90": 1,
            "Accurate passes, %": 1,
            "Forward passes per 90": 2,
            "Accurate forward passes, %": 1.5,
            "Progressive passes per 90": 3,
            "Passes to final third per 90": 2.5,
            "Accurate long passes, %": 1,
        }
    },
    "Advanced Playmaker": {
        "col": "_rs_pro_adv_playmaker",
        "metrics": {
            "Deep completions per 90": 1.5,
            "Smart passes per 90": 2,
            "xA per 90": 4,
            "Passes to penalty area per 90": 2,
        }
    },
    "Defensive Midfielder": {
        "col": "_rs_pro_def_mid",
        "metrics": {
            "Defensive duels per 90": 4,
            "Defensive duels won, %": 4,
            "PAdj Interceptions": 3,
            "Aerial duels per 90": 0.5,
            "Aerial duels won, %": 1,
        }
    },
    "Goal Threat CM": {
        "col": "_rs_pro_goal_threat_cm",
        "metrics": {
            "Non-penalty goals per 90": 3,
            "xG per 90": 3,
            "Shots per 90": 1.5,
            "Touches in box per 90": 2,
        }
    },
    "Ball-Carrying CM": {
        "col": "_rs_pro_ball_carry_cm",
        "metrics": {
            "Dribbles per 90": 4,
            "Successful dribbles, %": 2,
            "Progressive runs per 90": 3,
            "Accelerations per 90": 3,
        }
    },
}

_GROUP_ROLES = {
    "CF / ST": [
        ("_rs_Goal Threat CF",  "Goal Threat CF"),
        ("_rs_Link Up CF",      "Link-Up CF"),
        ("_rs_Target Man CF",   "Target Man CF"),
    ],
    "AM / W": [
        ("_rs_Goal Threat ATT", "Goal Threat"),
        ("_rs_Playmaker ATT",   "Playmaker"),
        ("_rs_Ball Carrier ATT","Ball Carrier"),
    ],
    # CM & DM: use _CM_ALL_ROLES, top 3 picked dynamically
    "CM":  None,
    "DM":  None,
    "FB": [
        ("_rs_Build Up FB",   "Build Up FB"),
        ("_rs_Attacking FB",  "Attacking FB"),
        ("_rs_Defensive FB",  "Defensive FB"),
    ],
    "CB": [
        ("_rs_Ball Playing CB", "Ball Playing CB"),
        ("_rs_Wide CB",         "Wide CB"),
        ("_rs_Box Defender",    "Box Defender"),
    ],
    "GK": [
        ("_rs_Shot Stopper GK", "Shot Stopper GK"),
        ("_rs_Ball Playing GK", "Ball Playing GK"),
        ("_rs_Sweeper GK",      "Sweeper GK"),
    ],
}

# ── Position-specific metrics ─────────────────────────────────────────────────
# ── CF / ST ──────────────────────────────────────────────────────────────────
_CF_ATT_METRICS = [
    ("Crosses",                   "Crosses per 90"),
    ("Crossing Accuracy %",       "Accurate crosses, %"),
    ("Goals: Non-Penalty",        "Non-penalty goals per 90"),
    ("xG",                        "xG per 90"),
    ("Conversion Rate %",         "Goal conversion, %"),
    ("Header Goals",              "Head goals per 90"),
    ("Expected Assists",          "xA per 90"),
    ("Progressive Runs",          "Progressive runs per 90"),
    ("Shots",                     "Shots per 90"),
    ("Shooting Accuracy %",       "Shots on target, %"),
    ("Touches in Opposition Box", "Touches in box per 90"),
]
_CF_DEF_METRICS = [
    ("Aerial Duels",              "Aerial duels per 90"),
    ("Aerial Duel Success %",     "Aerial duels won, %"),
    ("Defensive Duels",           "Defensive duels per 90"),
    ("Defensive Duel Success %",  "Defensive duels won, %"),
    ("PAdj. Interceptions",       "PAdj Interceptions"),
]
_CF_POS_METRICS = [
    ("Deep Completions",          "Deep completions per 90"),
    ("Dribbles",                  "Dribbles per 90"),
    ("Dribbling Success %",       "Successful dribbles, %"),
    ("Key Passes",                "Key passes per 90"),
    ("Passes",                    "Passes per 90"),
    ("Passing Accuracy %",        "Accurate passes, %"),
    ("Passes to Penalty Area",    "Passes to penalty area per 90"),
    ("Passes to Penalty Area %",  "Accurate passes to penalty area, %"),
    ("Smart Passes",              "Smart passes per 90"),
]

# ── AM / W ───────────────────────────────────────────────────────────────────
_AMW_ATT_METRICS = [
    ("Crosses",                   "Crosses per 90"),
    ("Crossing Accuracy %",       "Accurate crosses, %"),
    ("Goals: Non-Penalty",        "Non-penalty goals per 90"),
    ("xG",                        "xG per 90"),
    ("Conversion Rate %",         "Goal conversion, %"),
    ("Expected Assists",          "xA per 90"),
    ("Progressive Runs",          "Progressive runs per 90"),
    ("Shots",                     "Shots per 90"),
    ("Shooting Accuracy %",       "Shots on target, %"),
    ("Touches in Opposition Box", "Touches in box per 90"),
]
_AMW_DEF_METRICS = [
    ("Aerial Duels",              "Aerial duels per 90"),
    ("Aerial Duel Success %",     "Aerial duels won, %"),
    ("Defensive Duels",           "Defensive duels per 90"),
    ("Defensive Duel Success %",  "Defensive duels won, %"),
    ("PAdj. Interceptions",       "PAdj Interceptions"),
]
_AMW_POS_METRICS = [
    ("Accelerations",             "Accelerations per 90"),
    ("Deep Completions",          "Deep completions per 90"),
    ("Dribbles",                  "Dribbles per 90"),
    ("Dribbling Success %",       "Successful dribbles, %"),
    ("Forward Passes",            "Forward passes per 90"),
    ("Long Passes",               "Long passes per 90"),
    ("Key Passes",                "Key passes per 90"),
    ("Passes",                    "Passes per 90"),
    ("Passing Accuracy %",        "Accurate passes, %"),
    ("Passes to F3rd",            "Passes to final third per 90"),
    ("Passes F3rd %",             "Accurate passes to final third, %"),
    ("Passes to Penalty Area",    "Passes to penalty area per 90"),
    ("Passes to Penalty Area %",  "Accurate passes to penalty area, %"),
    ("Progressive Passes",        "Progressive passes per 90"),
    ("Prog Pass %",               "Accurate progressive passes, %"),
    ("Smart Passes",              "Smart passes per 90"),
]

# ── CM / DM ──────────────────────────────────────────────────────────────────
_CM_ATT_METRICS = [
    ("Crosses",                   "Crosses per 90"),
    ("Crossing Accuracy %",       "Accurate crosses, %"),
    ("Goals: Non-Penalty",        "Non-penalty goals per 90"),
    ("xG",                        "xG per 90"),
    ("Expected Assists",          "xA per 90"),
    ("Offensive Duels",           "Offensive duels per 90"),
    ("Offensive Duel Success %",  "Offensive duels won, %"),
    ("Progressive Runs",          "Progressive runs per 90"),
    ("Shots",                     "Shots per 90"),
    ("Touches in Opposition Box", "Touches in box per 90"),
]
_CM_DEF_METRICS = [
    ("Aerial Duels",              "Aerial duels per 90"),
    ("Aerial Duel Success %",     "Aerial duels won, %"),
    ("Defensive Duels",           "Defensive duels per 90"),
    ("Defensive Duel Success %",  "Defensive duels won, %"),
    ("Shots Blocked",             "Shots blocked per 90"),
    ("PAdj. Interceptions",       "PAdj Interceptions"),
]
_CM_POS_METRICS = [
    ("Deep Completions",          "Deep completions per 90"),
    ("Dribbles",                  "Dribbles per 90"),
    ("Dribbling Success %",       "Successful dribbles, %"),
    ("Forward Passes",            "Forward passes per 90"),
    ("Forward Passing %",         "Accurate forward passes, %"),
    ("Key passes",                "Key passes per 90"),
    ("Long Passes",               "Long passes per 90"),
    ("Long Passing %",            "Accurate long passes, %"),
    ("Passes",                    "Passes per 90"),
    ("Passing %",                 "Accurate passes, %"),
    ("Passes to Final 3rd",       "Passes to final third per 90"),
    ("Passes to Final 3rd %",     "Accurate passes to final third, %"),
    ("Passes to Penalty Area",    "Passes to penalty area per 90"),
    ("Pass to Penalty Area %",    "Accurate passes to penalty area, %"),
    ("Progessive Passes",         "Progressive passes per 90"),
    ("Progessive Passing %",      "Accurate progressive passes, %"),
    ("Smart Passes",              "Smart passes per 90"),
]

# ── FB ───────────────────────────────────────────────────────────────────────
_FB_ATT_METRICS = [
    ("Crosses",                   "Crosses per 90"),
    ("Crossing Accuracy %",       "Accurate crosses, %"),
    ("Goals: Non-Penalty",        "Non-penalty goals per 90"),
    ("xG",                        "xG per 90"),
    ("Expected Assists",          "xA per 90"),
    ("Offensive Duels",           "Offensive duels per 90"),
    ("Offensive Duel Success %",  "Offensive duels won, %"),
    ("Progressive Runs",          "Progressive runs per 90"),
    ("Shots",                     "Shots per 90"),
    ("Shooting Accuracy %",       "Shots on target, %"),
    ("Touches in Opposition Box", "Touches in box per 90"),
]
_FB_DEF_METRICS = [
    ("Aerial Duels",              "Aerial duels per 90"),
    ("Aerial Duel Success %",     "Aerial duels won, %"),
    ("Defensive Duels",           "Defensive duels per 90"),
    ("Defensive Duel Success %",  "Defensive duels won, %"),
    ("Shots Blocked",             "Shots blocked per 90"),
    ("PAdj. Interceptions",       "PAdj Interceptions"),
]
_FB_POS_METRICS = [
    ("Accelerations",             "Accelerations per 90"),
    ("Deep Completions",          "Deep completions per 90"),
    ("Dribbles",                  "Dribbles per 90"),
    ("Dribbling Success %",       "Successful dribbles, %"),
    ("Forward Passes",            "Forward passes per 90"),
    ("Forward Passing %",         "Accurate forward passes, %"),
    ("Key passes",                "Key passes per 90"),
    ("Long Passes",               "Long passes per 90"),
    ("Long Passing %",            "Accurate long passes, %"),
    ("Passes",                    "Passes per 90"),
    ("Passing %",                 "Accurate passes, %"),
    ("Passes to Final 3rd",       "Passes to final third per 90"),
    ("Passes to Final 3rd %",     "Accurate passes to final third, %"),
    ("Passes to Penalty Area",    "Passes to penalty area per 90"),
    ("Pass to Penalty Area %",    "Accurate passes to penalty area, %"),
    ("Progessive Passes",         "Progressive passes per 90"),
    ("Progessive Passing %",      "Accurate progressive passes, %"),
    ("Smart Passes",              "Smart passes per 90"),
]

# ── CB ───────────────────────────────────────────────────────────────────────
_CB_ATT_METRICS = [
    ("Goals: Non-Penalty",        "Non-penalty goals per 90"),
    ("xG",                        "xG per 90"),
    ("Offensive Duels",           "Offensive duels per 90"),
    ("Offensive Duel Success %",  "Offensive duels won, %"),
    ("Progressive Runs",          "Progressive runs per 90"),
]
_CB_DEF_METRICS = [
    ("Aerial Duels",              "Aerial duels per 90"),
    ("Aerial Duel Success %",     "Aerial duels won, %"),
    ("Defensive Duels",           "Defensive duels per 90"),
    ("Defensive Duel Success %",  "Defensive duels won, %"),
    ("PAdj Interceptions",        "PAdj Interceptions"),
    ("Shots Blocked",             "Shots blocked per 90"),
]
_CB_POS_METRICS = [
    ("Accelerations",             "Accelerations per 90"),
    ("Dribbles",                  "Dribbles per 90"),
    ("Dribbling  %",              "Successful dribbles, %"),
    ("Forward Passes",            "Forward passes per 90"),
    ("Forward Passing  %",        "Accurate forward passes, %"),
    ("Long Passes",               "Long passes per 90"),
    ("Long Passing  %",           "Accurate long passes, %"),
    ("Passes",                    "Passes per 90"),
    ("Passing Accuracy %",        "Accurate passes, %"),
    ("Passes to Final 3rd",       "Passes to final third per 90"),
    ("Passes to Final 3rd  %",    "Accurate passes to final third, %"),
    ("Progessive Passes",         "Progressive passes per 90"),
    ("Progessive Passing  %",     "Accurate progressive passes, %"),
]

# ── GK ───────────────────────────────────────────────────────────────────────
_GK_METRICS = [
    ("Exits",                     "Exits per 90"),
    ("Goals Prevented",           "Prevented goals per 90"),
    ("Goals Conceded",            "Conceded goals per 90"),
    ("Save Rate",                 "Save rate, %"),
    ("Shots Against",             "Shots against per 90"),
    ("xG Against",                "xG against per 90"),
]
_GK_POS_METRICS = [
    ("Long Passes",               "Long passes per 90"),
    ("Long Passing %",            "Accurate long passes, %"),
    ("Passes",                    "Passes per 90"),
    ("Passing Accuracy %",        "Accurate passes, %"),
]

_GROUP_METRICS = {
    "CF / ST": (_CF_ATT_METRICS,  _CF_DEF_METRICS,  _CF_POS_METRICS),
    "AM / W":  (_AMW_ATT_METRICS, _AMW_DEF_METRICS, _AMW_POS_METRICS),
    "CM":      (_CM_ATT_METRICS,  _CM_DEF_METRICS,  _CM_POS_METRICS),
    "DM":      (_CM_ATT_METRICS,  _CM_DEF_METRICS,  _CM_POS_METRICS),
    "FB":      (_FB_ATT_METRICS,  _FB_DEF_METRICS,  _FB_POS_METRICS),
    "CB":      (_CB_ATT_METRICS,  _CB_DEF_METRICS,  _CB_POS_METRICS),
    "GK":      (_GK_METRICS,      _GK_POS_METRICS,  []),
}


def _pro_rating_color_v2(v: float) -> str:
    v = float(v)
    for thr, col in [(85,"#2E6114"),(75,"#5C9E2E"),(66,"#7FBC41"),
                     (55,"#A7D763"),(41,"#F6D645"),(25,"#D77A2E"),(0,"#C63733")]:
        if v >= thr:
            return col
    return "#C63733"

def _show99(x) -> int:
    try: return max(0, min(99, int(float(x))))
    except: return 0

def _f2(n) -> str:
    try: return f"{int(n):02d}"
    except: return "00"

def _metric_pct_v2(row, met):
    col = f"{met} Percentile"
    if col in row.index:
        try:
            v = float(row[col])
            if not np.isnan(v): return v
        except: pass
    return np.nan

def _metric_val_v2(row, met):
    if met in row.index:
        try:
            v = float(row[met])
            if not np.isnan(v): return v
        except: pass
    return np.nan

def _avail_pairs(df, pairs):
    out = []
    for lab, met in pairs:
        if met in df.columns or f"{met} Percentile" in df.columns:
            out.append((lab, met))
    return out

def _sec_html_v2(title, pairs, df_view, row):
    pairs = _avail_pairs(df_view, pairs)
    rows = []
    for lab, met in pairs:
        pct  = _metric_pct_v2(row, met)
        p    = _show99(pct if not np.isnan(pct) else 0.0)
        rawv = _metric_val_v2(row, met)
        rtxt = "—" if np.isnan(rawv) else f"{rawv:.2f}".rstrip("0").rstrip(".")
        rows.append(
            "<div class=\"m-row\">"
            f"<div class=\"m-label\">{lab}</div>"
            "<div class=\"m-right\">"
            f"<div class=\"m-val\">{rtxt}</div>"
            f"<div class=\"m-badge\" style=\"background:{_pro_rating_color_v2(p)}\">{_f2(p)}</div>"
            "</div></div>"
        )
    if not rows:
        return ""
    return f"<div class=\"m-sec\"><div class=\"m-title\">{title}</div>{''.join(rows)}</div>"


_PRO_CSS = """
<style>
:root { --card:#141823; }
.pro-wrap{ display:flex; justify-content:center; }
.pro-card{
  position:relative; width:min(420px,96%);
  display:grid; grid-template-columns:96px 1fr 48px;
  gap:12px; align-items:start;
  background:var(--card); border:1px solid rgba(255,255,255,.06);
  border-radius:20px; padding:16px; margin-bottom:12px;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.03), 0 6px 24px rgba(0,0,0,.35);
}
.pro-avatar{ width:96px; height:96px; border-radius:12px; border:1px solid #2a3145; overflow:hidden; background:#0b0d12; }
.pro-avatar img{ width:100%; height:100%; object-fit:cover; image-rendering:auto; transform:translateZ(0); }
.flagchip{ display:inline-flex; align-items:center; gap:6px; background:transparent; border:none; padding:0; height:auto; }
.flagchip img{ width:26px; height:18px; border-radius:2px; display:block; }
.chip{ background:transparent; color:#a6a6a6; border:none; padding:0; border-radius:0; font-size:15px; line-height:18px; opacity:.92; }
.row{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:2px 0; }
.leftrow1{ margin-top:6px; } .leftrow-foot{ margin-top:2px; } .leftrow-contract{ margin-top:10px; }
.pill{
  min-width:36px; height:28px; padding:0 8px; border-radius:6px;
  display:inline-flex; align-items:center; justify-content:center;
  font-weight:800; font-size:18px; color:#0b0d12; box-sizing:border-box;
}
.name{ font-weight:800; font-size:22px; color:#e8ecff; margin-bottom:6px; letter-spacing:.2px; line-height:1.15; }
.sub{ color:#a8b3cf; font-size:15px; opacity:.9; }
.role-row{ display:flex; gap:8px; align-items:center; margin:2px 0; }
.posrow{ margin-top:13.5px; }
.postext{ font-weight:600; font-size:14.5px; letter-spacing:.2px; margin-right:11px; }
.rank{ position:absolute; top:10.5px; right:14px; color:#b7bfe1; font-weight:800; font-size:18px; text-align:right; pointer-events:none; }
.teamline{ color:#dbe3ff; font-size:14px; font-weight:600; margin-top:6.5px; letter-spacing:.05px; opacity:.95; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.tl-wrap{ position:relative; } .tl-has-crest{ padding-left:24px; }
.crest-icon{ height:1.35em; width:auto; object-fit:contain; image-rendering:auto; }
.crest-abs{ position:absolute; left:0; top:50%; transform:translateY(-50%); pointer-events:none; }
.grp-header{
  font-size:9px; font-weight:900; letter-spacing:.18em; color:#ef4444;
  text-transform:uppercase; margin:18px 0 8px 0;
  border-left:3px solid #ef4444; padding-left:8px;
}
.m-sec{ background:#121621; border:1px solid #242b3b; border-radius:14px; padding:9px 11px; }
.m-title{ color:#e8ecff; font-weight:800; letter-spacing:.02em; margin:4px 0 8px 0; font-size:12px; }
.m-row{ display:flex; align-items:center; gap:8px; padding:6px 6px; border-radius:8px; }
.m-label{ color:#c9d3f2; font-size:14.5px; flex:1 1 0%; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.m-right{ display:flex; align-items:center; gap:8px; flex:0 0 auto; }
.m-val{ color:#a8b3cf; font-size:12px; opacity:.9; min-width:48px; text-align:right; }
.m-badge{ min-width:40px; text-align:center; padding:2px 9px; border-radius:7px; font-weight:800; font-size:17px; color:#0b0d12; }
.metrics-grid{ display:grid; grid-template-columns:1fr; gap:10px; }
@media (min-width:720px){ .metrics-grid{ grid-template-columns:repeat(3,1fr); } }
</style>
"""

def render_pro_layout_v2(team_players_df: pd.DataFrame, df_sc, df_full: pd.DataFrame = None):
    st.markdown(_PRO_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("photo_map", {})
    st.session_state.setdefault("crest_map", {})

    _TOK_TO_GRP = {t: lbl for lbl, toks in PRO_LAYOUT_GROUPS for t in toks}
    def _player_grp(pos_str):
        return _TOK_TO_GRP.get(_tok(str(pos_str)), "CM")

    ALL_MET = list({met for _, met in
        _CF_ATT_METRICS + _CF_DEF_METRICS + _CF_POS_METRICS +
        _AMW_ATT_METRICS + _AMW_DEF_METRICS + _AMW_POS_METRICS +
        _CM_ATT_METRICS + _CM_DEF_METRICS + _CM_POS_METRICS +
        _FB_ATT_METRICS + _FB_DEF_METRICS + _FB_POS_METRICS +
        _CB_ATT_METRICS + _CB_DEF_METRICS + _CB_POS_METRICS +
        _GK_METRICS + _GK_POS_METRICS} |
        {met for role_cfg in _CM_ALL_ROLES.values() for met in role_cfg["metrics"]})

    @st.cache_data(show_spinner=False)
    def _build_ref(df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        df["_grp"] = df["Position"].apply(_player_grp)
        for met in ALL_MET:
            if met not in df.columns:
                continue
            df[met] = pd.to_numeric(df[met], errors="coerce")
            pct_col = f"{met} Percentile"
            df[pct_col] = np.nan
            for (lg, grp), idx in df.groupby(["League", "_grp"]).groups.items():
                pool = df.loc[idx, met].dropna()
                if pool.empty:
                    continue
                ranks = pool.rank(pct=True, method="average") * 100.0
                df.loc[ranks.index, pct_col] = ranks
        return df

    ref_src = df_full if (df_full is not None and not df_full.empty) else \
              df_sc   if (df_sc   is not None and not df_sc.empty)   else \
              team_players_df
    df_ref = _build_ref(ref_src)

    df_team = team_players_df.drop_duplicates(subset=["Player", "Team"]).copy()
    mins_col = "Minutes played"
    if mins_col in df_team.columns:
        df_team[mins_col] = pd.to_numeric(df_team[mins_col], errors="coerce")
        df_team = df_team.sort_values(mins_col, ascending=False)
    df_team = df_team.reset_index(drop=True)

    if df_team.empty:
        st.info("No players found for this team.")
        return

    def _role_score_from_ref(player, team, metrics: dict) -> float:
        mask = (df_ref["Player"] == player) & (df_ref["Team"] == team)
        hits = df_ref[mask]
        if hits.empty:
            return 0.0
        r = hits.iloc[0]
        total_w, wsum = 0.0, 0.0
        for met, w in metrics.items():
            pct_col = f"{met} Percentile"
            if pct_col in r.index and pd.notna(r[pct_col]):
                wsum += float(r[pct_col]) * w
                total_w += w
        return (wsum / total_w) if total_w > 0 else 0.0

    for rank_i, (_, row) in enumerate(df_team.iterrows(), start=1):
        player  = str(row.get("Player", "") or "")
        team    = str(row.get("Team",   "") or "")
        league  = str(row.get("League", "") or "")
        pos     = str(row.get("Position", "") or "")
        grp     = _player_grp(pos)

        try:    mins = int(float(row.get(mins_col) or 0))
        except: mins = 0
        mins_txt = f"{mins}\u2032"

        try:    age_val = int(float(row.get("Age") or 0)) if pd.notna(row.get("Age", np.nan)) else 0
        except: age_val = 0
        age_txt = f"{age_val}y.o." if age_val > 0 else "\u2014"

        cy  = pd.to_datetime(row.get("Contract expires"), errors="coerce")
        cyr = int(cy.year) if pd.notna(cy) else 0
        contract_txt = str(cyr) if cyr > 0 else "\u2014"

        birth = str(row.get("Birth country", "") or "")
        foot  = _get_foot(row) or "\u2014"
        flag  = _flag_html(birth)

        codes = [c for c in re.split(r"[,\s/;]+", pos.strip().upper()) if c]
        seen, ordered = set(), []
        for c in codes:
            if c not in seen: seen.add(c); ordered.append(c)
        pos_html = "".join(
            f'<span class="postext" style="color:{_pro_chip_color(c)}">{c}</span>'
            for c in ordered
        )

        # ── avatar via fixed resolve_player_photo ──────────────────────────
        key_id = f"{_norm_str(player)}|{_norm_str(team)}"
        avatar_url = resolve_player_photo(
            player=player, team=team, league=league,
            key_id=key_id,
            session_photo_map=st.session_state["photo_map"],
            global_overrides={},
        )

        # ── role pills — dynamic top-3 for CM/DM, fixed for others ───────────
        all_role_pairs = _GROUP_ROLES.get(grp)  # None means CM/DM dynamic
        pills_html = ""

        sc_row = None
        if df_sc is not None and not df_sc.empty:
            sc_mask = (df_sc["Player"] == player) & (df_sc["Team"] == team)
            sc_hits = df_sc[sc_mask]
            if not sc_hits.empty:
                sc_row = sc_hits.iloc[0]

        if all_role_pairs is None:
            # CM / DM: compute all 5 scores, show top 3
            scored_roles = []
            for role_name, role_cfg in _CM_ALL_ROLES.items():
                val = _show99(_role_score_from_ref(player, team, role_cfg["metrics"]))
                scored_roles.append((val, role_name))
            scored_roles.sort(key=lambda x: -x[0])
            for val, role_name in scored_roles[:3]:
                pills_html += (
                    f'<div class="role-row">'
                    f'<span class="pill" style="background:{_pro_rating_color_v2(val)}">{_f2(val)}</span>'
                    f'<span class="sub">{role_name}</span></div>'
                )
        else:
            for col_name, lbl in all_role_pairs[:3]:
                val = 0
                if sc_row is not None and col_name in sc_row.index and pd.notna(sc_row[col_name]):
                    val = _show99(sc_row[col_name])
                else:
                    rk = ROLE_KEY_MAP.get(_tok(pos), "ATT")
                    bucket_roles = ROLE_BUCKETS.get(rk, {})
                    metrics = {}
                    for rn, spec in bucket_roles.items():
                        if lbl.lower().replace(" ", "") in rn.lower().replace(" ", ""):
                            metrics = spec.get("metrics", {}); break
                    if not metrics and bucket_roles:
                        metrics = next(iter(bucket_roles.values())).get("metrics", {})
                    val = _show99(_role_score_from_ref(player, team, metrics))
                pills_html += (
                    f'<div class="role-row">'
                    f'<span class="pill" style="background:{_pro_rating_color_v2(val)}">{_f2(val)}</span>'
                    f'<span class="sub">{lbl}</span></div>'
                )

        # ── crest ──────────────────────────────────────────────────────────
        crest_store_key = f"{_norm_str(team)}|{_norm_str(league)}"
        crest_url = st.session_state["crest_map"].get(crest_store_key, "")
        if not crest_url:
            t_url = _get_fotmob_url(team)
            if t_url:
                tid = re.search(r"/teams/(\d+)/", t_url)
                crest_url = f"https://images.fotmob.com/image_resources/logo/teamlogo/{tid.group(1)}.png" if tid else ""

        if crest_url:
            teamline_html = (
                '<div class="teamline tl-wrap tl-has-crest">'
                f'<img class="crest-icon crest-abs" src="{crest_url}" alt="">'
                f'<span>{team} \u00b7 {league}</span></div>'
            )
        else:
            teamline_html = f'<div class="teamline">{team} \u00b7 {league}</div>'

        card_html = (
            '<div class="pro-wrap">'
            '<div class="pro-card">'
            '<div class="leftcol">'
            '<div class="pro-avatar">'
            f'<img src="{avatar_url}" alt="{player}" loading="lazy" />'
            '</div>'
            f'<div class="row leftrow1">{flag}<span class="chip">{age_txt}</span></div>'
            f'<div class="row leftrow-foot"><span class="chip">{mins_txt} &nbsp; {foot}</span></div>'
            f'<div class="row leftrow-contract"><span class="chip">{contract_txt}</span></div>'
            '</div>'
            '<div>'
            f'<div class="name">{player}</div>'
            f'{pills_html}'
            f'<div class="row posrow">{pos_html}</div>'
            f'{teamline_html}'
            '</div>'
            f'<div class="rank">#{rank_i:02d}</div>'
            '</div>'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

        atts, defs, poss = _GROUP_METRICS.get(grp, (_CM_ATT_METRICS, _CM_DEF_METRICS, _CM_POS_METRICS))

        ref_mask = (df_ref["Player"] == player) & (df_ref["Team"] == team)
        ref_rows = df_ref[ref_mask]
        ref_row  = ref_rows.iloc[0] if not ref_rows.empty else row

        with st.expander("Individual Metrics", expanded=False):
            s1 = _sec_html_v2("ATTACKING"   if grp != "GK" else "GOALKEEPING", atts, df_ref, ref_row)
            s2 = _sec_html_v2("DEFENSIVE"   if grp != "GK" else "POSSESSION",  defs, df_ref, ref_row)
            s3 = _sec_html_v2("POSSESSION", poss, df_ref, ref_row) if poss else ""
            sections = [s for s in [s1, s2, s3] if s]
            if sections:
                st.markdown('<div class="metrics-grid">' + "".join(sections) + "</div>",
                            unsafe_allow_html=True)
            else:
                st.caption("No percentile data available for this player.")


# ═══════════════════════════════════════════════════════════════════════════════
# PRO LAYOUT RENDER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div style="font-size:11px;font-weight:900;letter-spacing:.18em;color:#6b7280;'
            'text-transform:uppercase;margin-bottom:10px;">Pro Layout</div>',
            unsafe_allow_html=True)

if df_players is None:
    st.info("Upload a player CSV in the sidebar to see the Pro Layout.")
else:
    _pro_players = df_players[df_players["Team"] == sel_team]
    if _pro_players.empty:
        _tm = sel_team.lower()
        _match = df_players[df_players["Team"].str.lower().str.contains(_tm[:8], na=False)]
        if not _match.empty:
            _pro_players = df_players[df_players["Team"] == _match["Team"].iloc[0]]

    if _pro_players.empty:
        st.info(f"No squad data found for **{sel_team}** in the player CSV.")
    else:
        sq_min_mins = st.session_state.get("sq_minmins_val", 0)
        _pro_filt = _pro_players[_pro_players["Minutes played"] >= sq_min_mins].copy()
        render_pro_layout_v2(_pro_filt, df_players_sc, df_full=df_players)

st.markdown("---")
st.caption("TEAM HQ + SQUAD · Wyscout data · Percentile ranks computed within league pool")

# ═══════════════════════════════════════════════════════════════════════════════
# TEAM PLAYER RANKINGS — CIES-style image with role/impact/raw metric scores
# Add this block at the bottom of team_squad_app.py, after the Pro Layout section
# ═══════════════════════════════════════════════════════════════════════════════

import math as _math

# ── Position group helpers ────────────────────────────────────────────────────

_RANK_POS_GROUPS = {
    "All Positions":     None,
    "Goalkeepers (GK)":  {"GK"},
    "Center Backs (CB)": {"CB", "LCB", "RCB"},
    "Fullbacks (FB)":    {"LB", "RB", "LWB", "RWB"},
    "Central Mid (CM)":  {"DMF", "LDMF", "RDMF", "LCMF", "RCMF"},
    "Attackers (W/AM)":  {"LW", "LWF", "RW", "RWF", "AMF", "LAMF", "RAMF"},
    "Strikers (CF)":     {"CF"},
}

# Metrics per position group (raw Wyscout column names)
_RANK_POS_METRICS = {
    "GK": [
        ("Exits",             "Exits per 90"),
        ("Goals Prevented",   "Prevented goals per 90"),
        ("Goals Conceded",    "Conceded goals per 90"),
        ("Save Rate",         "Save rate, %"),
        ("Shots Against",     "Shots against per 90"),
        ("xG Against",        "xG against per 90"),
        ("Long Passes",       "Long passes per 90"),
        ("Long Pass %",       "Accurate long passes, %"),
        ("Passes",            "Passes per 90"),
        ("Pass %",            "Accurate passes, %"),
    ],
    "CB": [
        ("Aerial Duels",      "Aerial duels per 90"),
        ("Aerial Duel %",     "Aerial duels won, %"),
        ("Def Duels",         "Defensive duels per 90"),
        ("Def Duel %",        "Defensive duels won, %"),
        ("PAdj Int",          "PAdj Interceptions"),
        ("Shots Blocked",     "Shots blocked per 90"),
        ("Pass %",            "Accurate passes, %"),
        ("Fwd Pass %",        "Accurate forward passes, %"),
        ("Prog Passes",       "Progressive passes per 90"),
        ("Prog Runs",         "Progressive runs per 90"),
        ("Long Passes",       "Long passes per 90"),
        ("Long Pass %",       "Accurate long passes, %"),
    ],
    "FB": [
        ("Aerial Duels",      "Aerial duels per 90"),
        ("Aerial Duel %",     "Aerial duels won, %"),
        ("Def Duels",         "Defensive duels per 90"),
        ("Def Duel %",        "Defensive duels won, %"),
        ("PAdj Int",          "PAdj Interceptions"),
        ("Crosses",           "Crosses per 90"),
        ("Cross %",           "Accurate crosses, %"),
        ("Dribbles",          "Dribbles per 90"),
        ("Dribble %",         "Successful dribbles, %"),
        ("Prog Runs",         "Progressive runs per 90"),
        ("Prog Passes",       "Progressive passes per 90"),
        ("xA",                "xA per 90"),
        ("Pass to Box",       "Passes to penalty area per 90"),
    ],
    "CM": [
        ("Def Duels",         "Defensive duels per 90"),
        ("Def Duel %",        "Defensive duels won, %"),
        ("PAdj Int",          "PAdj Interceptions"),
        ("Dribbles",          "Dribbles per 90"),
        ("Dribble %",         "Successful dribbles, %"),
        ("Prog Runs",         "Progressive runs per 90"),
        ("Prog Passes",       "Progressive passes per 90"),
        ("Pass to F3rd",      "Passes to final third per 90"),
        ("xA",                "xA per 90"),
        ("Pass to Box",       "Passes to penalty area per 90"),
        ("xG",                "xG per 90"),
        ("Goals",             "Non-penalty goals per 90"),
        ("Touches in Box",    "Touches in box per 90"),
        ("Pass %",            "Accurate passes, %"),
    ],
    "ATT": [
        ("Dribbles",          "Dribbles per 90"),
        ("Dribble %",         "Successful dribbles, %"),
        ("Prog Runs",         "Progressive runs per 90"),
        ("Crosses",           "Crosses per 90"),
        ("Cross %",           "Accurate crosses, %"),
        ("xA",                "xA per 90"),
        ("Pass to Box",       "Passes to penalty area per 90"),
        ("xG",                "xG per 90"),
        ("Goals",             "Non-penalty goals per 90"),
        ("Shots",             "Shots per 90"),
        ("Shot %",            "Shots on target, %"),
        ("Touches in Box",    "Touches in box per 90"),
    ],
    "CF": [
        ("Aerial Duels",      "Aerial duels per 90"),
        ("Aerial Duel %",     "Aerial duels won, %"),
        ("xG",                "xG per 90"),
        ("Goals",             "Non-penalty goals per 90"),
        ("Shots",             "Shots per 90"),
        ("Shot %",            "Shots on target, %"),
        ("Touches in Box",    "Touches in box per 90"),
        ("Dribbles",          "Dribbles per 90"),
        ("Prog Runs",         "Progressive runs per 90"),
        ("xA",                "xA per 90"),
        ("Pass to Box",       "Passes to penalty area per 90"),
        ("Pass %",            "Accurate passes, %"),
    ],
}

# Map group label → metrics key
_RANK_GROUP_TO_KEY = {
    "All Positions":     "CM",   # fallback for "all" — handled separately
    "Goalkeepers (GK)":  "GK",
    "Center Backs (CB)": "CB",
    "Fullbacks (FB)":    "FB",
    "Central Mid (CM)":  "CM",
    "Attackers (W/AM)":  "ATT",
    "Strikers (CF)":     "CF",
}

# ── Composite score weights (same formulas as attachments) ────────────────────
_RANK_COMPLETE_WEIGHTS = {
    "CB": {
        "Aerial duels won, %": 0.15,
        "Defensive duels won, %": 0.15,
        "Accurate passes, %": 0.10,
        "Accurate forward passes, %": 0.10,
        "Dribbles per 90": 0.05,
        "Progressive runs per 90": 0.15,
        "Progressive passes per 90": 0.15,
        "PAdj Interceptions": 0.15,
    },
    "FB": {
        "PAdj Interceptions": 0.10,
        "Defensive duels won, %": 0.10,
        "Accurate passes, %": 0.10,
        "Defensive duels per 90": 0.05,
        "Dribbles per 90": 0.10,
        "Progressive runs per 90": 0.10,
        "Progressive passes per 90": 0.10,
        "Passes to final third per 90": 0.10,
        "xA per 90": 0.10,
        "Passes to penalty area per 90": 0.10,
        "Smart passes per 90": 0.05,
    },
    "CM": {
        "PAdj Interceptions": 0.10,
        "Defensive duels won, %": 0.10,
        "Accurate passes, %": 0.10,
        "Defensive duels per 90": 0.05,
        "Dribbles per 90": 0.10,
        "Progressive runs per 90": 0.10,
        "Progressive passes per 90": 0.10,
        "Passes to final third per 90": 0.05,
        "xA per 90": 0.10,
        "Passes to penalty area per 90": 0.10,
        "Non-penalty goals per 90": 0.05,
        "xG per 90": 0.05,
    },
    "ATT": {
        "Accurate passes, %": 0.10,
        "Dribbles per 90": 0.15,
        "Progressive runs per 90": 0.10,
        "Passes to final third per 90": 0.05,
        "xA per 90": 0.20,
        "Passes to penalty area per 90": 0.10,
        "Non-penalty goals per 90": 0.10,
        "xG per 90": 0.20,
    },
    "CF": {
        "Accurate passes, %": 0.10,
        "Dribbles per 90": 0.15,
        "Progressive runs per 90": 0.10,
        "xA per 90": 0.15,
        "Passes to penalty area per 90": 0.05,
        "Non-penalty goals per 90": 0.20,
        "xG per 90": 0.25,
    },
    "GK": {
        "Save rate, %": 0.40,
        "Prevented goals per 90": 0.30,
        "Exits per 90": 0.15,
        "Accurate long passes, %": 0.15,
    },
}

def _rank_pos_key(tok: str) -> str:
    t = tok.upper().strip()
    if t == "GK": return "GK"
    if t in {"CB", "LCB", "RCB"}: return "CB"
    if t in {"LB", "RB", "LWB", "RWB"}: return "FB"
    if t in {"DMF", "LDMF", "RDMF", "LCMF", "RCMF"}: return "CM"
    if t in {"LW", "LWF", "RW", "RWF", "AMF", "LAMF", "RAMF"}: return "ATT"
    if t == "CF": return "CF"
    return "CM"

def _rank_pct_within_df(df: pd.DataFrame, col: str, league: str, pos_key: str) -> pd.Series:
    """Compute percentile rank for a column within league+position reference from df_players."""
    if df is None or col not in df.columns:
        return pd.Series(np.nan, index=range(len(df)))
    mask = (df["League"].astype(str) == str(league)) & (df["_ftok"].isin(
        {"GK"} if pos_key == "GK" else
        {"CB", "LCB", "RCB"} if pos_key == "CB" else
        {"LB", "RB", "LWB", "RWB"} if pos_key == "FB" else
        {"DMF", "LDMF", "RDMF", "LCMF", "RCMF"} if pos_key == "CM" else
        {"LW", "LWF", "RW", "RWF", "AMF", "LAMF", "RAMF"} if pos_key == "ATT" else
        {"CF"}
    ))
    ref = df[mask][col]
    s = pd.to_numeric(ref, errors="coerce")
    if s.dropna().empty:
        return pd.Series(np.nan, index=ref.index)
    return s.rank(pct=True, method="average") * 100.0


@st.cache_data(show_spinner=False)
def _build_team_rank_df(
    team_players_bytes: bytes,   # serialised via to_json for caching
    all_players_bytes: bytes,
    team: str,
    league: str,
    min_mins: int,
) -> pd.DataFrame:
    """
    Build a per-player DataFrame with:
    - raw metric values
    - percentile ranks vs league+position pool
    - Complete Score + Role Scores
    """
    df_team_pl = pd.read_json(io.BytesIO(team_players_bytes))
    df_all = pd.read_json(io.BytesIO(all_players_bytes))

    # Ensure _ftok exists
    if "_ftok" not in df_all.columns:
        df_all["_ftok"] = df_all["Position"].apply(_tok)
    if "_ftok" not in df_team_pl.columns:
        df_team_pl["_ftok"] = df_team_pl["Position"].apply(_tok)

    # Filter by min minutes
    df_team_pl = df_team_pl[
        pd.to_numeric(df_team_pl.get("Minutes played", 0), errors="coerce").fillna(0) >= min_mins
    ].copy()

    if df_team_pl.empty:
        return pd.DataFrame()

    result_rows = []

    for _, row in df_team_pl.iterrows():
        player = str(row.get("Player", ""))
        pos_str = str(row.get("Position", ""))
        tok = _tok(pos_str)
        pk = _rank_pos_key(tok)
        lg = str(row.get("League", league))

        rec = {
            "Player": player,
            "Position": pos_str,
            "_ftok": tok,
            "_pos_key": pk,
            "Team": str(row.get("Team", team)),
            "League": lg,
            "Age": row.get("Age", ""),
            "Minutes played": row.get("Minutes played", 0),
            "Goals": row.get("Goals", 0),
            "Assists": row.get("Assists", 0),
            "Market value": row.get("Market value", ""),
            "Contract expires": row.get("Contract expires", ""),
            "Birth country": row.get("Birth country", ""),
        }

        # All raw metrics across all position groups
        all_metrics = {met for pairs in _RANK_POS_METRICS.values() for _, met in pairs}
        for met in all_metrics:
            if met in row.index:
                rec[met] = pd.to_numeric(row[met], errors="coerce")

        # Percentiles vs league+position reference
        pos_pairs = _RANK_POS_METRICS.get(pk, _RANK_POS_METRICS["CM"])
        for lbl, met in pos_pairs:
            if met not in df_all.columns:
                rec[f"_pct_{met}"] = np.nan
                continue
            # get ref pool
            pos_toks = {
                "GK": {"GK"},
                "CB": {"CB", "LCB", "RCB"},
                "FB": {"LB", "RB", "LWB", "RWB"},
                "CM": {"DMF", "LDMF", "RDMF", "LCMF", "RCMF"},
                "ATT": {"LW", "LWF", "RW", "RWF", "AMF", "LAMF", "RAMF"},
                "CF": {"CF"},
            }.get(pk, set())
            ref_mask = (df_all["League"].astype(str) == lg) & (df_all["_ftok"].isin(pos_toks))
            ref_s = pd.to_numeric(df_all.loc[ref_mask, met], errors="coerce").dropna()
            val = pd.to_numeric(row.get(met, np.nan), errors="coerce")
            if pd.isna(val) or ref_s.empty:
                rec[f"_pct_{met}"] = np.nan
            else:
                rec[f"_pct_{met}"] = float((ref_s < val).mean() * 100 + (ref_s == val).mean() * 50)

        # Complete Score
        wmap = _RANK_COMPLETE_WEIGHTS.get(pk, {})
        comp_vals, comp_wts = [], []
        for met, w in wmap.items():
            pct = rec.get(f"_pct_{met}", np.nan)
            if pd.notna(pct):
                comp_vals.append(float(pct) * w)
                comp_wts.append(w)
        rec["Complete Score"] = float(sum(comp_vals) / sum(comp_wts)) if comp_wts else np.nan

        # Role Scores (from ROLE_BUCKETS)
        rk = _role_key(pos_str)
        for role_name, spec in ROLE_BUCKETS.get(rk, {}).items():
            r_vals, r_wts = [], []
            for met, w in spec.get("metrics", {}).items():
                pct = rec.get(f"_pct_{met}", np.nan)
                if pd.notna(pct):
                    r_vals.append(float(pct) * w)
                    r_wts.append(w)
            rec[f"_role_{role_name}"] = float(sum(r_vals) / sum(r_wts)) if r_wts else np.nan

        # Impact Score (simplified: best role score × league factor)
        role_scores_vals = [v for k, v in rec.items() if k.startswith("_role_") and pd.notna(v)]
        base_score = float(max(role_scores_vals)) if role_scores_vals else (rec.get("Complete Score") or 0.0)
        ls = float(LEAGUE_STRENGTHS.get(lg, 70.0))
        ls_norm = np.clip(ls / 100.0, 0.30, 1.00)
        rec["_base_score"] = float(base_score)
        rec["_league_factor"] = float(ls_norm ** 1.6)  # gamma ~= 1.6 for beta=0.4
        rec["Impact Score"] = float(base_score * ls_norm ** 1.6)

        result_rows.append(rec)

    if not result_rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(result_rows)

    # Normalise Impact Score to 0–100 within team
    imp = pd.to_numeric(df_out["Impact Score"], errors="coerce")
    lo, hi = imp.min(), imp.max()
    if pd.notna(lo) and pd.notna(hi) and hi > lo:
        df_out["Impact Score"] = 100.0 * (imp - lo) / (hi - lo)
    else:
        df_out["Impact Score"] = imp.fillna(0.0)

    return df_out


def _rank_val_fmt(v) -> str:
    try:
        v = float(v)
        if np.isnan(v): return "—"
        if abs(v) >= 100: return f"{v:.0f}"
        if abs(v) >= 10:  return f"{v:.1f}"
        if abs(v) >= 1:   return f"{v:.2f}"
        return f"{v:.3f}"
    except:
        return "—"


def _rank_score_color(v: float) -> str:
    v = max(0.0, min(100.0, float(v)))
    if v <= 50:
        t = v / 50.0
        r = int(239 + (234 - 239) * t)
        g = int(68  + (179 - 68)  * t)
        b = int(68  + (8   - 68)  * t)
    else:
        t = (v - 50) / 50.0
        r = int(234 + (34  - 234) * t)
        g = int(179 + (197 - 179) * t)
        b = int(8   + (94  - 8)   * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _make_team_ranking_image(
    df_show: pd.DataFrame,
    rank_col: str,
    value_col: str,
    title_lines: list,
    theme: str,
    export_mode: str,
    show_age: bool,
    highlight_names: list,
    show_score_pill: bool,
    photo_func,
    badge_func,
) -> bytes:
    df_top = df_show.head(10).copy()  # default top 10
    if df_top.empty:
        return b""

    hi_set = {str(x).strip().lower() for x in (highlight_names or [])}

    def is_hi(row):
        return str(row.get("Player", "")).strip().lower() in hi_set

    if theme == "Dark":
        BG = "#0a0f1c"
        ROW_A, ROW_B = "#0f1628", "#0b1222"
        TXT, SUB, FOOT = "#ffffff", "#b8c0cf", "#9aa6bd"
        DIV = "#23304a"
        BAR_BG = "#1a2540"
        RANK_BG, RANK_EDGE = "#111a2e", "#2b3a5a"
        HILITE, HILITE_E = "#f6d46b", "#d2a100"
    else:
        BG = "#ffffff"
        ROW_A, ROW_B = "#f7f7f7", "#ffffff"
        TXT, SUB, FOOT = "#111111", "#777777", "#9b9b9b"
        DIV = "#e2e2e2"
        BAR_BG = "#e1e1e1"
        RANK_BG, RANK_EDGE = "#f3f3f3", "#c0c0c0"
        HILITE, HILITE_E = "#f6d46b", "#d2a100"

    # get score values for bar scaling
    scores = pd.to_numeric(df_top[rank_col], errors="coerce")
    max_score = float(scores.max()) if scores.notna().any() else 1.0

    footer_lines = [
        f"Ranking metric: {rank_col}  ·  Percentile ranks vs same-league, same-position pool",
        "Wyscout data  ·  Role scores = weighted metric percentiles per position bucket",
    ]

    N = len(df_top)

    # ── 1920×1080 mode ────────────────────────────────────────────────────────
    if export_mode == "1920×1080 (banner)":
        DPI = 100
        fig = plt.figure(figsize=(19.2, 10.8), dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, color=BG, zorder=0))

        L, R = 0.04, 0.96
        t1 = title_lines[0].upper() if title_lines else ""
        t2 = title_lines[1].upper() if len(title_lines) > 1 else ""
        t3 = title_lines[2].upper() if len(title_lines) > 2 else ""

        ax.text(L, 0.975, t1, fontsize=46, fontweight="bold", color=TXT, ha="left", va="top")
        ax.text(L, 0.918, t2, fontsize=32, fontweight="bold", color=TXT, ha="left", va="top")
        ax.text(L, 0.878, t3, fontsize=19, color=SUB, ha="left", va="top")

        hd_y, ft_y = 0.838, 0.040
        ax.plot([L, R], [hd_y]*2, color=DIV, lw=2)
        ax.plot([L, R], [ft_y]*2, color=DIV, lw=2)
        for i, line in enumerate(footer_lines):
            ax.text(L, ft_y - 0.018 - i * 0.023, line, fontsize=12, color=FOOT, ha="left", va="top")

        row_gap = (hd_y - ft_y - 0.01) / min(N, 15)
        row_h   = row_gap * 0.95
        ROW_TOP = hd_y - 0.015

        RANK_X  = L + 0.022
        BADGE_X = L + 0.100
        PHOTO_X = L + 0.175
        NAME_X  = L + 0.250

        BAR_L = L + 0.620
        BAR_R = R - 0.140
        BAR_H = row_h * 0.25
        VAL_X = R - 0.025

        for i, (_, row) in enumerate(df_top.iterrows()):
            y = ROW_TOP - (i + 0.5) * row_gap
            ax.add_patch(Rectangle((L, y - row_h/2), R-L, row_h,
                                   color=(ROW_A if i%2==0 else ROW_B), zorder=1))
            if is_hi(row):
                ax.add_patch(Rectangle((L, y-row_h/2), R-L, row_h,
                                       color=HILITE, alpha=0.20, zorder=2))
                ax.add_patch(Rectangle((L, y-row_h/2), R-L, row_h,
                                       fill=False, edgecolor=HILITE_E, lw=2.2, zorder=3))

            # Rank circle
            ax.scatter([RANK_X], [y], s=1200,
                       facecolor=RANK_BG, edgecolor=(HILITE_E if is_hi(row) else RANK_EDGE),
                       linewidths=2, zorder=4)
            ax.text(RANK_X, y, str(i+1), fontsize=15, fontweight="bold",
                    color=TXT, ha="center", va="center", zorder=5)

            # Team badge
            badge = badge_func(row)
            if badge is not None:
                zz = min(40.0/max(badge.shape[0], badge.shape[1], 1), 0.5)
                ax.add_artist(AnnotationBbox(OffsetImage(badge, zoom=zz),
                                             (BADGE_X, y), frameon=False, zorder=5))

            # Player photo
            photo = photo_func(row)
            if photo is not None:
                zz = min(48.0/max(photo.shape[0], photo.shape[1], 1), 0.5)
                ax.add_artist(AnnotationBbox(OffsetImage(photo, zoom=zz),
                                             (PHOTO_X, y), frameon=False, zorder=5))

            # Name + meta
            player_name = str(row.get("Player", "")).upper()
            pos_str = str(row.get("Position", "")).split(",")[0].strip().upper()
            team_str = str(row.get("Team", ""))
            meta = pos_str
            if show_age and pd.notna(row.get("Age")):
                meta += f"  ·  Age {int(float(row.get('Age', 0)))}"
            ax.text(NAME_X, y + row_h*0.20, player_name,
                    fontsize=26, fontweight="bold", color=TXT, ha="left", va="center", zorder=6)
            ax.text(NAME_X, y - row_h*0.23, meta,
                    fontsize=18, color=SUB, ha="left", va="center", zorder=6)

            # Bar
            v_bar = float(row[rank_col]) if pd.notna(row.get(rank_col)) else 0.0
            frac = min(1.0, max(0.0, v_bar / max_score)) if max_score else 0.0
            bar_col = _rank_score_color(v_bar) if show_score_pill else "#6b7cff"
            ax.add_patch(Rectangle((BAR_L, y-BAR_H/2), BAR_R-BAR_L, BAR_H, color=BAR_BG, zorder=2))
            ax.add_patch(Rectangle((BAR_L, y-BAR_H/2), (BAR_R-BAR_L)*frac, BAR_H,
                                   color=bar_col, zorder=3))

            # Value
            v_disp = row.get(value_col, row.get(rank_col))
            ax.text(VAL_X, y, _rank_val_fmt(v_disp),
                    fontsize=28, fontweight="bold", color=TXT, ha="right", va="center", zorder=6)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, facecolor=BG, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    # ── Standard auto-height mode ─────────────────────────────────────────────
    N = min(N, 10)  # top 10 default
    df_top = df_top.head(N)
    ROW_H    = 0.85
    HEADER_H = 1.75
    FOOT_H   = 0.72
    TOTAL_H  = HEADER_H + N * ROW_H + FOOT_H

    fig = plt.figure(figsize=(8.5, TOTAL_H), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1.0); ax.set_ylim(0, TOTAL_H); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1.0, TOTAL_H, color=BG, zorder=0))

    t1 = title_lines[0].upper() if title_lines else ""
    t2 = title_lines[1].upper() if len(title_lines) > 1 else ""
    t3 = title_lines[2].upper() if len(title_lines) > 2 else ""

    ty = TOTAL_H - 0.25
    ax.text(0.04, ty,        t1, fontsize=20, fontweight="bold", color=TXT, ha="left", va="top")
    ax.text(0.04, ty - 0.36, t2, fontsize=14, fontweight="bold", color=TXT, ha="left", va="top")
    ax.text(0.04, ty - 0.65, t3, fontsize=11, color=SUB,         ha="left", va="top")

    base_y = TOTAL_H - HEADER_H
    ax.plot([0.04, 0.96], [base_y + ROW_H/2 + 0.04]*2, color=DIV, lw=1.1, zorder=2)

    L, R = 0.04, 0.96
    RANK_X   = 0.075
    BADGE_X  = 0.135
    PHOTO_X  = 0.190
    NAME_X   = 0.240
    BAR_L, BAR_R = 0.64, 0.82
    BAR_H    = 0.14
    VAL_X    = 0.95

    for i, (_, row) in enumerate(df_top.iterrows()):
        y = base_y - i * ROW_H
        ax.add_patch(Rectangle((L, y-ROW_H/2), R-L, ROW_H,
                               color=(ROW_A if i%2==0 else ROW_B), zorder=1))
        if is_hi(row):
            ax.add_patch(Rectangle((L, y-ROW_H/2), R-L, ROW_H,
                                   color=HILITE, alpha=0.25, zorder=2))
            ax.add_patch(Rectangle((L, y-ROW_H/2), R-L, ROW_H,
                                   fill=False, edgecolor=HILITE_E, lw=1.3, zorder=3))

        ax.scatter([RANK_X], [y], s=500,
                   facecolor=RANK_BG, edgecolor=(HILITE_E if is_hi(row) else RANK_EDGE),
                   linewidths=1.2, zorder=4)
        ax.text(RANK_X, y, str(i+1), fontsize=10, fontweight="bold",
                color=TXT, ha="center", va="center", zorder=5)

        # Badge (small)
        badge = badge_func(row)
        if badge is not None:
            zz = min(22.0/max(badge.shape[0], badge.shape[1], 1), 0.4)
            ax.add_artist(AnnotationBbox(OffsetImage(badge, zoom=zz),
                                         (BADGE_X, y), frameon=False, zorder=5))

        # Photo (small)
        photo = photo_func(row)
        if photo is not None:
            zz = min(28.0/max(photo.shape[0], photo.shape[1], 1), 0.4)
            ax.add_artist(AnnotationBbox(OffsetImage(photo, zoom=zz),
                                         (PHOTO_X, y), frameon=False, zorder=5))

        player_name = str(row.get("Player", "")).upper()
        pos_str = str(row.get("Position", "")).split(",")[0].strip().upper()
        meta = pos_str
        if show_age and pd.notna(row.get("Age")):
            meta += f"  ·  {int(float(row.get('Age', 0)))}y"

        ax.text(NAME_X, y + 0.13, player_name,
                fontsize=16, fontweight="bold", color=TXT, ha="left", va="center", zorder=5)
        ax.text(NAME_X, y - 0.12, meta,
                fontsize=11, color=SUB, ha="left", va="center", zorder=5)

        v_bar = float(row[rank_col]) if pd.notna(row.get(rank_col)) else 0.0
        frac  = min(1.0, max(0.0, v_bar / max_score)) if max_score else 0.0
        bar_col = _rank_score_color(v_bar) if show_score_pill else "#6b7cff"
        ax.add_patch(Rectangle((BAR_L, y-BAR_H/2), BAR_R-BAR_L, BAR_H, color=BAR_BG, zorder=2))
        ax.add_patch(Rectangle((BAR_L, y-BAR_H/2), (BAR_R-BAR_L)*frac, BAR_H,
                               color=bar_col, zorder=3))

        v_disp = row.get(value_col, row.get(rank_col))
        ax.text(VAL_X, y, _rank_val_fmt(v_disp),
                fontsize=15, fontweight="bold", color=TXT, ha="right", va="center", zorder=6)

    ax.plot([L, R], [0.82]*2, color=DIV, lw=0.9, zorder=2)
    for j, line in enumerate(footer_lines):
        ax.text(L, 0.64 - j*0.19, line, fontsize=9.5, color=FOOT, ha="left", va="top", zorder=4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER SECTION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<div style="font-size:11px;font-weight:900;letter-spacing:.18em;color:#6b7280;'
    'text-transform:uppercase;margin-bottom:10px;">Team Player Rankings</div>',
    unsafe_allow_html=True,
)

if df_players is None:
    st.info("Upload a player CSV in the sidebar to use Team Player Rankings.")
else:
    _rank_team_players = df_players[df_players["Team"] == sel_team].copy()
    if _rank_team_players.empty:
        _tm = sel_team.lower()
        _match = df_players[df_players["Team"].str.lower().str.contains(_tm[:8], na=False)]
        if not _match.empty:
            _rank_team_players = df_players[df_players["Team"] == _match["Team"].iloc[0]].copy()

    if _rank_team_players.empty:
        st.info(f"No player data found for **{sel_team}**.")
    else:
        # ── Controls ──────────────────────────────────────────────────────────
        _rc1, _rc2, _rc3 = st.columns([1.4, 1.2, 1.2])
        with _rc1:
            _rank_pos_group = st.selectbox(
                "Position group",
                list(_RANK_POS_GROUPS.keys()),
                index=0,
                key="tr_pos_group",
            )
        with _rc2:
            _rank_min_mins = st.slider("Min minutes", 0, 3000, 0, 50, key="tr_min_mins")
        with _rc3:
            _rank_theme = st.selectbox("Theme", ["Light", "Dark"], key="tr_theme")

        # Build the scored DataFrame — rebuilds whenever team, league or min-mins changes
        _rank_cache_key = (sel_team, t_league, _rank_min_mins)
        if (st.session_state.get("_tr_cache_key") != _rank_cache_key
                or st.session_state.get("_tr_df") is None):
            st.session_state.pop("_tr_img_bytes", None)  # clear stale image on team change
            with st.spinner("Computing team player scores…"):
                try:
                    _tr_df = _build_team_rank_df(
                        team_players_bytes=_rank_team_players.to_json().encode(),
                        all_players_bytes=df_players.to_json().encode(),
                        team=sel_team,
                        league=t_league,
                        min_mins=_rank_min_mins,
                    )
                    st.session_state["_tr_df"] = _tr_df
                    st.session_state["_tr_cache_key"] = _rank_cache_key
                except Exception as _e:
                    st.error(f"Error building scores: {_e}")
                    _tr_df = pd.DataFrame()
        else:
            _tr_df = st.session_state["_tr_df"]

        if _tr_df is None or _tr_df.empty:
            st.info("No scored player data available.")
        else:
            # Filter by position group
            _allowed_pos = _RANK_POS_GROUPS[_rank_pos_group]
            if _allowed_pos is not None:
                _tr_filtered = _tr_df[_tr_df["_ftok"].isin(_allowed_pos)].copy()
            else:
                _tr_filtered = _tr_df.copy()

            if _tr_filtered.empty:
                st.info(f"No players in position group: {_rank_pos_group}")
            else:
                # ── Ranking metric selector ───────────────────────────────────
                _composite_opts = ["Complete Score", "Impact Score"]

                # Add role score options for selected group
                if _allowed_pos is not None:
                    _pk = _RANK_GROUP_TO_KEY.get(_rank_pos_group, "CM")
                    for _rn in ROLE_BUCKETS.get(_role_key(list(_allowed_pos)[0]) if _allowed_pos else "CM", {}):
                        _rc_name = f"_role_{_rn}"
                        if _rc_name in _tr_filtered.columns:
                            _composite_opts.append(_rc_name)

                # Raw metric columns
                _raw_met_key = _RANK_GROUP_TO_KEY.get(_rank_pos_group, "CM")
                _raw_met_pairs = _RANK_POS_METRICS.get(_raw_met_key, _RANK_POS_METRICS["CM"])
                _raw_met_cols = [met for _, met in _raw_met_pairs if met in _tr_filtered.columns]

                _rk1, _rk2 = st.columns([1.4, 2])
                with _rk1:
                    _rank_mode = st.radio(
                        "Rank by",
                        ["Composite score", "Raw metric"],
                        horizontal=True,
                        key="tr_rank_mode",
                    )
                with _rk2:
                    if _rank_mode == "Composite score":
                        # Pretty-print role names
                        def _pretty_composite(c):
                            if c.startswith("_role_"):
                                return c.replace("_role_", "")
                            return c
                        _rank_col_sel = st.selectbox(
                            "Metric",
                            _composite_opts,
                            format_func=_pretty_composite,
                            key="tr_comp_metric",
                        )
                        _value_col = _rank_col_sel
                    else:
                        _rank_col_sel = st.selectbox(
                            "Raw metric",
                            _raw_met_cols if _raw_met_cols else list(_tr_filtered.columns),
                            key="tr_raw_metric",
                        )
                        _value_col = _rank_col_sel

                # Sort
                _tr_sorted = _tr_filtered.copy()
                if _rank_col_sel in _tr_sorted.columns:
                    _tr_sorted[_rank_col_sel] = pd.to_numeric(_tr_sorted[_rank_col_sel], errors="coerce")
                    _tr_sorted = _tr_sorted.sort_values(_rank_col_sel, ascending=False).dropna(subset=[_rank_col_sel])

                # ── Image export controls ────────────────────────────────────
                _exp1, _exp2, _exp3 = st.columns([1.4, 1.2, 1.2])
                with _exp1:
                    _rank_export_mode = st.selectbox(
                        "Export format",
                        ["Standard (auto)", "1920×1080 (banner)"],
                        key="tr_export_mode",
                    )
                with _exp2:
                    _rank_show_age = st.toggle("Show age in image", False, key="tr_show_age")
                with _exp3:
                    _rank_score_pill = st.toggle("Color bar by score", True, key="tr_score_pill")

                _rank_hi_enabled = st.checkbox("Highlight players", False, key="tr_hi_enabled")
                _rank_hi_names = []
                if _rank_hi_enabled:
                    _rank_hi_names = st.multiselect(
                        "Players to highlight",
                        sorted(_tr_sorted["Player"].dropna().astype(str).unique()),
                        key="tr_hi_names",
                    )

                # Title lines
                _metric_label = _rank_col_sel.replace("_role_", "") if _rank_col_sel.startswith("_role_") else _rank_col_sel
                _rt1, _rt2, _rt3 = st.columns(3)
                with _rt1:
                    _rank_t1 = st.text_input("Title 1", f"{sel_team.upper()} SQUAD", key="tr_t1")
                with _rt2:
                    _rank_t2 = st.text_input("Title 2", _metric_label.upper(), key="tr_t2")
                with _rt3:
                    _rank_t3 = st.text_input("Title 3", _rank_pos_group.upper(), key="tr_t3")

                # ── Photo/badge helpers (using existing resolve_player_photo) ──
                def _tr_photo_func(row):
                    key_id = f"{_norm_str(str(row.get('Player','')))}|{_norm_str(str(row.get('Team','')))}"
                    url = resolve_player_photo(
                        player=str(row.get("Player", "")),
                        team=str(row.get("Team", "")),
                        league=str(row.get("League", "")),
                        key_id=key_id,
                        session_photo_map=st.session_state.setdefault("photo_map", {}),
                        global_overrides={},
                    )
                    if url and url.startswith("http"):
                        img = load_remote_img(url)
                        return img
                    return None

                def _tr_badge_func(row):
                    return get_team_badge(str(row.get("Team", sel_team)))

                # ── Generate image ────────────────────────────────────────────
                if st.button("🖼 Generate Ranking Image", key="tr_gen_btn"):
                    with st.spinner("Generating image…"):
                        _rank_img_bytes = _make_team_ranking_image(
                            df_show=_tr_sorted,
                            rank_col=_rank_col_sel,
                            value_col=_value_col,
                            title_lines=[_rank_t1, _rank_t2, _rank_t3],
                            theme=_rank_theme,
                            export_mode=_rank_export_mode,
                            show_age=_rank_show_age,
                            highlight_names=_rank_hi_names,
                            show_score_pill=_rank_score_pill,
                            photo_func=_tr_photo_func,
                            badge_func=_tr_badge_func,
                        )
                        st.session_state["_tr_img_bytes"] = _rank_img_bytes

                if st.session_state.get("_tr_img_bytes"):
                    st.image(st.session_state["_tr_img_bytes"], use_column_width=True)
                    st.download_button(
                        "⬇️ Download Ranking Image",
                        data=st.session_state["_tr_img_bytes"],
                        file_name=f"{sel_team.replace(' ','_')}_player_rankings.png",
                        mime="image/png",
                        key="tr_dl",
                    )

                # ── Metric breakdown table ─────────────────────────────────────
                with st.expander("📊 Full Metric Breakdown", expanded=False):
                    _show_raw_cols = ["Player", "Position", "Minutes played", "Age"]
                    _show_raw_cols += ["Complete Score", "Impact Score"]

                    # Role score columns
                    _role_cols = [c for c in _tr_sorted.columns if c.startswith("_role_")]
                    _show_raw_cols += _role_cols

                    # Raw metric columns (available)
                    for _, met in _raw_met_pairs:
                        if met in _tr_sorted.columns:
                            _show_raw_cols.append(met)
                        pct_col = f"_pct_{met}"
                        if pct_col in _tr_sorted.columns:
                            _show_raw_cols.append(pct_col)

                    _show_raw_cols = list(dict.fromkeys(c for c in _show_raw_cols if c in _tr_sorted.columns))

                    # Rename role cols for display
                    _rename_map = {c: c.replace("_role_", "🎯 ") for c in _role_cols}
                    _rename_map.update({f"_pct_{m}": f"Pct: {m}" for m in [met for _, met in _raw_met_pairs]})

                    st.dataframe(
                        _tr_sorted[_show_raw_cols]
                            .rename(columns=_rename_map)
                            .reset_index(drop=True),
                        use_container_width=True,
                    )

# ═══════════════════════════════════════════════════════════════════════════════
# END TEAM PLAYER RANKINGS SECTION
# ═══════════════════════════════════════════════════════════════════════════════

