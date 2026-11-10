"""Static Dark AI palette and CSS; no user content or external assets."""

BACKGROUND = '#090f20'
SURFACE = '#111d32'
TEXT = '#edf3ff'
MUTED = '#afbed6'
BLUE = '#8ab4ff'
VIOLET = '#b6a2ff'
GREEN = '#6cdbb2'
RED = '#ff98a6'
AMBER = '#f6cb7b'
FIELD_BACKGROUND = '#182a43'
FIELD_BORDER = '#637f9f'
FIELD_PLACEHOLDER = '#b8c9e2'
DISABLED_BACKGROUND = '#101d31'
DISABLED_TEXT = '#b4c4dc'

DARK_STYLE = '''<style>
:root {
 --ai-bg:#090f20; --ai-panel:#111d32; --ai-raised:#17243b;
 --ai-border:#30425f; --ai-text:#edf3ff; --ai-muted:#afbed6;
 --ai-blue:#8ab4ff; --ai-violet:#b6a2ff;
}
.stApp {color-scheme:dark; color:var(--ai-text);
 background:radial-gradient(ellipse at 12% 0%,rgba(49,91,184,.14),transparent 48%),
 radial-gradient(ellipse at 96% 8%,rgba(112,70,180,.12),transparent 46%),var(--ai-bg)}
[data-testid="stHeader"] {background:rgba(9,15,32,.96); border-bottom:1px solid rgba(100,130,180,.12)}
[data-testid="stMainBlockContainer"] {max-width:1600px; padding-top:2.6rem; padding-bottom:3rem}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#101a30,#0b1325 65%);
 border-right:1px solid #273751}
.st-key-brand {border-bottom:1px solid #30425f; padding-bottom:1.1rem; margin-bottom:.6rem}
.st-key-brand h1 {font-size:1.45rem; letter-spacing:-.045em; font-weight:700; color:var(--ai-text)}
.st-key-brand h1::before {content:""; display:inline-block; width:.65em; height:.65em;
 margin-right:.55em; border-radius:4px; background:linear-gradient(135deg,#83b4ff,#a38aff)}
.st-key-brand [data-testid="stCaptionContainer"] {font-size:.68rem; letter-spacing:.12em; color:#a9bada}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.25rem}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:.5rem .65rem;
 border:1px solid transparent; border-radius:10px; min-height:42px}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#172640}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
 background:linear-gradient(100deg,#1b3053,#242744); border-color:#445a83}
h1,h2,h3,h4 {color:var(--ai-text); letter-spacing:-.035em}
h2 {font-size:clamp(1.85rem,3vw,2.55rem); font-weight:700}
h3 {font-size:1.3rem; font-weight:650}
[data-testid="stCaptionContainer"] {color:var(--ai-muted); line-height:1.6}
[data-testid="stCaptionContainer"] p {color:inherit}
[data-testid="stWidgetLabel"] p {color:#dbe6fa; font-weight:550}
a {color:#a6c7ff}
a:visited {color:#c8b8ff}
hr {border-color:#2b3b56}
.st-key-intro {border:1px solid #3a4e76; border-left:4px solid #8ab4ff;
 padding:1.55rem 1.65rem; border-radius:18px;
 background:radial-gradient(ellipse at 95% 0%,rgba(132,92,215,.16),transparent 58%),
 linear-gradient(115deg,#152744,#111d33);
 box-shadow:0 16px 38px rgba(0,5,18,.22)}
.st-key-intro h3 {font-size:clamp(1.3rem,2.2vw,1.8rem)}
.st-key-intro p {color:#d2dff5; line-height:1.65}
.st-key-metrics {margin:.9rem 0 1.1rem}
[data-testid="stMetric"] {min-width:0; min-height:8.2rem; padding:1.1rem; border:1px solid #30425f;
 border-radius:15px; background:linear-gradient(140deg,#15233b,#101b2e);
 box-shadow:0 6px 20px rgba(0,5,18,.14)}
[data-testid="stMetricLabel"] p {color:var(--ai-muted); white-space:normal; line-height:1.5}
[data-testid="stMetricValue"] {font-variant-numeric:tabular-nums; letter-spacing:-.045em; color:var(--ai-text)}
[data-testid="stMetricValue"] > div {font-size:clamp(1.4rem,2.3vw,2rem); white-space:normal; overflow-wrap:anywhere}
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stForm"] {
 border-color:#30425f!important; border-radius:16px; background:#111d32}
[data-testid="stExpander"] {border:1px solid #30425f; border-radius:13px; background:#101b2e}
[data-testid="stExpander"] summary {color:var(--ai-text); padding:.8rem 1rem}
[data-testid="stExpander"] summary:hover {background:#182740; border-radius:12px}
[data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button {
 min-height:42px; border-radius:10px; border:1px solid #405573;
 background:#172841; color:var(--ai-text); font-weight:550;
 transition:border-color .12s ease,background-color .12s ease}
[data-testid="stButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover {
 border-color:#8ab4ff; background:#203656; color:#fff}
button[kind="primary"], [data-testid="stBaseButton-primary"] {
 background:linear-gradient(110deg,#3155c5,#6740b7)!important;
 border:1px solid #899bff!important; color:#fff!important;
 box-shadow:0 4px 14px rgba(44,63,151,.18)}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
 background:linear-gradient(110deg,#3a60d5,#754bc7)!important; border-color:#c0caff!important}
button:disabled {opacity:.55; box-shadow:none!important; cursor:not-allowed}
button:focus-visible, input:focus-visible, textarea:focus-visible, summary:focus-visible,
[role="tab"]:focus-visible {outline:3px solid #b6a2ff!important; outline-offset:3px}
/* Cover current React-Aria roots as well as older BaseWeb controls. Paint
   the whole control at rest, rather than depending on its focus indicator. */
:is([data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
 [data-testid="stNumberInputContainer"], [data-testid="stChatInput"],
 .stSelectbox [role="group"], .stMultiSelect [role="group"],
 [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div) {
 background-color:#182a43; border:1px solid #637f9f; border-radius:10px;
 color:#edf3ff; opacity:1}
:is([data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
 [data-testid="stNumberInputContainer"], [data-testid="stChatInput"],
 .stSelectbox [role="group"], .stMultiSelect [role="group"],
 [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div):hover {border-color:#7f9aba}
:is([data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
 [data-testid="stNumberInputContainer"], [data-testid="stChatInput"],
 .stSelectbox [role="group"], .stMultiSelect [role="group"],
 [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div):focus-within {
 border-color:#8ab4ff; box-shadow:0 0 0 2px rgba(138,180,255,.22)}
:is([data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
 [data-testid="stNumberInputContainer"], [data-testid="stChatInput"],
 .stSelectbox [role="group"], .stMultiSelect [role="group"],
 [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div) input,
:is([data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
 [data-testid="stNumberInputContainer"], [data-testid="stChatInput"],
 .stSelectbox [role="group"], .stMultiSelect [role="group"],
 [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div) textarea {background-color:transparent; color:#edf3ff; caret-color:#b6a2ff}
input::placeholder, textarea::placeholder {color:#b8c9e2!important; opacity:1}
:is([data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
 [data-testid="stNumberInputContainer"], [data-testid="stChatInput"],
 .stSelectbox [role="group"], .stMultiSelect [role="group"],
 [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div):has(input:disabled),
:is([data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
 [data-testid="stNumberInputContainer"], [data-testid="stChatInput"],
 .stSelectbox [role="group"], .stMultiSelect [role="group"],
 [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div):has(textarea:disabled) {
 background-color:#101d31; border-color:#60748f; box-shadow:none; opacity:1; cursor:not-allowed}
input:disabled, textarea:disabled {
 color:#b4c4dc!important; -webkit-text-fill-color:#b4c4dc!important; opacity:1; cursor:not-allowed}
input:disabled::placeholder, textarea:disabled::placeholder {
 color:#a8bad4!important; -webkit-text-fill-color:#a8bad4!important; opacity:1}
[role="listbox"], [data-baseweb="popover"] {background:#15233b; color:var(--ai-text);
 border:1px solid #405573; border-radius:12px; box-shadow:0 14px 36px rgba(0,5,18,.35)}
[role="option"] {color:var(--ai-text)}
[role="option"]:hover, [role="option"][data-focused="true"] {background:#253c60}
[role="option"][aria-selected="true"] {background:#29385b; color:#fff}
[role="tablist"] {border-bottom:1px solid #30425f; gap:1.2rem}
[role="tab"] {color:var(--ai-muted)}
[role="tab"][aria-selected="true"] {color:#b5cfff}
[data-testid="stDataFrame"], [data-testid="stTable"] {border:1px solid #30425f; border-radius:12px}
[data-testid="stTable"] th {background:#1b2b45; color:var(--ai-text)}
[data-testid="stTable"] td {color:var(--ai-text); border-color:#30425f}
[data-testid="stChatMessage"] {background:#111d32; border:1px solid #2b3d59; border-radius:16px; padding:1.15rem}
[data-testid="stChatInput"] {border-radius:14px}
[data-testid="stBottomBlockContainer"] {background:var(--ai-bg)}
[data-testid="stCode"] {border:1px solid #30425f; border-radius:12px}
[data-testid="stFileUploaderDropzone"] {background:#111d32; border:1px dashed #4d6389; border-radius:13px}
[data-testid="stAlert"] {border-radius:12px; border:1px solid rgba(138,180,255,.24)}
@media (max-width:780px) {
 .st-key-metrics [data-testid="stHorizontalBlock"] {flex-wrap:wrap}
 .st-key-metrics [data-testid="stColumn"] {flex:1 1 calc(50% - 1rem)!important; min-width:0}
}
@media (max-width:640px) {
 [data-testid="stMainBlockContainer"] {padding:1.4rem 1rem 2rem}
 .st-key-intro {padding:1.1rem; border-radius:14px}
 [data-testid="stMetric"] {padding:.8rem}
 [data-testid="stChatMessage"] {padding:.8rem}
}
@media (prefers-reduced-motion:reduce) {
 .stApp *, .stApp *::before, .stApp *::after {animation:none!important; transition:none!important; scroll-behavior:auto!important}
}
</style>'''
