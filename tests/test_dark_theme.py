"""Dark-theme contrast and chart presentation without changing recorded data."""
from pathlib import Path
import tomllib

import pytest

from ai_agent_pro.presentation import theme
from ai_agent_pro.presentation.dashboard import chart_spec


def contrast(first, second):
    def luminance(value):
        channels = [int(value[i:i+2],16)/255 for i in (1,3,5)]
        linear = [v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in channels]
        return sum(c*w for c,w in zip(linear,(.2126,.7152,.0722)))
    light,dark=sorted((luminance(first),luminance(second)),reverse=True)
    return (light+.05)/(dark+.05)


@pytest.mark.parametrize('background',[theme.BACKGROUND,theme.SURFACE,'#17243b'])
def test_dark_text_and_secondary_copy_have_accessible_contrast(background):
    assert contrast(theme.TEXT,background)>=4.5
    assert contrast(theme.MUTED,background)>=4.5


@pytest.mark.parametrize('color',['#3155c5','#6740b7','#3a60d5','#754bc7'])
def test_primary_button_white_text_contrast(color):
    assert contrast('#ffffff',color)>=4.5


def test_native_dark_theme_and_static_private_asset_free_styles():
    config=tomllib.loads((Path(__file__).resolve().parents[1]/'.streamlit/config.toml').read_text())
    assert config['theme']['base']=='dark'
    assert config['theme']['backgroundColor'].lower()==theme.BACKGROUND
    assert config['theme']['textColor'].lower()==theme.TEXT
    assert 'url(' not in theme.DARK_STYLE and '<script' not in theme.DARK_STYLE
    assert ':focus-visible' in theme.DARK_STYLE and 'prefers-reduced-motion' in theme.DARK_STYLE
    assert config['server']=={'address':'127.0.0.1','maxUploadSize':10}
    assert config['browser']['gatherUsageStats'] is False


def test_dark_chart_axes_and_palette_preserve_supplied_measurements():
    rows=[{'day':'2026-01-01','status':'completed','tasks':3}]
    spec=chart_spec(rows,'activity').to_dict()
    assert spec['data']['values']==rows
    assert spec['config']['background']=='transparent'
    assert spec['config']['axis']['labelColor']==theme.MUTED
    assert spec['encoding']['color']['scale']['range']==[
        theme.BLUE,theme.GREEN,theme.RED,theme.AMBER,theme.VIOLET]


def test_form_field_text_placeholder_and_persistent_border_contrast():
    assert contrast(theme.TEXT,theme.FIELD_BACKGROUND)>=4.5
    assert contrast(theme.FIELD_PLACEHOLDER,theme.FIELD_BACKGROUND)>=4.5
    assert contrast(theme.FIELD_BORDER,theme.FIELD_BACKGROUND)>=3
    assert contrast(theme.FIELD_BORDER,theme.SURFACE)>=3
    assert contrast(theme.DISABLED_TEXT,theme.DISABLED_BACKGROUND)>=4.5
