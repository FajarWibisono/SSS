# config.py - Konfigurasi untuk Enhanced SWING Screener

# Daftar saham default untuk screening
DEFAULT_STOCKS = [
    # Banking
    'BBCA.JK', 'BMRI.JK', 'BBRI.JK', 'BBNI.JK', 'BRIS.JK',

    # Telekomunikasi
    'TLKM.JK', 'ISAT.JK', 'EXCL.JK',

    # Consumer Goods
    'UNVR.JK', 'ICBP.JK', 'INDF.JK', 'KLBF.JK', 'KAEF.JK',

    # Automotive & Components
    'ASII.JK', 'AUTO.JK', 'IMAS.JK',

    # Mining
    'ANTM.JK', 'INCO.JK', 'TINS.JK', 'PTBA.JK',

    # Energy
    'PGAS.JK', 'AKRA.JK',

    # Property
    'BSDE.JK', 'LPKR.JK', 'PWON.JK',

    # Technology
    'GOTO.JK', 'BUKA.JK',

    # Cement
    'SMGR.JK', 'INTP.JK',

    # Tobacco
    'GGRM.JK', 'HMSP.JK'
]

# Parameter default untuk screening
DEFAULT_PARAMS = {
    'weekly_rsi_min': 52,
    'weekly_adx_min': 18,
    'weekly_uo_min': 45,
    'daily_rsi_min': 50,
    'daily_rsi_max': 75,
    'volume_multiplier': 1.2,
    'market_cap_threshold': 500000000,  # 500M IDR
    'weekly_score_threshold': 7,
    'daily_score_threshold': 6
}

# Konfigurasi chart
CHART_CONFIG = {
    'period': '3mo',
    'interval': '1d',
    'height': 600,
    'colors': {
        'sma5': 'orange',
        'sma20': 'red',
        'rsi': 'purple',
        'volume': 'blue'
    }
}

# Konfigurasi export
EXPORT_CONFIG = {
    'filename_format': 'swing_screening_results_%Y%m%d_%H%M.csv',
    'columns_to_export': [
        'Ticker', 'Price', 'Market_Cap', 'Volume_Ratio',
        'Daily_RSI', 'Weekly_RSI', 'Weekly_ADX', 'Weekly_UO',
        'Weekly_Score', 'Daily_Score', 'Total_Score'
    ]
}
