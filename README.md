# SSS

Aplikasi screener saham berbasis algoritma Enhanced SWING Trading dengan target success rate 81%+.

## 🚀 Fitur Utama

- **Two-Stage Screening**: Weekly trend analysis + Daily entry signals
- **Multi-Timeframe Analysis**: Kombinasi indikator weekly dan daily
- **Real-time Data**: Menggunakan Yahoo Finance API
- **Interactive Dashboard**: Interface yang user-friendly dengan Streamlit
- **Customizable Parameters**: Dapat disesuaikan sesuai preferensi trading
- **Visual Analysis**: Chart teknikal untuk analisis detail
- **Export Results**: Download hasil screening ke CSV

## 📋 Algoritma Screening

### Weekly Screening (9 Kriteria)
1. **Higher Highs & Higher Lows**: Konfirmasi uptrend
2. **Price > SMA20**: Harga di atas moving average 20 periode
3. **SMA Alignment**: SMA5 > SMA10 > SMA20 (trend bullish)
4. **RSI Bullish**: RSI > 52 (momentum positif)
5. **ADX Strong**: ADX > 18 (trend strength)
6. **Ultimate Oscillator**: UO > 45 (momentum bullish)
7. **MACD Bullish**: MACD line > Signal line
8. **Price > SMA50**: Konfirmasi trend jangka menengah
9. **Weekly Bullish Candle**: Close > Open

### Daily Screening (9 Kriteria)
1. **Price > Daily SMA20**: Harga di atas MA20 harian
2. **Price > Daily SMA5**: Harga di atas MA5 harian
3. **Price > Weekly SMA5**: Harga di atas MA5 mingguan
4. **RSI Range**: RSI antara 50-75 (momentum tanpa overbought)
5. **MACD Bullish**: MACD crossover bullish
6. **Volume Surge**: Volume > 1.2x rata-rata
7. **Bullish Candle**: Close > Open
8. **Normal Volatility**: ATR dalam batas wajar
9. **Strong Body**: Body candle > 40% dari range
