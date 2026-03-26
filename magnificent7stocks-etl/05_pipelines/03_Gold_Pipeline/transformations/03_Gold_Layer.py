import dlt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# =============================================================
# ✅ 1. SOURCE TABLE
# =============================================================
@dlt.table(
    name="03_Gold_Layer_silver_source",
    comment="Source data directly from the silver delta table."
)
def gold_layer_silver_source():
    df = spark.read.table("workspace.magnificent7stocks.02_silver_layer")

    return df.filter(F.col("datetime") >= F.add_months(F.current_timestamp(), -3))


# =============================================================
# ✅ 2. HOURLY RETURNS
# =============================================================
@dlt.table(
    name="03_Gold_Layer_hourly_returns",
    comment="Hourly returns for each stock."
)
def gold_layer_hourly_returns():
    df = dlt.read("03_Gold_Layer_silver_source")
    w = Window.partitionBy("ticker").orderBy("datetime")

    return (
        df.withColumn("prev_close", F.lag("close").over(w))
          .withColumn("price_change", F.col("close") - F.col("prev_close"))
          .withColumn("return_pct", F.col("price_change") / F.col("prev_close") * 100)
    )


# =============================================================
# ✅ 3. VOLATILITY METRICS
# =============================================================
@dlt.table(
    name="03_Gold_Layer_volatility",
    comment="Volatility metrics per hour."
)
def gold_layer_volatility():
    df = dlt.read("03_Gold_Layer_silver_source")

    return (
        df.withColumn("range", F.col("high") - F.col("low"))
          .withColumn("body_size", F.abs(F.col("close") - F.col("open")))
          .withColumn("range_pct", (F.col("high") - F.col("low")) / F.col("close") * 100)
    )


# =============================================================
# ✅ 4. ROLLING AVERAGES (3h, 6h)
# =============================================================
@dlt.table(
    name="03_Gold_Layer_rolling_averages",
    comment="Rolling moving averages for 3h and 6h windows."
)
def gold_layer_rolling_averages():
    df = dlt.read("03_Gold_Layer_silver_source")

    w3 = Window.partitionBy("ticker").orderBy("datetime").rowsBetween(-2, 0)
    w6 = Window.partitionBy("ticker").orderBy("datetime").rowsBetween(-5, 0)

    return (
        df.withColumn("ma3h", F.avg("close").over(w3))
          .withColumn("ma6h", F.avg("close").over(w6))
    )


# =============================================================
# ✅ 5. VOLUME SPIKES
# =============================================================
@dlt.table(
    name="03_Gold_Layer_volume_activity",
    comment="Volume spike detection using 6h average volume."
)
def gold_layer_volume_activity():
    df = dlt.read("03_Gold_Layer_silver_source")

    w6 = Window.partitionBy("ticker").orderBy("datetime").rowsBetween(-5, 0)

    return (
        df.withColumn("avg_volume_6h", F.avg("volume").over(w6))
          .withColumn("volume_spike_factor", F.col("volume") / F.col("avg_volume_6h"))
    )


# =============================================================
# ✅ 6. HOURLY RANKINGS
# =============================================================
@dlt.table(
    name="03_Gold_Layer_hourly_rankings",
    comment="Ranks stocks by price movement and volume for each hour."
)
def gold_layer_hourly_rankings():
    df = dlt.read("03_Gold_Layer_hourly_returns")
    w = Window.partitionBy("datetime")

    return (
        df.withColumn("price_change_rank", F.rank().over(w.orderBy(F.col("price_change").desc())))
          .withColumn("volume_rank", F.rank().over(w.orderBy(F.col("volume").desc())))
    )


# =============================================================
# ✅ 7. VWAP
# =============================================================
@dlt.table(
    name="03_Gold_Layer_vwap",
    comment="VWAP based on cumulative (price*volume) / cumulative volume."
)
def gold_layer_vwap():
    df = dlt.read("03_Gold_Layer_silver_source")

    w = Window.partitionBy("ticker").orderBy("datetime") \
              .rowsBetween(Window.unboundedPreceding, Window.currentRow)

    return (
        df.withColumn("cum_price_vol", F.sum(F.col("close") * F.col("volume")).over(w))
          .withColumn("cum_volume", F.sum("volume").over(w))
          .withColumn("vwap", F.col("cum_price_vol") / F.col("cum_volume"))
    )


# =============================================================
# ✅ 8. RSI
# =============================================================
@dlt.table(
    name="03_Gold_Layer_rsi",
    comment="RSI indicator using 14-period window."
)
def gold_layer_rsi():
    df = dlt.read("03_Gold_Layer_hourly_returns")

    w14 = Window.partitionBy("ticker").orderBy("datetime").rowsBetween(-13, 0)

    return (
        df.withColumn("gain", F.when(F.col("price_change") > 0, F.col("price_change")).otherwise(0))
          .withColumn("loss", F.when(F.col("price_change") < 0, -F.col("price_change")).otherwise(0))
          .withColumn("avg_gain", F.avg("gain").over(w14))
          .withColumn("avg_loss", F.avg("loss").over(w14))
          .withColumn("rs", F.col("avg_gain") / F.col("avg_loss"))
          .withColumn("rsi", 100 - (100 / (1 + F.col("rs"))))
    )


# =============================================================
# ✅ 9. MACD
# =============================================================
@dlt.table(
    name="03_Gold_Layer_macd",
    comment="MACD technical indicator."
)
def gold_layer_macd():
    df = dlt.read("03_Gold_Layer_silver_source")

    w = Window.partitionBy("ticker").orderBy("datetime")

    df = (
        df.withColumn("ema12", F.avg("close").over(w.rowsBetween(-11, 0)))
          .withColumn("ema26", F.avg("close").over(w.rowsBetween(-25, 0)))
    )

    df = df.withColumn("macd", F.col("ema12") - F.col("ema26"))

    df = df.withColumn(
        "signal",
        F.avg("macd").over(w.rowsBetween(-8, 0))
    )

    return df.withColumn("histogram", F.col("macd") - F.col("signal"))


# =============================================================
# ✅ 10. BOLLINGER BANDS
# =============================================================
@dlt.table(
    name="03_Gold_Layer_bollinger",
    comment="Bollinger Bands using a 20-period moving window."
)
def gold_layer_bollinger():
    df = dlt.read("03_Gold_Layer_silver_source")

    w20 = Window.partitionBy("ticker").orderBy("datetime").rowsBetween(-19, 0)

    return (
        df.withColumn("sma20", F.avg("close").over(w20))
          .withColumn("std20", F.stddev("close").over(w20))
          .withColumn("upper_band", F.col("sma20") + 2 * F.col("std20"))
          .withColumn("lower_band", F.col("sma20") - 2 * F.col("std20"))
    )


# =============================================================
# ✅ 11. CANDLES
# =============================================================
@dlt.table(
    name="03_Gold_Layer_candles",
    comment="Clean OHLCV dataset for dashboards."
)
def gold_layer_candles():
    df = dlt.read("03_Gold_Layer_silver_source")
    return df.select("datetime", "ticker", "open", "high", "low", "close", "volume")


# =============================================================
# ✅ 12. VALUE AT RISK (VaR)
# =============================================================
@dlt.table(
    name="03_Gold_Layer_value_at_risk",
    comment="Historical Value-at-Risk: 95% and 99% VaR based on hourly returns."
)
def gold_layer_value_at_risk():
    df = dlt.read("03_Gold_Layer_hourly_returns")

    w = (
        Window.partitionBy("ticker")
              .orderBy("datetime")
              .rowsBetween(-99, 0)
    )

    return (
        df.withColumn("VaR_95", F.expr("percentile_approx(return_pct, 0.05)").over(w))
          .withColumn("VaR_99", F.expr("percentile_approx(return_pct, 0.01)").over(w))
    )