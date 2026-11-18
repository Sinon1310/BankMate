"""
Altair charts in the BankMate design system.

Conventions (from the data-visualisation guidance):
- bars <= 24px thick with 4px rounded data-ends, growing from one baseline;
- hairline solid gridlines, no axis domain/ticks, muted axis text;
- labels and values use text colours, never the series colour;
- single-series charts have no legend (the panel title names them);
  two-series charts always have one;
- every mark has a tooltip; every chart's data is also in the
  transaction table on the same page (table view).

Credit/debit colours were validated as a categorical pair on the white
panel surface (worst colour-blind separation ΔE 9.0, all checks pass).
"""

import altair as alt
import pandas as pd

from ui.data import format_inr
from ui.theme import TOKENS


FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
GRID = "#EEEAE2"

BAR_SIZE = 18
ROW_HEIGHT = 36

# Vega expression for rupee tick labels, e.g. 25000 -> ₹25,000; -500 -> −₹500
RUPEE_TICKS = "(datum.value < 0 ? '−' : '') + '₹' + format(abs(datum.value), ',')"


def _domain(values, headroom=0.22):
    """Zero-anchored domain with room for value labels beyond the bar ends."""

    low, high = min(0.0, min(values)), max(0.0, max(values))
    span = (high - low) or 1.0
    return [low - (span * headroom if low < 0 else 0), high + (span * headroom if high > 0 else 0)]


def _style(chart, height):
    return (
        chart.properties(height=height, background="transparent", padding={"left": 0, "right": 8, "top": 8, "bottom": 0})
        .configure_view(stroke=None)
        .configure_axis(
            labelFont=FONT,
            titleFont=FONT,
            labelFontSize=12,
            titleFontSize=12,
            labelColor=TOKENS["text_muted"],
            titleColor=TOKENS["text_muted"],
            titleFontWeight=500,
            gridColor=GRID,
            gridWidth=1,
            domain=False,
            ticks=False,
            labelPadding=8,
        )
        .configure_legend(
            labelFont=FONT,
            labelFontSize=12,
            labelColor=TOKENS["text"],
            symbolType="square",
            symbolSize=110,
            orient="top",
            direction="horizontal",
            title=None,
            padding=0,
            offset=12,
        )
        .configure_text(font=FONT)
    )


def merchant_bars(items):
    """Ranked horizontal bars of debit totals: [(merchant, Decimal)]."""

    frame = pd.DataFrame(
        [{"merchant": m, "amount": float(v), "label": format_inr(v)} for m, v in items]
    )
    order = list(frame["merchant"])

    base = alt.Chart(frame).encode(
        y=alt.Y("merchant:N", sort=order, title=None, axis=alt.Axis(labelLimit=140)),
        x=alt.X(
            "amount:Q",
            title=None,
            axis=alt.Axis(labelExpr=RUPEE_TICKS, tickCount=4, grid=True),
            scale=alt.Scale(domain=_domain(frame["amount"]), nice=False),
        ),
        tooltip=[
            alt.Tooltip("merchant:N", title="Merchant"),
            alt.Tooltip("label:N", title="Debits"),
        ],
    )

    bars = base.mark_bar(size=BAR_SIZE, cornerRadiusEnd=4, color=TOKENS["debit"])
    labels = base.mark_text(align="left", dx=6, fontSize=12, color=TOKENS["text_muted"]).encode(
        text="label:N"
    )

    return _style(bars + labels, height=ROW_HEIGHT * len(frame) + 12)


def daily_flow_columns(items):
    """
    Diverging columns per day: credits above zero, debits below.
    items: [(date, credit Decimal, debit Decimal)].
    """

    rows = []

    for day, credit, debit in items:
        label = f"{day.day} {day.strftime('%b')}"
        if credit:
            rows.append({"day": label, "order": day.isoformat(), "type": "Credit",
                         "value": float(credit), "amount": format_inr(credit)})
        if debit:
            rows.append({"day": label, "order": day.isoformat(), "type": "Debit",
                         "value": -float(debit), "amount": format_inr(debit)})

    frame = pd.DataFrame(rows)
    order = list(dict.fromkeys(frame.sort_values("order")["day"]))

    bars = alt.Chart(frame).mark_bar(size=22, cornerRadiusEnd=4).encode(
        x=alt.X("day:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, grid=False)),
        y=alt.Y(
            "value:Q",
            title=None,
            axis=alt.Axis(labelExpr=RUPEE_TICKS, tickCount=5),
            scale=alt.Scale(nice=True),
        ),
        color=alt.Color(
            "type:N",
            scale=alt.Scale(domain=["Credit", "Debit"], range=[TOKENS["credit"], TOKENS["debit"]]),
            legend=alt.Legend(),
        ),
        tooltip=[
            alt.Tooltip("day:N", title="Date"),
            alt.Tooltip("type:N", title="Type"),
            alt.Tooltip("amount:N", title="Amount"),
        ],
    )

    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color=TOKENS["border_strong"], strokeWidth=1
    ).encode(y="y:Q")

    return _style(bars + zero, height=260)


def net_flow_bars(items):
    """Horizontal bars of net flow per customer: [(id, name, Decimal)]."""

    frame = pd.DataFrame([
        {
            "customer": f"{name} · {cid}",
            "value": float(net),
            "label": format_inr(net, signed=True).replace("-", "−"),
        }
        for cid, name, net in items
    ])
    order = list(frame["customer"])

    base = alt.Chart(frame).encode(
        y=alt.Y("customer:N", sort=order, title=None, axis=alt.Axis(labelLimit=180)),
        x=alt.X(
            "value:Q",
            title=None,
            axis=alt.Axis(labelExpr=RUPEE_TICKS, tickCount=4),
            scale=alt.Scale(domain=_domain(frame["value"], headroom=0.45), nice=False),
        ),
        tooltip=[
            alt.Tooltip("customer:N", title="Customer"),
            alt.Tooltip("label:N", title="Net transaction flow"),
        ],
    )

    bars = base.mark_bar(size=BAR_SIZE, cornerRadiusEnd=4, color=TOKENS["primary"])
    # Labels sit beyond the bar end on whichever side the bar grows.
    positive = base.transform_filter("datum.value >= 0").mark_text(
        align="left", dx=6, fontSize=12, color=TOKENS["text_muted"]
    ).encode(text="label:N")
    negative = base.transform_filter("datum.value < 0").mark_text(
        align="right", dx=-6, fontSize=12, color=TOKENS["text_muted"]
    ).encode(text="label:N")
    labels = positive + negative
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color=TOKENS["border_strong"], strokeWidth=1
    ).encode(x="x:Q")

    return _style(bars + labels + zero, height=ROW_HEIGHT * len(frame) + 12)
