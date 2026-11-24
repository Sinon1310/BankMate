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

Colours: the categorical order blue, amber (debit), teal (credit), violet
was validated on the dark panel surface #111A2C with the dataviz
validate_palette.js script (lightness band, chroma, colour-blind and
normal-vision separation, contrast: all checks pass).
"""

import altair as alt
import pandas as pd

from ui.data import format_inr
from ui.theme import TOKENS


FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
GRID = TOKENS["border"]

BAR_SIZE = 18
ROW_HEIGHT = 36

# Vega expression for rupee tick labels, e.g. 25000 -> ₹25,000; -500 -> −₹500
RUPEE_TICKS = "(datum.value < 0 ? '−' : '') + '₹' + format(abs(datum.value), ',')"
# Compact Indian units for large axes: 1,20,00,000 -> ₹1.2Cr, 4,50,000 -> ₹4.5L
RUPEE_COMPACT = (
    "(datum.value < 0 ? '−' : '') + '₹' + (abs(datum.value) >= 1e7 ? format(abs(datum.value) / 1e7, '.1~f') + 'Cr'"
    " : abs(datum.value) >= 1e5 ? format(abs(datum.value) / 1e5, '.1~f') + 'L'"
    " : abs(datum.value) >= 1e3 ? format(abs(datum.value) / 1e3, '.1~f') + 'K' : format(abs(datum.value), ','))"
)


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


# ---------- Stage UI: monthly, category, ranked and status charts ----------

def _month_label(month, with_year=True):
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    name = names[int(month[5:]) - 1]
    return f"{name} {month[2:4]}" if with_year else name


def _labeller(rows):
    """Month labels without the year when all months share one (short enough for narrow screens)."""
    single_year = len({r["month"][:4] for r in rows}) <= 1
    return lambda month: _month_label(month, with_year=not single_year)


def monthly_flow(rows, height=260):
    """
    Grouped monthly columns: credits and debits (two series, legend).
    rows: app.ledger.customer_monthly output.
    """

    data, label_for = [], _labeller(rows)
    for row in rows:
        f, label = row["figures"], label_for(row["month"])
        for kind, value in (("Credits", f["total_credits"]), ("Debits", f["total_debits"])):
            data.append({"month": label, "order": row["month"], "type": kind, "value": float(value),
                         "amount": format_inr(value), "net": format_inr(f["net_cash_flow"], signed=True)})
    frame = pd.DataFrame(data)
    order = list(dict.fromkeys(frame.sort_values("order")["month"]))

    bars = alt.Chart(frame).mark_bar(size=16, cornerRadiusEnd=4).encode(
        x=alt.X("month:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, grid=False)),
        xOffset=alt.XOffset("type:N", sort=["Credits", "Debits"]),
        y=alt.Y("value:Q", title=None, axis=alt.Axis(labelExpr=RUPEE_COMPACT, tickCount=5), scale=alt.Scale(nice=True)),
        color=alt.Color("type:N", scale=alt.Scale(domain=["Credits", "Debits"], range=[TOKENS["credit"], TOKENS["debit"]]),
                        legend=alt.Legend()),
        tooltip=[alt.Tooltip("month:N", title="Month"), alt.Tooltip("type:N", title="Type"),
                 alt.Tooltip("amount:N", title="Amount"), alt.Tooltip("net:N", title="Net transaction flow")],
    )
    return _style(bars, height=height)


def balance_trend(rows, height=220):
    """Month-end balance (single series, no legend). rows: customer_monthly output."""

    label_for = _labeller(rows)
    frame = pd.DataFrame([{"month": label_for(r["month"]), "order": r["month"],
                           "value": float(r["closing_balance"]), "amount": format_inr(r["closing_balance"])}
                          for r in rows])
    order = list(frame.sort_values("order")["month"])
    base = alt.Chart(frame).encode(
        x=alt.X("month:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, grid=False)),
        y=alt.Y("value:Q", title=None, axis=alt.Axis(labelExpr=RUPEE_COMPACT, tickCount=4),
                scale=alt.Scale(zero=False, nice=True)),
        tooltip=[alt.Tooltip("month:N", title="Month end"), alt.Tooltip("amount:N", title="Balance")],
    )
    line = base.mark_line(color=TOKENS["primary"], strokeWidth=2)
    points = base.mark_circle(size=64, color=TOKENS["primary"], stroke=TOKENS["surface"], strokeWidth=2, opacity=1)
    return _style(line + points, height=height)


def format_inr_compact(value):
    """₹1.27Cr / ₹48.4L / ₹9,000 (Indian units) for bar-end labels."""

    amount = float(value)
    sign = "−" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 1e7:
        return f"{sign}₹{amount / 1e7:.2f}Cr"
    if amount >= 1e5:
        return f"{sign}₹{amount / 1e5:.1f}L"
    return sign + format_inr(round(amount))


def ranked_bars(items, color=None, height_per_row=ROW_HEIGHT, label_limit=170, signed=False):
    """Horizontal ranked bars: [(label, Decimal, tooltip_title)] (single series).
    Bar-end labels use compact Indian units; tooltips give the exact amount."""

    frame = pd.DataFrame([{"label": label, "value": float(value), "short": format_inr_compact(value),
                           "amount": format_inr(value, signed=signed).replace("-", "−"), "title": title}
                          for label, value, title in items])
    order = list(frame["label"])
    base = alt.Chart(frame).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=label_limit)),
        x=alt.X("value:Q", title=None, axis=alt.Axis(labelExpr=RUPEE_COMPACT, tickCount=4),
                scale=alt.Scale(domain=_domain(frame["value"], headroom=0.35), nice=False)),
        tooltip=[alt.Tooltip("label:N", title="Item"), alt.Tooltip("amount:N", title="Amount")],
    )
    bars = base.mark_bar(size=BAR_SIZE, cornerRadiusEnd=4, color=color or TOKENS["primary"])
    positive = base.transform_filter("datum.value >= 0").mark_text(
        align="left", dx=6, fontSize=12, color=TOKENS["text_muted"]).encode(text="short:N")
    negative = base.transform_filter("datum.value < 0").mark_text(
        align="right", dx=-6, fontSize=12, color=TOKENS["text_muted"]).encode(text="short:N")
    return _style(bars + positive + negative, height=height_per_row * len(frame) + 12)


def count_bars(items, height_per_row=32):
    """Horizontal bars of counts: [(label, int)] (single series)."""

    frame = pd.DataFrame([{"label": label, "value": count} for label, count in items])
    order = list(frame["label"])
    base = alt.Chart(frame).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=170)),
        x=alt.X("value:Q", title=None, axis=alt.Axis(tickMinStep=1, format="d", tickCount=4),
                scale=alt.Scale(domain=[0, max(1, max(frame["value"])) * 1.3], nice=False)),
        tooltip=[alt.Tooltip("label:N", title="Status"), alt.Tooltip("value:Q", title="Count")],
    )
    bars = base.mark_bar(size=16, cornerRadiusEnd=4, color=TOKENS["primary"])
    labels = base.mark_text(align="left", dx=6, fontSize=12, color=TOKENS["text_muted"]).encode(text="value:Q")
    return _style(bars + labels, height=height_per_row * len(frame) + 12)
