import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import json
    import os
    import altair as alt
    import pandas as pd
    from datetime import datetime

    IDLE_THRESHOLD = 30 * 60  # gaps > 30 min = idle
    PROJECTS_DIR = os.path.expanduser("~/.claude/projects/")
    return IDLE_THRESHOLD, PROJECTS_DIR, alt, datetime, json, mo, os, pd


@app.cell
def _(PROJECTS_DIR, os):
    _home_escaped = os.path.expanduser("~").lstrip("/").replace("/", "-").replace(".", "-")
    _home_prefix = f"-{_home_escaped}-"

    project_paths = {}
    for proj_name in sorted(os.listdir(PROJECTS_DIR)):
        _p = os.path.join(PROJECTS_DIR, proj_name)
        if not os.path.isdir(_p):
            continue
        if not any(f.endswith(".jsonl") for f in os.listdir(_p)):
            continue
        label = proj_name.replace(_home_prefix, "~/", 1)
        project_paths[label] = _p

    project_labels = list(project_paths.keys())
    return project_labels, project_paths


@app.cell
def _(mo):
    days_picker = mo.ui.dropdown(
        options={"7 days": 7, "14 days": 14, "30 days": 30},
        value="14 days",
        label="Window",
    )
    return (days_picker,)


@app.cell
def _(
    IDLE_THRESHOLD,
    alt,
    datetime,
    days_picker,
    json,
    mo,
    os,
    pd,
    project_paths,
):
    from datetime import timezone as _tz, timedelta as _td

    def _parse(ts):
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))

    def _scan_all(project_paths):
        rows = []
        for proj_label, proj_path in project_paths.items():
            for fname in os.listdir(proj_path):
                if not fname.endswith(".jsonl"):
                    continue
                fpath = os.path.join(proj_path, fname)
                first_ts = last_ts = None
                turns = 0
                active_s = 0
                claude_s = 0
                out_tokens = 0
                prev_type = prev_ts = None
                with open(fpath) as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                        except:
                            continue
                        t = obj.get("type")
                        if t not in ("user", "assistant"):
                            continue
                        ts = obj.get("timestamp")
                        if not ts:
                            continue
                        if first_ts is None:
                            first_ts = ts
                        last_ts = ts
                        if t == "assistant":
                            _usage = obj.get("message", {}).get("usage", {})
                            out_tokens += _usage.get("output_tokens", 0)
                        if prev_ts:
                            delta = (_parse(ts) - _parse(prev_ts)).total_seconds()
                            if delta <= IDLE_THRESHOLD:
                                if prev_type == "user" and t == "assistant":
                                    turns += 1
                                    claude_s += delta
                                active_s += delta
                        prev_type, prev_ts = t, ts
                if turns == 0 or first_ts is None:
                    continue
                start = _parse(first_ts).astimezone()
                rows.append({
                    "project": proj_label,
                    "date": start.date(),
                    "start": start,
                    "turns": turns,
                    "duration_min": active_s / 60,
                    "output_density": round(out_tokens / (claude_s / 60)) if claude_s > 0 else 0,
                })
        return pd.DataFrame(rows)

    _all = _scan_all(project_paths)
    _now = datetime.now(_tz.utc)
    _days = days_picker.value
    _nd = _all[_all["start"] >= _now - _td(days=_days)] if not _all.empty else _all

    # ── stat cards ────────────────────────────────────────────────────────────
    _n_proj = len(project_paths)
    _n_sessions = len(_all)
    _n_nd = len(_nd)
    _turns_nd = int(_nd["turns"].sum()) if not _nd.empty else 0
    _time_nd = _nd["duration_min"].sum() if not _nd.empty else 0

    _cards = mo.hstack([
        mo.stat(label="Projects", value=str(_n_proj)),
        mo.stat(label="Total sessions", value=str(_n_sessions)),
        mo.stat(label=f"Sessions ({_days}d)", value=str(_n_nd)),
        mo.stat(label=f"Turns ({_days}d)", value=str(_turns_nd)),
        mo.stat(label=f"Active time ({_days}d)", value=f"{_time_nd:.0f} min"),
    ], justify="start")

    # ── recent sessions bar chart ─────────────────────────────────────────────
    if not _nd.empty:
        _nd_plot = _nd.copy()
        _nd_plot["date_str"] = _nd_plot["date"].apply(lambda d: d.strftime("%b %d"))
        _nd_plot["date_iso"] = _nd_plot["date"].apply(lambda d: d.isoformat())
        _nd_plot["project_short"] = _nd_plot["project"].str.split("/").str[-1]
        _date_sort = alt.EncodingSortField(field="date_iso", op="min", order="ascending")

        _bar = alt.Chart(_nd_plot).mark_bar().encode(
            x=alt.X("date_str:O", title="Date", sort=_date_sort),
            y=alt.Y("sum(turns):Q", title="Total turns"),
            color=alt.Color("project_short:N", legend=alt.Legend(title="Project")),
            tooltip=[
                alt.Tooltip("date_str:O", title="Date"),
                alt.Tooltip("project_short:N", title="Project"),
                alt.Tooltip("turns:Q", title="Turns"),
                alt.Tooltip("duration_min:Q", title="Duration (min)", format=".0f"),
            ],
        ).properties(title=f"Turns per day — last {_days} days", height=180, width=860)

        _time_bar = alt.Chart(_nd_plot).mark_bar().encode(
            x=alt.X("date_str:O", title="Date", sort=_date_sort),
            y=alt.Y("sum(duration_min):Q", title="Active time (min)"),
            color=alt.Color("project_short:N", legend=None),
            tooltip=[
                alt.Tooltip("date_str:O", title="Date"),
                alt.Tooltip("project_short:N", title="Project"),
                alt.Tooltip("sum(duration_min):Q", title="Active time (min)", format=".0f"),
                alt.Tooltip("turns:Q", title="Turns"),
            ],
        ).properties(title="Active time per day", height=180, width=420)

        _density_bar = alt.Chart(_nd_plot).mark_circle(size=80, opacity=0.7).encode(
            x=alt.X("date_str:O", title="Date", sort=_date_sort),
            y=alt.Y("output_density:Q", title="Tokens / Claude-min"),
            color=alt.Color("project_short:N", legend=None),
            tooltip=[
                alt.Tooltip("date_str:O", title="Date"),
                alt.Tooltip("project_short:N", title="Project"),
                alt.Tooltip("output_density:Q", title="Output density (tok/min)", format=".0f"),
                alt.Tooltip("turns:Q", title="Turns"),
                alt.Tooltip("duration_min:Q", title="Duration (min)", format=".0f"),
            ],
        ).properties(title="Output density per session (tokens/min of Claude time)", height=180, width=420)

        _macro_chart = mo.ui.altair_chart(
            alt.vconcat(_bar, alt.hconcat(_time_bar, _density_bar))
            .configure_view(strokeWidth=0)
            .configure_axis(labelFont="Inter, sans-serif", titleFont="Inter, sans-serif")
        )
    else:
        _macro_chart = mo.md(f"_No sessions in the last {_days} days._")

    mo.vstack([
        mo.md("## Claude Session Overview"),
        days_picker,
        _cards,
        _macro_chart,
        mo.md("---"),
    ])
    return


@app.cell
def _(mo, project_labels):

    project_picker = mo.ui.dropdown(
        options=project_labels,
        label="Project",
        value=project_labels[0],
    )
    mo.vstack([
        mo.md("## Claude Project Sessions"),
        project_picker
    ])
    return (project_picker,)


@app.cell
def _(IDLE_THRESHOLD, datetime, json, mo, os, project_paths, project_picker):
    def parse_ts(ts):
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))

    def _topic_from(text):
        skip = ("<ide_opened_file>", "<ide_selection>")
        return not any(s in text for s in skip) and text.strip()

    def scan_sessions(proj_path):
        """Fast pass: read only enough of each file to get metadata."""
        sessions = []
        for fname in os.listdir(proj_path):
            if not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(proj_path, fname)
            first_ts = last_ts = topic = None
            turns = user_prev = 0
            prev_type = prev_ts = None

            with open(fpath) as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except:
                        continue
                    t = obj.get("type")
                    if t not in ("user", "assistant"):
                        continue
                    ts = obj.get("timestamp")
                    if not ts:
                        continue
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts

                    # topic: first real user text
                    if not topic and t == "user":
                        for c in obj.get("message", {}).get("content", []) or []:
                            if isinstance(c, dict) and c.get("type") == "text":
                                txt = c.get("text", "")
                                if _topic_from(txt):
                                    topic = txt[:80]
                                    break

                    # count turns (no idle filtering here — fast enough)
                    if prev_type == "user" and t == "assistant":
                        if prev_ts:
                            delta = (parse_ts(ts) - parse_ts(prev_ts)).total_seconds()
                            if delta <= IDLE_THRESHOLD:
                                turns += 1
                    prev_type, prev_ts = t, ts

            if turns == 0 or first_ts is None:
                continue

            start = parse_ts(first_ts).astimezone()
            sessions.append({
                "fname": fname,
                "fpath": fpath,
                "start": start,
                "date": start.strftime("%Y-%m-%d %H:%M"),
                "turns": turns,
                "topic": topic or "(no topic)",
            })

        sessions.sort(key=lambda s: s["start"])
        return sessions

    def load_messages(fpath):
        """Full load for one selected session."""
        msgs = []
        with open(fpath) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("type") in ("user", "assistant"):
                        msgs.append(obj)
                except:
                    pass
        msgs.sort(key=lambda m: m["timestamp"])
        return msgs

    proj_path = project_paths.get(project_picker.value, "")
    all_sessions = scan_sessions(proj_path) if proj_path else []

    import pandas as _pd
    _tbl = _pd.DataFrame([
        {"Date": s["date"], "Turns": s["turns"], "Topic": s["topic"], "File": s["fname"]}
        for s in all_sessions
    ]) if all_sessions else _pd.DataFrame()
    session_table = mo.ui.table(_tbl, selection="single") if not _tbl.empty else None
    session_table if session_table is not None else mo.md("_No sessions._")
    return all_sessions, load_messages, parse_ts, session_table


@app.cell
def _(all_sessions, mo, session_table):
    if not all_sessions:
        mo.stop(True, mo.md("No sessions found for this project."))

    # Derive selected session from table row click; default to most recent
    _val = session_table.value if session_table is not None else []
    if len(_val) > 0:
        _date = _val.iloc[0]["Date"]
        selected = next((s for s in all_sessions if s["date"] == _date), all_sessions[-1])
    else:
        selected = all_sessions[-1]
    return (selected,)


@app.cell
def _(IDLE_THRESHOLD, load_messages, parse_ts, pd, selected):
    rows = []
    if selected:
        messages = load_messages(selected["fpath"])
        t0 = parse_ts(messages[0]["timestamp"])

        for i in range(len(messages) - 1):
            curr, nxt = messages[i], messages[i + 1]
            t_start = parse_ts(curr["timestamp"])
            t_end = parse_ts(nxt["timestamp"])
            delta = (t_end - t_start).total_seconds()

            elapsed_start = (t_start - t0).total_seconds() / 60
            elapsed_end = (t_end - t0).total_seconds() / 60

            if delta > IDLE_THRESHOLD:
                kind = "idle"
            elif curr["type"] == "user" and nxt["type"] == "assistant":
                kind = "claude"
            elif curr["type"] == "assistant" and nxt["type"] == "user":
                # tool_result blocks mean Claude called a tool; gap is tool execution time.
                # Cap at 5 min — longer gaps mean user walked away mid-tool-call.
                _nxt_content = nxt.get("message", {}).get("content", []) or []
                _is_tool = any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in _nxt_content
                )
                if _is_tool and delta > 5 * 60:
                    kind = "idle"
                elif _is_tool:
                    kind = "tool"
                else:
                    kind = "you"
            else:
                kind = "other"

            _out = _inp = _cache = _total_chars = _code_chars = 0
            if curr["type"] == "assistant":
                _usage = curr.get("message", {}).get("usage", {})
                _out   = _usage.get("output_tokens", 0)
                _inp   = _usage.get("input_tokens", 0)
                _cache = (_usage.get("cache_read_input_tokens", 0)
                          + _usage.get("cache_creation_input_tokens", 0))
                # scan content for code fences
                for _blk in curr.get("message", {}).get("content", []) or []:
                    if isinstance(_blk, dict) and _blk.get("type") == "text":
                        _txt = _blk.get("text", "")
                        _total_chars += len(_txt)
                        import re as _re
                        for _m in _re.finditer(r"```.*?```", _txt, _re.DOTALL):
                            _code_chars += len(_m.group())

            rows.append({
                "turn": i,
                "kind": kind,
                "start_min": elapsed_start,
                "end_min": elapsed_end,
                "duration_s": delta,
                "tokens": _out,
                "input_tokens": _inp,
                "cache_tokens": _cache,
                "total_chars": _total_chars,
                "code_chars": _code_chars,
            })

    df = pd.DataFrame(rows)
    df
    return (df,)


@app.cell
def _(alt, df, mo, pd, selected):
    if selected is None or df.empty:
        mo.stop(True, mo.md("No data."))

    _guide = mo.md("""
    ###How to read these charts:
    - **Timeline** — each bar is one turn. 🟣 Purple = Claude generating; 🟢 Green = tool executing; 🟡 Amber = you reading/thinking/typing. Hover for exact seconds.
    - **Distribution** — how long each side's turns typically take. A tight cluster = consistent pace; long tail = occasional slow turns.
    - **Cumulative** — running total time per side as the session goes on. The gap between lines shows the imbalance.
    """)
    # ── 1. Timeline swimlane ──────────────────────────────────────────────────
    color_scale = alt.Scale(
        domain=["claude", "tool", "you", "idle"],
        range=["#6366f1", "#10b981", "#f59e0b", "#e5e7eb"],
    )

    active_df = df[df["kind"] != "idle"].copy()

    # Compress x-axis: skip idle gaps, accumulate only active duration
    _cursor = 0.0
    _starts, _ends = [], []
    for _, _row in active_df.iterrows():
        _starts.append(_cursor)
        _cursor += _row["duration_s"] / 60
        _ends.append(_cursor)
    active_df = active_df.copy()
    active_df["cx_start"] = _starts
    active_df["cx_end"]   = _ends

    swimlane = alt.Chart(active_df).mark_bar(
        height=40, cornerRadiusTopRight=3, cornerRadiusBottomRight=3
    ).encode(
        x=alt.X("cx_start:Q", title="Active minutes (idle gaps removed)"),
        x2="cx_end:Q",
        y=alt.Y("kind:N", title=None, sort=["claude", "tool", "you"],
                axis=alt.Axis(labelFontSize=13)),
        color=alt.Color("kind:N", scale=color_scale, legend=None),
        tooltip=[
            alt.Tooltip("kind:N", title="Who"),
            alt.Tooltip("duration_s:Q", title="Duration (s)", format=".1f"),
            alt.Tooltip("start_min:Q", title="Wall-clock start (min)", format=".1f"),
        ],
    ).properties(
        title=f"Turn-by-turn timeline  —  {selected['date']}  (compressed, idle removed)",
        height=140,
        width=900,
    )

    # ── 2. Duration distribution ──────────────────────────────────────────────
    hist = alt.Chart(active_df).mark_bar(opacity=0.8).encode(
        x=alt.X("duration_s:Q", bin=alt.Bin(maxbins=30), title="Duration (s)"),
        y=alt.Y("count():Q", title="# turns"),
        color=alt.Color("kind:N", scale=color_scale),
        tooltip=["kind:N", "count():Q"],
    ).properties(
        title="Response time distribution",
        height=200,
        width=430,
    )

    # ── 3. Cumulative time area chart ─────────────────────────────────────────
    cumdf_claude = active_df[active_df["kind"] == "claude"][["cx_end", "duration_s"]].copy()
    cumdf_tool   = active_df[active_df["kind"] == "tool"][["cx_end", "duration_s"]].copy()
    cumdf_you    = active_df[active_df["kind"] == "you"][["cx_end", "duration_s"]].copy()
    cumdf_claude["cum_min"] = cumdf_claude["duration_s"].cumsum() / 60
    cumdf_tool["cum_min"]   = cumdf_tool["duration_s"].cumsum() / 60
    cumdf_you["cum_min"]    = cumdf_you["duration_s"].cumsum() / 60
    cumdf_claude["kind"] = "claude"
    cumdf_tool["kind"]   = "tool"
    cumdf_you["kind"]    = "you"
    cum_combined = pd.concat([cumdf_claude, cumdf_tool, cumdf_you])

    cum_chart = alt.Chart(cum_combined).mark_line(strokeWidth=2).encode(
        x=alt.X("cx_end:Q", title="Active minutes (idle removed)"),
        y=alt.Y("cum_min:Q", title="Cumulative minutes"),
        color=alt.Color("kind:N", scale=color_scale, legend=alt.Legend(title="Who")),
        tooltip=[
            alt.Tooltip("kind:N", title="Who"),
            alt.Tooltip("cx_end:Q", title="At active-min", format=".1f"),
            alt.Tooltip("cum_min:Q", title="Cumulative (min)", format=".1f"),
        ],
    ).properties(
        title="Cumulative time spent",
        height=200,
        width=430,
    )

    chart = alt.vconcat(
        swimlane,
        alt.hconcat(hist, cum_chart),
    ).configure_view(strokeWidth=0).configure_axis(
        labelFont="Inter, sans-serif",
        titleFont="Inter, sans-serif",
    )

    mo.vstack([_guide, mo.ui.altair_chart(chart)])
    return


@app.cell
def _(IDLE_THRESHOLD, df, mo, selected):
    if selected and not df.empty:
        _active = df[df["kind"] != "idle"]
        _claude_min = _active[_active["kind"] == "claude"]["duration_s"].sum() / 60
        _tool_min   = _active[_active["kind"] == "tool"]["duration_s"].sum() / 60
        _you_min    = _active[_active["kind"] == "you"]["duration_s"].sum() / 60
        _idle_min   = df[df["kind"] == "idle"]["duration_s"].sum() / 60
        _total      = _claude_min + _tool_min + _you_min

        # tokens
        _out_tok   = int(df["tokens"].sum())
        _inp_tok   = int(df["input_tokens"].sum())
        _cache_tok = int(df["cache_tokens"].sum())
        _cache_pct = round(_cache_tok / _inp_tok * 100) if _inp_tok else 0

        # output density: output tokens per minute of claude time
        _density = round(_out_tok / _claude_min) if _claude_min > 0 else 0

        # session type from code fence ratio
        _tot_chars  = df["total_chars"].sum()
        _code_chars = df["code_chars"].sum()
        _code_ratio = _code_chars / _tot_chars if _tot_chars > 0 else 0
        if _code_ratio > 0.35:
            _stype = "💻 code-heavy"
        elif _code_ratio > 0.1:
            _stype = "🔀 mixed"
        else:
            _stype = "💬 thinking/prose"

        _ct = _active[_active["kind"] == "claude"]
        _avg_lat = _ct["duration_s"].mean() if not _ct.empty else 0

        _summary = mo.md(f"""
    ### Session summary

    **Session type:** {_stype} &nbsp;|&nbsp; {_code_ratio*100:.0f}% of output was in code blocks

    | Time | |
    |--|--|
    | Active time | {_total:.1f} min |
    | Idle time (gaps >{IDLE_THRESHOLD//60}min) | {_idle_min:.1f} min |
    | Claude generating | {_claude_min:.1f} min ({_claude_min/_total*100:.0f}%) |
    | Tool execution | {_tool_min:.1f} min ({_tool_min/_total*100:.0f}%) |
    | You responding | {_you_min:.1f} min ({_you_min/_total*100:.0f}%) |
    | Turns | {selected['turns']} |
    | Claude avg latency | {_avg_lat:.1f}s |

    | Tokens | |
    |--|--|
    | Output tokens | {_out_tok:,} |
    | Input tokens | {_inp_tok:,} |
    | Cached input | {_cache_tok:,} ({_cache_pct}% of input) |
    | Input/output ratio | {round(_inp_tok/_out_tok) if _out_tok else "—"}:1 |
    | **Output density** | **{_density} tokens/min** of Claude time |

    > **Output density** — how much Claude produced per minute it was active.
    > Compare sessions to each other: higher = more generative, lower = more conversational.
        """)
    else:
        _summary = mo.md("_No session selected._")

    _summary
    return


if __name__ == "__main__":
    app.run()
