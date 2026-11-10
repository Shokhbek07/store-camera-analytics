"""The live page served at / when the service runs with --http: one card per camera, refreshed every second
from /status.json. No external files or libraries, so it works offline."""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>storecam live</title>
<style>
:root { --bg: #f6f6f4; --card: #fff; --text: #1d1d1b; --muted: #6b6b66; --line: #e2e2dd;
        --ok: #2f7d4f; --alert: #b3261e; --alert-bg: #fbe9e7; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161615; --card: #20201e; --text: #ecece8; --muted: #9a9a93; --line: #34342f;
          --ok: #6fcf97; --alert: #ff8a80; --alert-bg: #3a1d1b; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px; background: var(--bg); color: var(--text);
       font: 15px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
header { max-width: 960px; margin: 0 auto 16px; display: flex; justify-content: space-between;
         align-items: baseline; gap: 12px; flex-wrap: wrap; }
h1 { font-size: 18px; margin: 0; }
#state { color: var(--muted); font-size: 13px; }
#cams { max-width: 960px; margin: 0 auto; display: grid; gap: 16px;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }
.cam { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }
.cam h2 { font-size: 15px; margin: 0 0 12px; display: flex; align-items: center; gap: 8px; }
.dot { width: 9px; height: 9px; border-radius: 50%; background: var(--alert); }
.dot.on { background: var(--ok); }
.nums { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.num b { display: block; font-size: 28px; font-variant-numeric: tabular-nums; }
.num span, .meta { color: var(--muted); font-size: 13px; }
.meta { margin-top: 10px; }
.alert { margin-top: 10px; padding: 8px 10px; border-radius: 6px; background: var(--alert-bg);
         color: var(--alert); font-weight: 600; font-size: 14px; }
</style>
</head>
<body>
<header><h1>storecam live</h1><div id="state">connecting...</div></header>
<main id="cams"></main>
<script>
const NAMES = { queue: "Queue alert", empty_counter: "Empty counter, customers waiting" };
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function num(value, label) {
  const d = el("div", "num");
  d.append(el("b", null, value), el("span", null, label));
  return d;
}
function card(name, s) {
  const c = el("section", "cam");
  const h = el("h2");
  h.append(el("span", "dot" + (s.connected ? " on" : "")), el("span", null, name));
  const nums = el("div", "nums");
  nums.append(num(s.in, "in"), num(s.out, "out"), num(s.queue, "in queue"));
  const staff = s.staff_present === null ? "no staff zone"
    : s.staff_present ? "staff at counter" : "no staff at counter";
  const meta = el("div", "meta", `${staff} · ${s.frames_processed} frames · ${s.reconnects} reconnects`);
  c.append(h, nums, meta);
  for (const a of s.open_alerts) {
    const open = s.second === null ? "" : `, open ${Math.max(0, Math.round(s.second - a.start))} s`;
    c.append(el("div", "alert", `${NAMES[a.kind] || a.kind}: peak ${a.peak}${open}`));
  }
  return c;
}
async function tick() {
  try {
    const r = await fetch("status.json", { cache: "no-store" });
    const data = await r.json();
    document.getElementById("cams").replaceChildren(...Object.entries(data).map(([n, s]) => card(n, s)));
    document.getElementById("state").textContent = "updated " + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById("state").textContent = "service not reachable, retrying";
  }
}
tick();
setInterval(tick, 1000);
</script>
</body>
</html>
"""
