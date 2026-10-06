"""Live viewer for span-switched decoding on held-out tool-calling turns.

Serves a web page: pick a held-out turn (the same items as
`scripts.eval_toolcall`), then watch the model decode it. AR tokens stream in
one by one; each diffusion block appears as masked slots that fill in with
every denoising pass and turn green when committed. Uses the HF reference
decoder (`src.span_decode`), i.e. exactly what the evaluation runs.

    CUDA_VISIBLE_DEVICES=3 uv run python -m scripts.span_viewer \
        --run e18b_entropy_spans --port 7860
    # from your laptop: ssh -L 7860:localhost:7860 <this machine>
    # then open http://localhost:7860

`--run` is a directory under runs/sft/ (its config.yaml and
train/adapter_final.pt are used) or `base` (no adapter, AR).
"""

import argparse
import json
import pathlib
import queue
import threading
import time

import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from transformers import AutoTokenizer

from scripts import eval_toolcall
from src import span_decode, span_model, spans

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "sft"


class Cancelled(Exception):
    """Raised inside the decoder when the browser goes away."""


class Viewer:
    """Model, tokenizer and held-out items for one run."""

    def __init__(self, run: str, max_new_tokens: int) -> None:
        """Loads the model (with the run's adapter) and the items."""
        self.run = run
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(
            span_model.DEFAULT_REPO, trust_remote_code=True
        )
        model = span_model.load_base(span_model.DEFAULT_REPO, "cuda")
        if run == "base":
            diff_spans, labels, self.block_size = False, None, 8
        else:
            cfg = yaml.safe_load((RUNS / run / "config.yaml").read_text())
            diff_spans = cfg["data"]["diff_spans"]
            labels = cfg["data"].get("span_labels")
            self.block_size = cfg["loss"]["block_size"]
            model = span_model.load_adapter(
                model, RUNS / run / "train" / "adapter_final.pt"
            )
        self.model = model.eval()
        self.items = {
            it["id"]: it
            for it in eval_toolcall.items(None, diff_spans, self.tok, labels)
        }
        self.lock = threading.Lock()

    def piece(self, token: int) -> str:
        """Display text of one token."""
        if token == spans.DIFF_OPEN:
            return "<diff>"
        if token == spans.DIFF_CLOSE:
            return "</diff>"
        return self.tok.decode([token], skip_special_tokens=False)

    def listing(self) -> list[dict]:
        """Short descriptions of the held-out turns."""
        out = []
        for it in self.items.values():
            text = self.tok.decode(it["prompt"][-400:], skip_special_tokens=False)
            last = text.rsplit("<|im_start|>", 2)[-2] if "<|im_start|>" in text else text
            calls, _ = eval_toolcall.parse_calls(it["reference"])
            out.append({
                "id": it["id"], "turn": it["turn"], "calls": len(calls),
                "preview": last.replace("<|im_end|>", "").strip()[:300],
            })  # fmt: skip
        return out

    def context(self, item_id: str) -> dict:
        """Last messages of the prompt and the reference turn."""
        it = self.items[item_id]
        tail = self.tok.decode(it["prompt"][-1500:], skip_special_tokens=False)
        return {"tail": tail, "reference": it["reference"]}

    def stream(self, item_id: str, mode: str, delay: float):
        """Yields server-sent events while decoding one turn."""
        it = self.items[item_id]
        events: queue.Queue = queue.Queue()
        state = {"cancel": False, "sleep": 0.0, "ids": [], "text": ""}

        def delta(new_ids: list[int]) -> str:
            # Decode the whole output so far and send only the new suffix, so
            # characters split across byte-level tokens render correctly.
            state["ids"] += new_ids
            text = self.tok.decode(state["ids"], skip_special_tokens=False)
            text = text.replace("<SPECIAL_18>", "<diff>").replace("<SPECIAL_19>", "</diff>")
            if "\ufffd" in text[len(state["text"]):]:
                return ""  # incomplete character: wait for the next token
            out, state["text"] = text[len(state["text"]):], text
            return out

        def on_event(ev: dict) -> None:
            if state["cancel"]:
                raise Cancelled
            if ev["type"] == "ar":
                ev["text"] = delta([ev["token"]])
            if ev["type"] == "commit":
                ev["committed"] = delta(ev["tokens"])
            if "tokens" in ev:
                ev["texts"] = [
                    None if t == ev.get("mask_id") else self.piece(t)
                    for t in ev["tokens"]
                ]
            events.put(ev)
            if delay:
                time.sleep(delay)
                state["sleep"] += delay

        def run() -> None:
            t0 = time.time()
            try:
                with self.lock:
                    res = span_decode.span_generate(
                        self.model,
                        torch.tensor([it["prompt"]], device="cuda"),
                        eos_token_id=self.tok.convert_tokens_to_ids("<|im_end|>"),
                        max_new_tokens=self.max_new_tokens,
                        block_size=self.block_size,
                        mode=mode,
                        on_event=on_event,
                    )
                    torch.cuda.synchronize()
                text = self.tok.decode(res.token_ids, skip_special_tokens=False)
                gen, ok = eval_toolcall.parse_calls(text)
                ref, _ = eval_toolcall.parse_calls(it["reference"])
                seconds = time.time() - t0 - state["sleep"]
                events.put({
                    "type": "done", "tokens": len(res.token_ids), "nfe": res.nfe,
                    "seconds": seconds, "exact": gen == ref, "parsed": ok,
                    "segments": " ".join(f"{s.mode}{s.num_tokens}" for s in res.segments),
                })  # fmt: skip
            except Cancelled:
                pass
            except Exception as e:  # surfaced in the page
                events.put({"type": "error", "message": repr(e)})
            finally:
                events.put(None)

        threading.Thread(target=run, daemon=True).start()
        try:
            while (ev := events.get()) is not None:
                yield f"data: {json.dumps(ev)}\n\n"
        finally:
            state["cancel"] = True


def make_app(viewer: Viewer) -> FastAPI:
    """The web app."""
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE.replace("__RUN__", viewer.run)

    @app.get("/items")
    def items() -> list[dict]:
        return viewer.listing()

    @app.get("/context")
    def context(id: str) -> dict:
        if id not in viewer.items:
            raise HTTPException(404)
        return viewer.context(id)

    @app.get("/stream")
    def stream(id: str, mode: str = "spans", delay_ms: int = 0):
        if id not in viewer.items or mode not in ("spans", "ar"):
            raise HTTPException(404)
        if viewer.lock.locked():
            raise HTTPException(409, "busy: another decode is running")
        return StreamingResponse(
            viewer.stream(id, mode, delay_ms / 1000),
            media_type="text/event-stream",
        )

    return app


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Span Decoding Viewer</title>
<style>
:root{--bg:#fafaf8;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e3e2dc;--ar:#1d1d1b;
--dlm:#0b7a53;--dlmbg:#e3f4ec;--live:#b45309;--livebg:#fdf1de;--mask:#d6d4cc;--sel:#eef0ff;--tag:#4338ca}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1d1d1b;--ink:#ecebe6;--muted:#9a998f;--line:#33322e;
--ar:#ecebe6;--dlm:#5fd3a2;--dlmbg:#123a2b;--live:#f5b056;--livebg:#3b2a10;--mask:#4a4943;--sel:#262a45;--tag:#a5b4fc}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header h1{font-size:16px;margin:0}header .run{color:var(--muted)}
main{display:grid;grid-template-columns:340px 1fr;height:calc(100vh - 53px)}
@media (max-width:800px){main{grid-template-columns:1fr;height:auto}#list{max-height:40vh}}
#side{border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0}
#side input{margin:10px;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink)}
#list{overflow:auto;flex:1}
.item{padding:8px 12px;border-bottom:1px solid var(--line);cursor:pointer}
.item:hover{background:var(--sel)}.item.on{background:var(--sel)}
.item .meta{font-size:12px;color:var(--muted)}.item .pv{font-size:12px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
#main{overflow:auto;padding:14px 16px;min-width:0}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
button,select{padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);cursor:pointer}
button.primary{background:var(--tag);color:#fff;border-color:var(--tag)}
.stats{display:flex;gap:16px;flex-wrap:wrap;font-variant-numeric:tabular-nums;margin:8px 0 12px;color:var(--muted)}
.stats b{color:var(--ink)}
details{margin:8px 0;border:1px solid var(--line);border-radius:8px;background:var(--panel)}
details summary{padding:6px 10px;cursor:pointer;color:var(--muted)}
pre{white-space:pre-wrap;word-break:break-word;margin:0;padding:10px;font:12.5px/1.5 ui-monospace,monospace}
#out{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;min-height:200px;
white-space:pre-wrap;word-break:break-word;font:13px/1.7 ui-monospace,monospace}
.blk{border-radius:4px;padding:1px 0;background:var(--livebg);outline:1px dashed var(--live)}
.blk.done{background:var(--dlmbg);outline:none;color:var(--dlm)}
.slot.m{display:inline-block;width:.9em;height:1em;margin:0 1px;vertical-align:-2px;border-radius:2px;background:var(--mask)}
.mark{color:var(--tag);font-weight:600}
.legend{display:flex;gap:14px;font-size:12px;color:var(--muted);flex-wrap:wrap}
.legend span i{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;vertical-align:-2px}
.result{margin-top:10px;font-weight:600}.ok{color:var(--dlm)}.bad{color:#dc2626}
</style></head><body>
<header><h1>Span decoding viewer</h1><span class="run">adapter: __RUN__</span>
<div class="legend"><span><i style="background:var(--ink)"></i>AR token</span>
<span><i style="background:var(--livebg);outline:1px dashed var(--live)"></i>diffusion block (denoising)</span>
<span><i style="background:var(--dlmbg)"></i>committed block</span><span><i style="background:var(--mask)"></i>mask</span></div></header>
<main><div id="side"><input id="q" placeholder="filter turns..."><div id="list"></div></div>
<div id="main">
<div class="controls">
<select id="mode"><option value="spans">spans (diffusion inside &lt;diff&gt;)</option><option value="ar">all AR</option></select>
<label>delay <select id="delay"><option value="0">0 ms</option><option value="30" selected>30 ms</option><option value="100">100 ms</option><option value="300">300 ms</option></select></label>
<button class="primary" id="go" disabled>Decode</button><button id="stop" disabled>Stop</button></div>
<div id="title" style="color:var(--muted)">Pick a turn on the left.</div>
<details id="ctx"><summary>Conversation so far (prompt tail)</summary><pre id="tail"></pre></details>
<div class="stats" id="stats"></div>
<div id="out"></div><div class="result" id="res"></div>
<details><summary>Reference turn</summary><pre id="ref"></pre></details>
</div></main>
<script>
const $=id=>document.getElementById(id);let items=[],cur=null,es=null;
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function fmtText(t){return esc(t).replace(/&lt;(\/?)diff&gt;/g,'<span class="mark">&lt;$1diff&gt;</span>')}
function fmt(t){return (t==='<diff>'||t==='</diff>')?`<span class="mark">${esc(t)}</span>`:esc(t)}
fetch('items').then(r=>r.json()).then(d=>{items=d;draw()});
$('q').oninput=draw;
function draw(){const q=$('q').value.toLowerCase();$('list').innerHTML=items.filter(i=>!q||i.preview.toLowerCase().includes(q)||i.id.includes(q))
.map(i=>`<div class="item${cur===i.id?' on':''}" data-id="${i.id}"><div class="meta">${i.id.slice(0,8)} · turn ${i.turn} · ${i.calls?i.calls+' call(s)':'answer'}</div><div class="pv">${esc(i.preview)}</div></div>`).join('');
document.querySelectorAll('.item').forEach(e=>e.onclick=()=>pick(e.dataset.id))}
function pick(id){cur=id;draw();$('go').disabled=false;const it=items.find(i=>i.id===id);
$('title').textContent=`${id} · turn ${it.turn} · reference ${it.calls?it.calls+' tool call(s)':'text answer'}`;
fetch('context?id='+id).then(r=>r.json()).then(c=>{$('tail').textContent=c.tail;$('ref').textContent=c.reference});
$('out').innerHTML='';$('res').textContent='';$('stats').innerHTML=''}
let S;function stats(){const tps=S.nfe?(S.tokens/S.nfe).toFixed(2):'-';const dl=S.tokens?(100*S.dlm/S.tokens).toFixed(1):'0';
$('stats').innerHTML=`<span>tokens <b>${S.tokens}</b></span><span>forward passes <b>${S.nfe}</b></span><span>tokens/pass <b>${tps}</b></span><span>diffusion share <b>${dl}%</b></span><span>blocks <b>${S.blocks}</b></span>`+(S.sec?`<span>compute <b>${S.sec.toFixed(1)}s</b> (${(S.tokens/S.sec).toFixed(1)} tok/s)</span>`:'')}
$('go').onclick=()=>{if(!cur)return;if(es)es.close();$('out').innerHTML='';$('res').textContent='';
S={tokens:0,nfe:1,dlm:0,blocks:0,sec:0};stats();let live=null;const out=$('out');
es=new EventSource(`stream?id=${cur}&mode=${$('mode').value}&delay_ms=${$('delay').value}`);$('go').disabled=true;$('stop').disabled=false;
es.onmessage=m=>{const e=JSON.parse(m.data);S.nfe=e.nfe||S.nfe;
if(e.type==='ar'){out.insertAdjacentHTML('beforeend',fmtText(e.text));S.tokens++}
else if(e.type==='block'){if(!live){live=document.createElement('span');live.className='blk';out.appendChild(live);S.blocks++}
live.innerHTML=e.texts.map(t=>t===null?'<span class="slot m"></span>':fmt(t)).join('')}
else if(e.type==='commit'){if(live){live.innerHTML=fmtText(e.committed);live.classList.add('done');live=null}S.tokens+=e.tokens.length;S.dlm+=e.tokens.length}
else if(e.type==='done'){S.tokens=e.tokens;S.nfe=e.nfe;S.sec=e.seconds;
$('res').innerHTML=e.exact?'<span class="ok">✓ tool calls match the reference</span>':'<span class="bad">✗ tool calls differ from the reference</span>';finish()}
else if(e.type==='error'){$('res').innerHTML=`<span class="bad">${esc(e.message)}</span>`;finish()}
stats();$('main').scrollTop=$('main').scrollHeight};
es.onerror=()=>{finish()}};
function finish(){if(es){es.close();es=null}$('go').disabled=false;$('stop').disabled=true}
$('stop').onclick=finish;
</script></body></html>"""


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="e18b_entropy_spans")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    viewer = Viewer(args.run, args.max_new_tokens)
    uvicorn.run(make_app(viewer), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
