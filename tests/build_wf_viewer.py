#!/usr/bin/env python3
"""Build a standalone, self-contained pixel-waveform VIEWER (one HTML file) from a
ti_waveforms.npz produced by threshold_induction_study.py (--wf-events > 0).

  python tests/build_wf_viewer.py ti_waveforms.npz -o wf_viewer.html

Runs anywhere (numpy only, no GPU). The viewer embeds the waveform records as JSON and
draws, per selected pixel, the induced CURRENT vs time and the running CHARGE vs time with
the discriminator threshold, the recorded hit time(s), and the truth charge-arrival time --
so you can see exactly why each pixel did or didn't fire. Pick pixels from a per-event map
coloured by hit category (single collection / single induction / multi / charged-no-hit)."""
import argparse, base64, json, os
import numpy as np


def load_records(path, max_pts=200, max_recs=600):
    d = np.load(path, allow_pickle=True)
    n = len(d["pix_id"])
    recs = []
    for i in range(min(n, max_recs)):
        t = np.asarray(d["t"][i], float)
        cur = np.asarray(d["cur"][i], float)
        chg = np.asarray(d["chg"][i], float)
        if t.size > max_pts:                              # even-stride downsample for display
            k = np.linspace(0, t.size - 1, max_pts).round().astype(int)
            t, cur, chg = t[k], cur[k], chg[k]
        r3 = lambda a: [round(float(v), 3) for v in a]
        recs.append(dict(
            sample=str(d["sample"][i]), ev=int(d["ev"][i]), id=int(d["pix_id"][i]),
            x=round(float(d["x"][i]), 3), y=round(float(d["y"][i]), 3), cat=str(d["cat"][i]),
            q_coll=round(float(d["q_coll"][i]), 1), t_coll=(None if not np.isfinite(d["t_coll"][i])
                                                            else round(float(d["t_coll"][i]), 3)),
            sig_t=(None if not np.isfinite(d["sig_t"][i]) else round(float(d["sig_t"][i]), 3)),
            n_hits=int(d["n_hits"][i]), thr=round(float(d["thr"][i]), 0),
            hit_t=r3(np.atleast_1d(d["hit_t"][i])), hit_q=[round(float(v), 0)
                                                           for v in np.atleast_1d(d["hit_q"][i])],
            t=r3(t), cur=r3(cur), chg=r3(chg)))
    return recs


CSS = r"""
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#EEF1F6;--panel:#FCFDFE;--ink:#151A22;--muted:#59636F;--faint:#8A94A1;--line:#E1E6ED;
  --grid:#E8ECF2;--accent:#2F6BB0;--coll:#C24A4A;--ind:#3B6FB6;--multi:#7E6CAD;--charged:#E1A730;--quiet:#9AA6B4;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:"SF Mono","JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0B0F14;--panel:#141A22;--ink:#E9EDF3;--muted:#9BA6B3;--faint:#6C7784;--line:#26303B;
  --grid:#222B36;--accent:#6BA3E0;--coll:#E0776E;--ind:#6BA3E0;--multi:#A99BD6;--charged:#E7BE63;--quiet:#6C7784;}}
:root[data-theme="light"]{--bg:#EEF1F6;--panel:#FCFDFE;--ink:#151A22;--muted:#59636F;--faint:#8A94A1;--line:#E1E6ED;--grid:#E8ECF2;--accent:#2F6BB0;--coll:#C24A4A;--ind:#3B6FB6;--multi:#7E6CAD;--charged:#E1A730;--quiet:#9AA6B4}
:root[data-theme="dark"]{--bg:#0B0F14;--panel:#141A22;--ink:#E9EDF3;--muted:#9BA6B3;--faint:#6C7784;--line:#26303B;--grid:#222B36;--accent:#6BA3E0;--coll:#E0776E;--ind:#6BA3E0;--multi:#A99BD6;--charged:#E7BE63;--quiet:#6C7784}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5;padding:22px;-webkit-font-smoothing:antialiased}
h1{font-family:var(--serif);font-weight:600;font-size:24px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13.5px;margin:4px 0 18px;max-width:80ch}
.bar{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin-bottom:16px;
  font-family:var(--mono);font-size:12px}
.bar label{color:var(--faint);text-transform:uppercase;letter-spacing:.08em;margin-right:6px}
select{font-family:var(--mono);font-size:13px;background:var(--panel);color:var(--ink);
  border:1px solid var(--line);border-radius:7px;padding:5px 9px}
.wrap{display:grid;grid-template-columns:minmax(300px,0.9fr) 1.4fr;gap:18px;align-items:start}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.card h2{font-family:var(--mono);font-size:11px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--faint);font-weight:600;margin-bottom:12px}
#map{width:100%;height:auto;display:block;touch-action:none}
#map circle{cursor:pointer}
.legend{display:flex;flex-wrap:wrap;gap:10px 16px;margin-top:12px;font-family:var(--mono);font-size:11px;color:var(--muted)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.meta{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;font-family:var(--mono);font-size:12.5px;margin-bottom:14px}
.meta .k{color:var(--faint)}.meta .v{color:var(--ink);font-variant-numeric:tabular-nums}
.tag{display:inline-block;padding:1px 8px;border-radius:20px;font-size:11px;font-weight:600;color:#fff}
canvas{width:100%;display:block;border-radius:6px}
.cap{font-family:var(--mono);font-size:11px;color:var(--faint);margin:4px 0 14px}
.hint{color:var(--faint);font-family:var(--mono);font-size:11.5px;margin-top:8px}
"""

JS = r"""
const R = __DATA__;
const CATNAME = {coll1:"single collection",ind1:"single induction",multi:"multi-hit",
  charged_nohit:"charge, no hit",quiet:"quiet"};
const CATVAR = {coll1:"--coll",ind1:"--ind",multi:"--multi",charged_nohit:"--charged",quiet:"--quiet"};
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let sample=null, ev=null, sel=null;

const bySample = {};
R.forEach(r=>{(bySample[r.sample]=bySample[r.sample]||{});(bySample[r.sample][r.ev]=bySample[r.sample][r.ev]||[]).push(r);});
const samples = Object.keys(bySample);

const selS=document.getElementById("selSample"), selE=document.getElementById("selEvent");
samples.forEach(s=>selS.add(new Option(s,s)));
function fillEvents(){selE.innerHTML="";Object.keys(bySample[sample]).sort((a,b)=>a-b).forEach(e=>selE.add(new Option("event "+e,e)));}
selS.onchange=()=>{sample=selS.value;fillEvents();ev=selE.value;sel=null;draw();};
selE.onchange=()=>{ev=selE.value;sel=null;draw();};

function pixels(){return (bySample[sample]||{})[ev]||[];}

function drawMap(){
  const P=pixels(); const NS="http://www.w3.org/2000/svg"; const map=document.getElementById("map");
  map.innerHTML="";
  const W=460,H=360,pad=34;
  map.setAttribute("viewBox",`0 0 ${W} ${H}`);
  const xs=P.map(p=>p.x),ys=P.map(p=>p.y);
  let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  if(x1-x0<1){x0-=1;x1+=1;} if(y1-y0<1){y0-=1;y1+=1;}
  const sx=v=>pad+(v-x0)/(x1-x0)*(W-2*pad), sy=v=>H-pad-(v-y0)/(y1-y0)*(H-2*pad);
  // axes
  const ax=(x1n,y1n,x2n,y2n)=>{const l=document.createElementNS(NS,"line");
    l.setAttribute("x1",x1n);l.setAttribute("y1",y1n);l.setAttribute("x2",x2n);l.setAttribute("y2",y2n);
    l.setAttribute("stroke",cssv("--line"));l.setAttribute("stroke-width","1");map.appendChild(l);};
  ax(pad,H-pad,W-pad,H-pad); ax(pad,pad,pad,H-pad);
  const lab=(t,x,y,anc)=>{const e=document.createElementNS(NS,"text");e.textContent=t;
    e.setAttribute("x",x);e.setAttribute("y",y);e.setAttribute("fill",cssv("--faint"));
    e.setAttribute("font-family",cssv("--mono"));e.setAttribute("font-size","10");e.setAttribute("text-anchor",anc);map.appendChild(e);};
  lab("pixel x (cm)",W/2,H-8,"middle");
  const yl=document.createElementNS(NS,"text");yl.textContent="pixel y (cm)";
  yl.setAttribute("transform",`translate(11,${H/2}) rotate(-90)`);yl.setAttribute("fill",cssv("--faint"));
  yl.setAttribute("font-family",cssv("--mono"));yl.setAttribute("font-size","10");yl.setAttribute("text-anchor","middle");map.appendChild(yl);
  P.forEach(p=>{const c=document.createElementNS(NS,"circle");
    c.setAttribute("cx",sx(p.x));c.setAttribute("cy",sy(p.y));
    c.setAttribute("r",(sel&&sel.id===p.id)?7:4.5);
    c.setAttribute("fill",cssv(CATVAR[p.cat]||"--quiet"));
    c.setAttribute("stroke",(sel&&sel.id===p.id)?cssv("--ink"):"none");c.setAttribute("stroke-width","1.5");
    c.setAttribute("opacity",(sel&&sel.id!==p.id)?0.55:0.95);
    c.onclick=()=>{sel=p;draw();};
    const ti=document.createElementNS(NS,"title");ti.textContent=`pix ${p.id} · ${CATNAME[p.cat]}`;c.appendChild(ti);
    map.appendChild(c);});
}

function trace(cv, xs, ys, opts){
  const dpr=window.devicePixelRatio||1; const cssw=cv.clientWidth, cssh=opts.h;
  cv.width=cssw*dpr; cv.height=cssh*dpr; cv.style.height=cssh+"px";
  const g=cv.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,cssw,cssh);
  const pl=48,pr=14,pt=14,pb=30, W=cssw-pl-pr, H=cssh-pt-pb;
  let xmin=Math.min(...xs),xmax=Math.max(...xs); if(xmax-xmin<1e-6)xmax=xmin+1;
  let ymin=Math.min(0,...ys),ymax=Math.max(...ys,1e-6); if(opts.ymin!==undefined)ymin=Math.min(ymin,opts.ymin);
  (opts.extra||[]).forEach(v=>{if(v!=null){ymax=Math.max(ymax,v);ymin=Math.min(ymin,v);}});
  const padY=(ymax-ymin)*0.08+1e-6; ymin-=padY; ymax+=padY;
  const X=v=>pl+(v-xmin)/(xmax-xmin)*W, Y=v=>pt+(1-(v-ymin)/(ymax-ymin))*H;
  // grid + axes
  g.strokeStyle=cssv("--grid"); g.lineWidth=1; g.fillStyle=cssv("--faint");
  g.font="10px "+cssv("--mono"); g.textAlign="right"; g.textBaseline="middle";
  for(let i=0;i<=4;i++){const yy=ymin+(ymax-ymin)*i/4; const py=Y(yy);
    g.beginPath();g.moveTo(pl,py);g.lineTo(pl+W,py);g.stroke();
    g.fillText(yy.toPrecision(3).replace(/\.?0+$/,""),pl-6,py);}
  g.textAlign="center";g.textBaseline="top";
  for(let i=0;i<=5;i++){const xx=xmin+(xmax-xmin)*i/5; g.fillText(xx.toFixed(1),X(xx),pt+H+6);}
  // zero line
  if(ymin<0&&ymax>0){g.strokeStyle=cssv("--line");g.beginPath();g.moveTo(pl,Y(0));g.lineTo(pl+W,Y(0));g.stroke();}
  // threshold
  if(opts.thr!=null){g.strokeStyle=cssv("--faint");g.setLineDash([5,4]);g.beginPath();
    g.moveTo(pl,Y(opts.thr));g.lineTo(pl+W,Y(opts.thr));g.stroke();g.setLineDash([]);
    g.fillStyle=cssv("--faint");g.textAlign="left";g.fillText("Q_thr",pl+3,Y(opts.thr)-8);}
  // t_coll marker
  if(opts.tcoll!=null){g.strokeStyle=cssv("--charged");g.setLineDash([2,3]);g.beginPath();
    g.moveTo(X(opts.tcoll),pt);g.lineTo(X(opts.tcoll),pt+H);g.stroke();g.setLineDash([]);}
  // hit markers
  (opts.hits||[]).forEach(ht=>{g.strokeStyle=cssv("--accent");g.lineWidth=1.4;g.beginPath();
    g.moveTo(X(ht),pt);g.lineTo(X(ht),pt+H);g.stroke();});
  // trace
  g.strokeStyle=opts.color;g.lineWidth=1.8;g.beginPath();
  xs.forEach((xv,i)=>{const px=X(xv),py=Y(ys[i]);i?g.lineTo(px,py):g.moveTo(px,py);});g.stroke();
  // ylabel
  g.save();g.translate(12,pt+H/2);g.rotate(-Math.PI/2);g.fillStyle=cssv("--faint");
  g.textAlign="center";g.textBaseline="middle";g.fillText(opts.ylab,0,0);g.restore();
  g.fillStyle=cssv("--faint");g.textAlign="right";g.textBaseline="bottom";
  g.fillText("t (µs)",pl+W,pt+H+22);
}

function drawWaves(){
  const m=document.getElementById("meta"), c1=document.getElementById("cvCur"), c2=document.getElementById("cvChg");
  if(!sel){m.innerHTML='<div class="k">pick a pixel</div><div class="v">&larr; from the event map</div>';
    [c1,c2].forEach(c=>{const g=c.getContext("2d");g.clearRect(0,0,c.width,c.height);});return;}
  const p=sel, col=cssv(CATVAR[p.cat]||"--quiet");
  m.innerHTML=`
    <div class="k">category</div><div class="v"><span class="tag" style="background:${col}">${CATNAME[p.cat]}</span></div>
    <div class="k">pixel id</div><div class="v">${p.id} &nbsp;(${p.x.toFixed(2)}, ${p.y.toFixed(2)}) cm</div>
    <div class="k">charge landed</div><div class="v">${(p.q_coll/1e3).toFixed(2)} ×10³ e⁻ ${p.q_coll>0?"":"(none)"}</div>
    <div class="k">arrival t_coll</div><div class="v">${p.t_coll==null?"—":p.t_coll.toFixed(2)+" µs"}${p.sig_t!=null?"  (σ<sub>t</sub> "+p.sig_t.toFixed(2)+")":""}</div>
    <div class="k">recorded hits</div><div class="v">${p.n_hits} ${p.hit_t.length?"@ "+p.hit_t.map(t=>t.toFixed(2)).join(", ")+" µs":""}</div>
    <div class="k">hit charge</div><div class="v">${p.hit_q.length?p.hit_q.map(q=>(q/1e3).toFixed(1)).join(", ")+" ×10³ e⁻":"—"}</div>`;
  trace(c1, p.t, p.cur, {h:190, color:col, ylab:"induced current (arb)", tcoll:p.t_coll, hits:p.hit_t});
  trace(c2, p.t, p.chg, {h:190, color:col, ylab:"charge (e⁻)", thr:p.thr, tcoll:p.t_coll,
    hits:p.hit_t, extra:[p.thr]});
}

function draw(){drawMap();drawWaves();}
sample=samples[0]; selS.value=sample; fillEvents(); ev=selE.value;
// auto-select first single-hit pixel
const P0=pixels(); sel=P0.find(p=>p.cat==="coll1")||P0.find(p=>p.cat==="ind1")||P0[0]||null;
draw();
new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
matchMedia("(prefers-color-scheme:dark)").addEventListener("change",draw);
addEventListener("resize",drawWaves);
"""

HTML = """<title>Pixel waveform viewer</title>
<style>%CSS%</style>
<h1>Pixel waveform viewer</h1>
<p class="sub">Per-pixel induced <b>current</b> and running <b>charge</b> vs time for single-hit
pixels in the study. The charge trace shows the discriminator threshold, the recorded hit
time(s) (blue) and the truth charge-arrival time t<sub>coll</sub> (amber) &mdash; so you can see
whether a hit fired <i>because</i> its charge arrived (collection) or <i>before</i> it, driven by
a neighbour's transient (induction). Pick a sample, an event, then a pixel from the map.</p>
<div class="bar">
  <span><label>sample</label><select id="selSample"></select></span>
  <span><label>event</label><select id="selEvent"></select></span>
</div>
<div class="wrap">
  <div class="card">
    <h2>Event map &mdash; click a pixel</h2>
    <svg id="map" xmlns="http://www.w3.org/2000/svg"></svg>
    <div class="legend">
      <span><i style="background:var(--coll)"></i>single collection</span>
      <span><i style="background:var(--ind)"></i>single induction</span>
      <span><i style="background:var(--multi)"></i>multi-hit</span>
      <span><i style="background:var(--charged)"></i>charge, no hit</span>
      <span><i style="background:var(--quiet)"></i>quiet</span>
    </div>
    <div class="hint">only a curated subset of pixels per event is saved</div>
  </div>
  <div class="card">
    <h2>Selected pixel</h2>
    <div class="meta" id="meta"></div>
    <canvas id="cvCur"></canvas>
    <div class="cap">induced current vs time &mdash; bipolar (up-then-down) = induction; unipolar = collection</div>
    <canvas id="cvChg"></canvas>
    <div class="cap">running charge (integral of current) vs time &mdash; a hit fires when it crosses Q_thr</div>
  </div>
</div>
<script>%JS%</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("waveforms", help="ti_waveforms.npz from the study")
    ap.add_argument("-o", "--out", default="wf_viewer.html")
    a = ap.parse_args()
    recs = load_records(a.waveforms)
    html = (HTML.replace("%CSS%", CSS)
                .replace("%JS%", JS.replace("__DATA__", json.dumps(recs)))
    )
    open(a.out, "w").write(html)
    print("wrote %s (%d pixels, %.0f KB)" % (a.out, len(recs), os.path.getsize(a.out) / 1024))


if __name__ == "__main__":
    main()
