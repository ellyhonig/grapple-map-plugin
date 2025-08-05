/* -----------------------------------------------------------
 * 3-D GrappleMap demo – super-compact version with
 *  • 14-joint avatar  (foot-knee-hip-shoulder-hand + head)
 *  • minimal sphere-constraint IK
 *  • camera orbit / zoom
 *  • search-able pose list (from states.js)
 * --------------------------------------------------------- */

class V3 {
  constructor(x=0,y=0,z=0){ this.x=x; this.y=y; this.z=z; }
  clone(){ return new V3(this.x,this.y,this.z); }
  add(v){ this.x+=v.x; this.y+=v.y; this.z+=v.z; return this; }
  sub(v){ this.x-=v.x; this.y-=v.y; this.z-=v.z; return this; }
  mul(s){ this.x*=s; this.y*=s; this.z*=s; return this; }
  len(){ return Math.hypot(this.x,this.y,this.z); }
  norm(){ const l=this.len(); return l?this.mul(1/l):this; }
  dot(v){ return this.x*v.x+this.y*v.y+this.z*v.z; }
  cross(v){return new V3(this.y*v.z-this.z*v.y,this.z*v.x-this.x*v.z,this.x*v.y-this.y*v.x);}
}
const PARENT = [
  1,   // 0: leftFoot  ← leftKnee
  2,   // 1: leftKnee  ← leftHip
 -1,   // 2: leftHip   ← (root)
  2,   // 3: core      ← leftHip
  3,   // 4: leftShoulder ← core
  4,   // 5: leftElbow   ← leftShoulder
  5,   // 6: leftHand    ← leftElbow
  3,   // 7: head        ← core
  9,   // 8: rightHand   ← rightElbow
 10,   // 9: rightElbow  ← rightShoulder
  3,   //10: rightShoulder ← core
  2,   //11: rightHip    ← leftHip (root)
 11,   //12: rightKnee   ← rightHip
 12    //13: rightFoot   ← rightKnee
];
const HEAD_IDX       = 7;
const HIP_IDX        = 2;
const SHOULDER_L_IDX = 4;
const SHOULDER_R_IDX = 10;

/* ---------- tiny skeleton/IK helpers ---------------------------------- */
const COLOR  = ['#e74c3c','#3498db'];                                      // red / blue
const BONES = [
  [0,1],[1,2],[2,3],    // left leg & hip-core
  [3,4],[4,5],[5,6],    // left arm
  [3,7],                // neck
  [3,10],[10,9],[9,8],  // right arm
  [2,11],[11,12],[12,13]// right leg
];

function propagateDescendants(i, Δ, av){
  av.j.forEach((p, idx)=>{
    if (PARENT[idx] === i){
      const old = p.clone();
      p.add(Δ);
      clampToParent(idx, av);
      const realΔ = p.clone().sub(old);
      propagateDescendants(idx, realΔ, av);
    }
  });
}
function clampRadialDelta(i, Δ, av){
  const maxW = av.shoulderWidth;
  const partners = (i === HEAD_IDX)
    ? [SHOULDER_L_IDX, SHOULDER_R_IDX]
    : [HEAD_IDX];
  let d = Δ.clone();
  for (const j of partners){
    const v = av.j[i].clone().sub(av.j[j]);
    const L = v.len();
    if (L < 1e-6) continue;
    const u = v.mul(1/L);                   // unit from j→i
    const rad = u.dot(d);                   // how much Δ pushes out
    if (rad <= 0) continue;                 // inward or tangent ok
    const allowed = maxW - L;
    if (L + rad > maxW){
      const remove = rad - allowed;
      d = d.sub(u.mul(remove));             // strip excess radial
    }
  }
  return d;
}
function clampNeckShoulder(i, av){
  const W = av.shoulderWidth,
        h = av.j[HEAD_IDX],
        sL = av.j[SHOULDER_L_IDX],
        sR = av.j[SHOULDER_R_IDX];

  const clampPoint = (a, bIdx)=>{            // pull a toward b if too far
    const b = av.j[bIdx], dir = a.clone().sub(b), d = dir.len();
    if (d > W){
      const newPos = b.clone().add(dir.mul(W/d)),
            Δ = newPos.clone().sub(a);
      av.j[bIdx === HEAD_IDX ? HEAD_IDX : i] = newPos;
      propagateDescendants(bIdx === HEAD_IDX ? HEAD_IDX : i, Δ, av);
      clampToParent(bIdx === HEAD_IDX ? HEAD_IDX : i, av);
    }
  };

  if (i === HEAD_IDX){ clampPoint(h, SHOULDER_L_IDX); clampPoint(h, SHOULDER_R_IDX); }
  else if (i === SHOULDER_L_IDX || i === SHOULDER_R_IDX){ clampPoint(h, i); }
}

function makeAvatar(skel){
  const j  = skel.map(p => new V3(p.x,p.y,p.z));
  const len= j.map((_,i)=>PARENT[i]<0 ? 0 : j[i].clone().sub(j[PARENT[i]]).len());
  const shoulderWidth = j[SHOULDER_L_IDX].clone().sub(j[SHOULDER_R_IDX]).len();
  return { j, len, shoulderWidth };
}

function clampToParent(i,av){
  const p=PARENT[i]; if(p<0) return;
  const dir = av.j[i].clone().sub(av.j[p]);
  const L   = av.len[i]||0;
  if(!L) return;
  av.j[i] = dir.len()<1e-6 ? av.j[p].clone().add(new V3(L,0,0))
                           : av.j[p].clone().add(dir.norm().mul(L));
}

function moveJoint(i, Δ, av){
  // prevent radial stretch on head/shoulder
  if ([HEAD_IDX, SHOULDER_L_IDX, SHOULDER_R_IDX].includes(i)){
    Δ = clampRadialDelta(i, Δ, av);
  }

  if (PARENT[i] < 0){
    av.j.forEach((p, idx)=>{ if (idx !== i) p.add(Δ); });
    av.j[i].add(Δ);
    return;
  }

  const oldPos = av.j[i].clone();
  av.j[i].add(Δ);
  clampToParent(i, av);
  const realΔ = av.j[i].clone().sub(oldPos);
  propagateDescendants(i, realΔ, av);
}

/* ---------- state & UI ------------------------------------------------ */
const canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d');
let W=canvas.width=innerWidth, H=canvas.height=innerHeight;
addEventListener('resize',()=>{W=canvas.width=innerWidth;H=canvas.height=innerHeight;});

let radius=8, theta=Math.PI/4, phi=Math.PI/6;
const basis=()=>{                       // camera basis
  const sp=Math.sin(phi),   cp=Math.cos(phi),
        st=Math.sin(theta), ct=Math.cos(theta);
  const pos=new V3(radius*cp*st, radius*sp, radius*cp*ct);
  const fwd=pos.clone().mul(-1).norm(),  up=new V3(0,1,0);
  const right=fwd.cross(up).norm(),      up2=right.clone().cross(fwd).norm();
  return {pos,fwd,right,up:up2};
};

function project(p,b){
  const r=p.clone().sub(b.pos);
  const x=r.dot(b.right), y=r.dot(b.up), z=r.dot(b.fwd);
  if(z<0.1) return null;
  const s=H*0.8;
  return {x:s*x/z+W/2,y:-s*y/z+H/2,z};
}

/* ---------- load poses (from states.js or optional json) -------------- */
let STATES=[];
(async()=>{
  if(Array.isArray(window.PRECOMPUTED_STATES)) STATES=window.PRECOMPUTED_STATES;
  else {
    const res=await fetch('grapple_states.json');    // fallback local JSON
    STATES=await res.json();
  }
  STATES=STATES.slice(0,500);                        // safety cap
  populateDropdown();
  applyState(0);
})();

function populateDropdown(){
  const sel=document.getElementById('state-select'), q=document.getElementById('state-search');
  const rebuild=()=>{
    const f=(q.value||'').trim().toLowerCase();
    sel.innerHTML='';
    STATES.forEach((s,i)=>{
      if(!f||s.name.toLowerCase().includes(f)){
        const o=document.createElement('option');
        o.value=i; o.textContent=s.name.replace(/\n/g,' / ');
        sel.appendChild(o);
      }
    });
  };
  q.oninput=rebuild; rebuild();
  sel.onchange=e=>applyState(+e.target.value);
}

/* ---------- live avatars ---------------------------------------------- */
const AV=[null,null];                        // red, blue
let selected={idx:-1,av:-1},  dragging=false,x0=0,y0=0;

/* skeleton from state (14 points already in file) -> avatar ------------ */
function applyState(idx){
  const s=STATES[idx];
  AV[0]=makeAvatar(s.skeleton1 || s.chain1 /*fallback*/);
  AV[1]=makeAvatar(s.skeleton2 || s.chain2);
  document.getElementById('nearest-state').textContent=s.name.replace(/\n/g,' / ');
}

/* ---------- picking & interaction ------------------------------------ */
canvas.oncontextmenu=e=>e.preventDefault();
canvas.onmousedown=e=>{
  x0=e.clientX; y0=e.clientY;
  if(e.button===2){ dragging='cam'; return; }
  const b=basis();
  let best={d:1e9,av:-1,idx:-1};
  AV.forEach((av,a)=>{
    av.j.forEach((p,i)=>{
      const pr=project(p,b); if(!pr) return;
      const d=Math.hypot(pr.x-e.clientX, pr.y-e.clientY);
      if(d<best.d && d<12) best={d,av:a,idx:i};
    });
  });
  if(best.av>-1){ selected=best; dragging='joint'; }
};
addEventListener('mouseup',()=>{ dragging=false; selected.idx=-1; });
addEventListener('mousemove',e=>{
  if(!dragging) return;
  const dx=e.clientX-x0, dy=e.clientY-y0; x0=e.clientX; y0=e.clientY;
  if(dragging==='cam'){
    theta-=dx*0.005; phi =Math.min(Math.PI-0.01, Math.max(0.01,phi-dy*0.005)); return;
  }
  const b=basis(), av=AV[selected.av], i=selected.idx;
  const depth=av.j[i].clone().sub(b.pos).dot(b.fwd), scale=depth/(H*0.8);
  const delta=b.right.clone().mul(dx*scale).add(b.up.clone().mul(-dy*scale));
  moveJoint(i,delta,av);
});
addEventListener('wheel',e=>{ radius=Math.max(3,Math.min(20,radius*Math.exp(e.deltaY*0.001))); });

/* ---------- rendering ------------------------------------------------- */
function drawAvatar(av,col,b){
  const proj = av.j.map(p=>project(p,b));
  ctx.strokeStyle = ctx.fillStyle = col;
  BONES.forEach(([a,b])=>{
    const A=proj[a], B=proj[b]; if(!A||!B) return;
    ctx.lineWidth = Math.max(1,8/((A.z+B.z)*0.5));
    ctx.beginPath(); ctx.moveTo(A.x,A.y); ctx.lineTo(B.x,B.y); ctx.stroke();
  });
  proj.forEach(p=>{
    if(!p) return;
    const r = Math.max(2,10/p.z);
    ctx.beginPath(); ctx.arc(p.x,p.y,r,0,Math.PI*2); ctx.fill();
  });
    // --- neck endpoint at 80% of core→head ---
  const core3D = av.j[PARENT[HEAD_IDX]];              // core idx = 3
  const head3D = av.j[HEAD_IDX];
  const neckDir = head3D.clone().sub(core3D).norm();
  const neckLen = av.len[HEAD_IDX] * 0.8;
  const neckPoint3D = core3D.clone().add(neckDir.mul(neckLen));
  const neckProj = project(neckPoint3D, b);

  // --- shoulders projection ---
  const sl = proj[SHOULDER_L_IDX], sr = proj[SHOULDER_R_IDX];

  if (neckProj) {
    ctx.lineWidth =  Math.max(1, 6 / neckProj.z);
    ctx.beginPath();
    if (sl) { ctx.moveTo(sl.x, sl.y); ctx.lineTo(neckProj.x, neckProj.y); }
    if (sr) { ctx.moveTo(sr.x, sr.y); ctx.lineTo(neckProj.x, neckProj.y); }
    ctx.stroke();
  }
}

function linkNumber(){
  // quick 2-D signed crossing count (XY)
  const segs=a=>a.j.slice(0,-1).map((p,i)=>[p,a.j[i+1]]);
  const o=(a,b,c)=> (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
  let c=0; segs(AV[0]).forEach(as=>{
    segs(AV[1]).forEach(bs=>{
      const [a1,a2]=as,[b1,b2]=bs;
      if(o(a1,a2,b1)*o(a1,a2,b2)<0 && o(b1,b2,a1)*o(b1,b2,a2)<0)
        c+=o(a1,a2,b1)>0?1:-1;
    });
  });
  return c/2;
}

(function loop(){
  ctx.clearRect(0,0,W,H);
  if(!AV[0]) return requestAnimationFrame(loop);
  const b=basis();
  drawAvatar(AV[0],COLOR[0],b);
  drawAvatar(AV[1],COLOR[1],b);
  document.getElementById('link-value').textContent=linkNumber().toFixed(1);
  requestAnimationFrame(loop);
})();
