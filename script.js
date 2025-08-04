/*
 * Simple 3D grapple demo based on topology coordinates.
 *
 * This script renders two chains (avatars) in a 3D space using a basic
 * perspective projection. Joints can be selected with the left mouse
 * button and dragged to new positions. Bone lengths remain fixed by
 * propagating positional changes along each chain. A rudimentary
 * linking number is computed by projecting the chains onto the XY plane
 * and summing oriented crossings.
 *
 * Camera controls: Right‑drag to rotate around the origin, scroll to zoom.
 */

class Vec3 {
  constructor(x = 0, y = 0, z = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
  }
  clone() {
    return new Vec3(this.x, this.y, this.z);
  }
  copy(v) {
    this.x = v.x;
    this.y = v.y;
    this.z = v.z;
    return this;
  }
  add(v) {
    this.x += v.x;
    this.y += v.y;
    this.z += v.z;
    return this;
  }
  sub(v) {
    this.x -= v.x;
    this.y -= v.y;
    this.z -= v.z;
    return this;
  }
  multiplyScalar(s) {
    this.x *= s;
    this.y *= s;
    this.z *= s;
    return this;
  }
  length() {
    return Math.hypot(this.x, this.y, this.z);
  }
  normalize() {
    const len = this.length();
    if (len > 0) {
      this.multiplyScalar(1 / len);
    }
    return this;
  }
  dot(v) {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }
  cross(v) {
    return new Vec3(
      this.y * v.z - this.z * v.y,
      this.z * v.x - this.x * v.z,
      this.x * v.y - this.y * v.x
    );
  }
}

// Chain class: represents a kinematic chain composed of joints and fixed segment lengths
class Chain {
  constructor(joints) {
    this.joints = joints; // array of Vec3
    this.lengths = [];
    for (let i = 0; i < joints.length - 1; i++) {
      const diff = joints[i + 1].clone().sub(joints[i]);
      this.lengths.push(diff.length());
    }
  }
  // Adjust the chain after moving joint at index idx
  adjust(idx) {
    // forward propagation: adjust joints after idx
    for (let i = idx; i < this.joints.length - 1; i++) {
      const a = this.joints[i];
      const b = this.joints[i + 1];
      const dir = b.clone().sub(a);
      const len = dir.length();
      const targetLen = this.lengths[i];
      if (len === 0) continue;
      dir.multiplyScalar((targetLen) / len);
      b.copy(a.clone().add(dir));
    }
    // backward propagation: adjust joints before idx
    for (let i = idx; i > 0; i--) {
      const a = this.joints[i];
      const b = this.joints[i - 1];
      const dir = b.clone().sub(a);
      const len = dir.length();
      const targetLen = this.lengths[i - 1];
      if (len === 0) continue;
      dir.multiplyScalar((targetLen) / len);
      b.copy(a.clone().add(dir));
    }
  }
}

(() => {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  let width, height;

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  // Camera parameters
  let camRadius = 10;
  let camTheta = Math.PI / 4; // yaw angle
  let camPhi = Math.PI / 6;   // pitch angle

  // Chains (avatars)
  const chain1 = new Chain([
    new Vec3(-1, 0, -1),
    new Vec3(-1, 1, -1),
    new Vec3(-1, 2, -1),
    new Vec3(-1, 3, -1),
    new Vec3(-1, 4, -1)
  ]);
  const chain2 = new Chain([
    new Vec3(1, 0, 1),
    new Vec3(1, 1, 1),
    new Vec3(1, 2, 1),
    new Vec3(1, 3, 1),
    new Vec3(1, 4, 1)
  ]);
  const chains = [chain1, chain2];

  // Interaction state
  let selected = null; // {chain: chain, index: number}
  let rotating = false;
  let lastX = 0;
  let lastY = 0;

  // Prevent context menu on right-click
  canvas.addEventListener('contextmenu', e => e.preventDefault());

  // Picking on mousedown
  canvas.addEventListener('mousedown', e => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    lastX = x;
    lastY = y;
    if (e.button === 0) {
      // Left button: attempt to select a joint
      const nearest = pickJoint(x, y);
      if (nearest) {
        selected = nearest;
      }
    } else if (e.button === 2) {
      // Right button: rotate camera
      rotating = true;
    }
  });

  document.addEventListener('mouseup', e => {
    selected = null;
    rotating = false;
  });

  document.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const dx = x - lastX;
    const dy = y - lastY;
    lastX = x;
    lastY = y;
    if (selected) {
      // Move selected joint in world space based on screen delta
      moveSelectedJoint(dx, dy);
    } else if (rotating) {
      // Rotate camera around origin
      camTheta -= dx * 0.005;
      camPhi -= dy * 0.005;
      // Clamp phi to avoid singularities
      const eps = 0.001;
      camPhi = Math.max(eps, Math.min(Math.PI - eps, camPhi));
    }
  });

  canvas.addEventListener('wheel', e => {
    const delta = e.deltaY;
    camRadius *= Math.exp(delta * 0.001);
    camRadius = Math.max(2, Math.min(50, camRadius));
    e.preventDefault();
  });

  // Compute camera basis vectors
  function computeCameraBasis() {
    // Camera position in world
    const sinPhi = Math.sin(camPhi);
    const cosPhi = Math.cos(camPhi);
    const sinTheta = Math.sin(camTheta);
    const cosTheta = Math.cos(camTheta);
    const camPos = new Vec3(
      camRadius * cosPhi * sinTheta,
      camRadius * sinPhi,
      camRadius * cosPhi * cosTheta
    );
    // Forward: from camera towards origin
    const forward = camPos.clone().multiplyScalar(-1).normalize();
    // Approx global up vector
    const globalUp = new Vec3(0, 1, 0);
    let right = forward.cross(globalUp).normalize();
    // If forward is parallel to up, choose another up vector
    if (right.length() === 0) {
      right = new Vec3(1, 0, 0);
    }
    const up = right.clone().cross(forward).normalize();
    return { camPos, forward, right, up };
  }

  // Project a 3D point to 2D canvas coordinates
  function projectPoint(pt, basis) {
    const { camPos, forward, right, up } = basis;
    const relative = pt.clone().sub(camPos);
    // Coordinates in camera space
    const xCam = relative.dot(right);
    const yCam = relative.dot(up);
    const zCam = relative.dot(forward);
    // If point is behind the camera, skip projection by returning null
    if (zCam <= 0.1) return null;
    const fovScale = height * 0.8; // focal length scaled by canvas
    const x2d = (fovScale * xCam) / zCam + width / 2;
    const y2d = (-fovScale * yCam) / zCam + height / 2;
    return { x: x2d, y: y2d, z: zCam };
  }

  // Pick the closest joint near the mouse position
  function pickJoint(px, py) {
    const basis = computeCameraBasis();
    let minDist = 12; // pixel threshold
    let hit = null;
    chains.forEach((chain, ci) => {
      chain.joints.forEach((pt, idx) => {
        const proj = projectPoint(pt, basis);
        if (!proj) return;
        const dx = proj.x - px;
        const dy = proj.y - py;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < minDist) {
          minDist = dist;
          hit = { chain, index: idx };
        }
      });
    });
    return hit;
  }

  // Move selected joint based on screen delta
  function moveSelectedJoint(dx, dy) {
    const basis = computeCameraBasis();
    const { right, up } = basis;
    // Determine movement scale relative to distance from camera
    const selPt = selected.chain.joints[selected.index];
    // Project selected point to camera space to get depth
    const rel = selPt.clone().sub(basis.camPos);
    const depth = rel.dot(basis.forward);
    const scale = (depth / (height * 0.8));
    // Compute world delta: right and up directions scaled by screen movement
    const worldDelta = right.clone().multiplyScalar(dx * scale).add(
      up.clone().multiplyScalar(-dy * scale)
    );
    selPt.add(worldDelta);
    // Adjust chain to maintain bone lengths
    selected.chain.adjust(selected.index);
  }

  // Compute approximate linking number (project onto XY plane)
  function computeLinking() {
    let crossings = 0;
    const segs = (chain) => {
      const arr = [];
      for (let i = 0; i < chain.joints.length - 1; i++) {
        const p = chain.joints[i];
        const q = chain.joints[i + 1];
        arr.push({ p, q });
      }
      return arr;
    };
    const s1 = segs(chain1);
    const s2 = segs(chain2);
    // 2D oriented intersection test on XY plane
    function orient(a, b, c) {
      return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    }
    function onSegment(a, b, c) {
      return Math.min(a.x, b.x) <= c.x && c.x <= Math.max(a.x, b.x) &&
             Math.min(a.y, b.y) <= c.y && c.y <= Math.max(a.y, b.y);
    }
    for (let segA of s1) {
      for (let segB of s2) {
        const a1 = segA.p;
        const a2 = segA.q;
        const b1 = segB.p;
        const b2 = segB.q;
        const o1 = orient(a1, a2, b1);
        const o2 = orient(a1, a2, b2);
        const o3 = orient(b1, b2, a1);
        const o4 = orient(b1, b2, a2);
        if ((o1 * o2 < 0) && (o3 * o4 < 0)) {
          // count orientation of crossing
          const sign = o1 > 0 ? 1 : -1;
          crossings += sign;
        }
      }
    }
    // The linking number of open curves is half the sum of signed crossings
    return crossings / 2;
  }

  // Render loop
  function render() {
    ctx.clearRect(0, 0, width, height);
    // Compute camera basis once per frame
    const basis = computeCameraBasis();
    const { camPos, forward, right, up } = basis;
    // Draw each chain
    chains.forEach((chain, ci) => {
      // Choose color for chain
      const color = ci === 0 ? '#e74c3c' : '#3498db';
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      // Draw segments
      ctx.beginPath();
      let first = true;
      for (let i = 0; i < chain.joints.length; i++) {
        const proj = projectPoint(chain.joints[i], basis);
        if (!proj) continue;
        if (first) {
          ctx.moveTo(proj.x, proj.y);
          first = false;
        } else {
          ctx.lineTo(proj.x, proj.y);
        }
      }
      ctx.stroke();
      // Draw joints as circles
      ctx.fillStyle = color;
      for (let i = 0; i < chain.joints.length; i++) {
        const proj = projectPoint(chain.joints[i], basis);
        if (!proj) continue;
        const r = 5;
        ctx.beginPath();
        ctx.arc(proj.x, proj.y, r, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    // Display linking number
    const linkValueEl = document.getElementById('link-value');
    linkValueEl.textContent = computeLinking().toFixed(1);
    requestAnimationFrame(render);
  }
  render();
})();