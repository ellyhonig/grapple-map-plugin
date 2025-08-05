/*
 * Enhanced 3D grapple demo based on topology coordinates and a simple
 * finite state machine. This script renders two chains (avatars) in a 3D
 * space and allows the user to select between pre-defined states or
 * automatically loaded entries from the GrappleMap database. Joints can
 * be selected with the left mouse button and dragged to new positions.
 * Bone lengths remain fixed by propagating positional changes along each
 * chain. A rudimentary linking number is computed by projecting the
 * chains onto the XY plane and summing oriented crossings. The current
 * linking number is displayed in the overlay. A drop‑down menu allows
 * selecting between available states; when a state is selected the chains
 * are updated accordingly.
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
      dir.multiplyScalar(targetLen / len);
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
      dir.multiplyScalar(targetLen / len);
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
    new Vec3(-1, 3.5, -1),
    new Vec3(-1, 4, -1)
  ]);
  const chain2 = new Chain([
    new Vec3(1, 0, 1),
    new Vec3(1, 1, 1),
    new Vec3(1, 2, 1),
    new Vec3(1, 3, 1),
    new Vec3(1, 3.5, 1),
    new Vec3(1, 4, 1)
  ]);
  const chains = [chain1, chain2];

  // --- Finite state machine support ---
  // Array of available states. Each state contains a name, positions for both chains,
  // and a precomputed feature vector used for nearest-neighbour queries in the
  // latent (topology) space. A state's feature vector is of the form
  // [linkingNumber, centreDistance] where centreDistance measures the
  // separation between the chains' centroids.
  const states = [];
  // Index of the last applied state. This is used to avoid repeatedly
  // re‑applying the same state when the user drags a joint.
  let currentStateIndex = -1;

  // The skeletons for the two avatars in the current state. These arrays
  // contain 10 points each representing major body parts (feet, knees,
  // hips, shoulders, hands and head) for left and right sides. They are
  // initialised with the fallback "Initial" pose and updated whenever
  // applyState() is invoked.
  let currentSkeleton1 = [];
  let currentSkeleton2 = [];
  // Store previous joint positions at the start of a drag so that we can
  // revert if the new configuration self‑intersects or collides.
  let prevPositions = null;

  /**
   * Return a small set of fallback states if loading the GrappleMap database fails.
   * Each state defines the joint positions for both chains. These positions
   * illustrate different topological configurations such as no crossing,
   * crossing and twisting.
   */
  function getFallbackStates() {
    // Helper to create a mirrored skeleton from a simplified six‑joint chain.
    // The chain layout is [foot, knee, hip, shoulder, hand, head]. We
    // reconstruct a 10‑joint skeleton by mirroring left‑side joints across
    // the hips to obtain right‑side joints. This yields an approximate
    // humanoid figure for fallback poses when the full GrappleMap data is
    // unavailable.
    function skeletonFromChain(chain) {
      const [footL, kneeL, hip, shoulder, hand, head] = chain;
      // Mirror a point across the hip in the x‑direction
      const mirror = (p) => {
        const dx = p.x - hip.x;
        return { x: hip.x - dx, y: p.y, z: p.z };
      };
      const footR = mirror(footL);
      const kneeR = mirror(kneeL);
      const shoulderR = mirror(shoulder);
      const handR = mirror(hand);
      return [
        footL,
        kneeL,
        hip,
        shoulder,
        hand,
        head,
        handR,
        shoulderR,
        kneeR,
        footR
      ];
    }
    // Define fallback poses with skeletons derived from their chains
    const base = [
      {
        name: 'Initial',
        chain1: [
          { x: -1, y: 0, z: -1 },
          { x: -1, y: 1, z: -1 },
          { x: -1, y: 2, z: -1 },
          { x: -1, y: 3, z: -1 },
          { x: -1, y: 3.5, z: -1 },
          { x: -1, y: 4, z: -1 }
        ],
        chain2: [
          { x: 1, y: 0, z: 1 },
          { x: 1, y: 1, z: 1 },
          { x: 1, y: 2, z: 1 },
          { x: 1, y: 3, z: 1 },
          { x: 1, y: 3.5, z: 1 },
          { x: 1, y: 4, z: 1 }
        ]
      },
      {
        name: 'Cross',
        chain1: [
          { x: -2, y: 0, z: -1 },
          { x: -1, y: 1, z: -1 },
          { x: 0, y: 2, z: -1 },
          { x: 1, y: 3, z: -1 },
          { x: 1.5, y: 3.5, z: -1 },
          { x: 2, y: 4, z: -1 }
        ],
        chain2: [
          { x: 2, y: 0, z: 1 },
          { x: 1, y: 1, z: 1 },
          { x: 0, y: 2, z: 1 },
          { x: -1, y: 3, z: 1 },
          { x: -1.5, y: 3.5, z: 1 },
          { x: -2, y: 4, z: 1 }
        ]
      },
      {
        name: 'Twist',
        chain1: [
          { x: -1, y: 0, z: 0 },
          { x: -0.5, y: 1, z: 0.5 },
          { x: 0, y: 2, z: 1 },
          { x: 0.5, y: 3, z: 0.5 },
          { x: 0.75, y: 3.5, z: 0.25 },
          { x: 1, y: 4, z: 0 }
        ],
        chain2: [
          { x: 1, y: 0, z: 0 },
          { x: 0.5, y: 1, z: -0.5 },
          { x: 0, y: 2, z: -1 },
          { x: -0.5, y: 3, z: -0.5 },
          { x: -0.75, y: 3.5, z: -0.25 },
          { x: -1, y: 4, z: 0 }
        ]
      }
    ];
    return base.map((b) => {
      return {
        name: b.name,
        chain1: b.chain1,
        chain2: b.chain2,
        skeleton1: skeletonFromChain(b.chain1),
        skeleton2: skeletonFromChain(b.chain2)
      };
    });
  }

  /**
   * Decode a position string from the GrappleMap database. The encoded
   * representation uses base62 digits (a‑zA‑Z0‑9) to store joint
   * coordinates at millimeter precision. Each coordinate is encoded by
   * two base62 digits and scaled to meters. The decoding logic is based on
   * the official C++ implementation (see persistence.cpp in GrappleMap).
   *
   * @param {string} s Encoded position string (concatenation of four lines)
   * @returns {Array<{x:number,y:number,z:number}>} Array of joint positions
   *          for both players (player 0 joints followed by player 1 joints).
   */
  function decodePositionString(s) {
    const base62 = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    const map = {};
    for (let i = 0; i < base62.length; i++) map[base62[i]] = i;
    let offset = 0;
    const nextdigit = () => {
      // skip whitespace
      while (offset < s.length && /\s/.test(s[offset])) offset++;
      return map[s[offset++]];
    };
    const g = () => {
      const d0 = nextdigit() * 62;
      const d1 = nextdigit();
      return (d0 + d1) / 1000;
    };
    const positions = [];
    const jointCount = 23;
    const totalJoints = jointCount * 2;
    for (let i = 0; i < totalJoints; i++) {
      const x = g() - 2;
      const y = g();
      const z = g() - 2;
      positions.push({ x, y, z });
    }
    return positions;
  }

  /**
   * Derive a simplified chain of six joints (foot, knee, hip, shoulder,
   * hand, head) from a full set of 23 joint positions for one player. The
   * positions are averaged across corresponding left/right joints. The indices
   * of the underlying joints are based on the order defined in GrappleMap's
   * playerJoints array (see positions.hpp).
   *
   * @param {Array<{x:number,y:number,z:number}>} joints Array of 23 joint
   *    positions for one player.
   * @returns {Array<{x:number,y:number,z:number}>} Simplified chain with
   *    six joint positions.
   */
  function deriveChain(joints) {
    // Helper to average a list of points
    const avg = (indices) => {
      const out = { x: 0, y: 0, z: 0 };
      indices.forEach((idx) => {
        out.x += joints[idx].x;
        out.y += joints[idx].y;
        out.z += joints[idx].z;
      });
      const inv = 1 / indices.length;
      out.x *= inv;
      out.y *= inv;
      out.z *= inv;
      return out;
    };
    // Indices mapping for joints in GrappleMap
    const LeftToe = 0, RightToe = 1;
    const LeftHeel = 2, RightHeel = 3;
    const LeftAnkle = 4, RightAnkle = 5;
    const LeftKnee = 6, RightKnee = 7;
    const LeftHip = 8, RightHip = 9;
    const LeftShoulder = 10, RightShoulder = 11;
    const LeftElbow = 12, RightElbow = 13;
    const LeftWrist = 14, RightWrist = 15;
    const LeftHand = 16, RightHand = 17;
    //const LeftFingers = 18, RightFingers = 19;
    const Core = 20;
    const Neck = 21;
    const Head = 22;
    // Build simplified joints
    const foot = avg([LeftToe, RightToe, LeftHeel, RightHeel, LeftAnkle, RightAnkle]);
    const knee = avg([LeftKnee, RightKnee]);
    const hip = avg([LeftHip, RightHip, Core]);
    const shoulder = avg([LeftShoulder, RightShoulder]);
    const hand = avg([LeftHand, RightHand, LeftWrist, RightWrist]);
    const head = joints[Head];
    return [foot, knee, hip, shoulder, hand, head];
  }

  /**
  /**
   * Derive an extended humanoid skeleton from the full set of joints for one
   * player.  This function returns 18 points representing key body parts on
   * both sides along with a central core and head.  The returned array
   * contains the following points (in order):
   *   0: left toe (LeftToe)
   *   1: left ankle (LeftAnkle)
   *   2: left knee (LeftKnee)
   *   3: left hip (LeftHip)
   *   4: core (Core)
   *   5: left shoulder (LeftShoulder)
   *   6: left elbow (LeftElbow)
   *   7: left wrist (LeftWrist)
   *   8: left hand (LeftHand)
   *   9: head (Head)
   *   10: right hand (RightHand)
   *   11: right wrist (RightWrist)
   *   12: right elbow (RightElbow)
   *   13: right shoulder (RightShoulder)
   *   14: right hip (RightHip)
   *   15: right knee (RightKnee)
   *   16: right ankle (RightAnkle)
   *   17: right toe (RightToe)
   * This richer skeleton exposes ankles and wrists so that the avatars
   * resemble human figures more closely.  When available, we take the
   * corresponding GrappleMap joint directly.  Otherwise, fallback
   * approximations will interpolate between the neighbouring joints.
   *
   * @param {Array<{x:number,y:number,z:number}>} joints 23‑joint positions
   * @returns {Array<{x:number,y:number,z:number}>} 18‑joint skeleton
   */
  function deriveSkeletonPoints(joints) {
    // Direct mappings from the 23‑joint list
    const leftToe = joints[0];
    const rightToe = joints[1];
    const leftAnkle = joints[4];
    const rightAnkle = joints[5];
    const leftKnee = joints[6];
    const rightKnee = joints[7];
    const leftHip = joints[8];
    const rightHip = joints[9];
    const leftShoulder = joints[10];
    const rightShoulder = joints[11];
    const leftElbow = joints[12];
    const rightElbow = joints[13];
    const leftWrist = joints[14];
    const rightWrist = joints[15];
    const leftHand = joints[16];
    const rightHand = joints[17];
    // We do not use the finger points separately; palms are sufficient
    const core = joints[20];
    const head = joints[22];
    return [leftToe, leftAnkle, leftKnee, leftHip, core,
            leftShoulder, leftElbow, leftWrist, leftHand, head,
            rightHand, rightWrist, rightElbow, rightShoulder,
            rightHip, rightKnee, rightAnkle, rightToe];
  }

  /**
   * Parse the raw GrappleMap database text into an array of state objects.
   * Each state represents a single pose (sequence with exactly one encoded
   * position) extracted from the database. If an entry contains multiple
   * encoded positions (e.g. drills), each encoded block is treated as a
   * separate state. Only the first description line is used as the state
   * name; if missing a generic name is generated. The number of states is
   * limited to avoid overwhelming the browser.
   *
   * @param {string} text The raw contents of GrappleMap.txt
   * @param {number} [limit=100] Maximum number of states to extract
   * @returns {Array<{name:string,chain1:Array,chain2:Array}>}
   */
  function parseGrappleMap(text, limit = Infinity) {
    /*
     * Parse the GrappleMap database by scanning line by line.  Each entry
     * consists of one or more description lines (not starting with
     * whitespace and not beginning with "tags:"), followed by a tags line,
     * and then exactly four encoded lines starting with whitespace.  The
     * encoded lines store 23 joint positions for each player in a base62
     * format.  This parser reconstructs the entries without relying on
     * blank lines, which are absent in the database.  It yields up to
     * `limit` poses.
     */
    const result = [];
    const lines = text.split(/\n/);
    let nameLines = [];
    let encLines = [];
    const finalize = () => {
      if (!encLines.length) return;
      const name = nameLines.join('\n') || `Pose ${result.length + 1}`;
      const encoded = encLines.map((l) => l.trim()).join('');
      try {
        const positions = decodePositionString(encoded);
        if (positions.length >= 46) {
          const p0 = positions.slice(0, 23);
          const p1 = positions.slice(23, 46);
          const chain1Pos = deriveChain(p0);
          const chain2Pos = deriveChain(p1);
          const skeleton1 = deriveSkeletonPoints(p0);
          const skeleton2 = deriveSkeletonPoints(p1);
          result.push({
            name: name.trim(),
            chain1: chain1Pos,
            chain2: chain2Pos,
            skeleton1,
            skeleton2
          });
        }
      } catch (err) {
        console.warn('Failed to decode position for entry', name, err);
      }
      nameLines = [];
      encLines = [];
    };
    for (const line of lines) {
      if (!line.trim()) continue;
      if (/^\s/.test(line)) {
        // Encoded line begins with whitespace
        encLines.push(line);
        continue;
      }
      const trimmed = line.trim();
      if (trimmed.startsWith('tags:')) {
        // Ignore tags line
        continue;
      }
      // If we encounter a new description line and have accumulated encoded
      // lines from the previous entry, finalize the previous entry first.
      if (encLines.length) {
        finalize();
        if (result.length >= limit) break;
      }
      nameLines.push(trimmed);
    }
    // Finalize last entry
    if (encLines.length && result.length < limit) {
      finalize();
    }
    // In case the limit was reached within finalize()
    return result.slice(0, limit);
  }

  /**
   * Compute the centroid of a chain of Vec3 points.
   * @param {Vec3[]} pts Array of Vec3.
   * @returns {Vec3}
   */
  function centroidVec(pts) {
    const c = new Vec3();
    pts.forEach((p) => {
      c.x += p.x;
      c.y += p.y;
      c.z += p.z;
    });
    const inv = 1 / pts.length;
    c.x *= inv;
    c.y *= inv;
    c.z *= inv;
    return c;
  }

  /**
   * Compute the approximate linking number between two sets of joints.
   * This is identical to computeLinking() but accepts explicit joint arrays,
   * enabling us to calculate features for arbitrary states.
   * @param {Vec3[]} jointsA
   * @param {Vec3[]} jointsB
   * @returns {number}
   */
  function computeLinkingBetween(jointsA, jointsB) {
    let crossings = 0;
    // Build segments for chain A
    const segA = [];
    for (let i = 0; i < jointsA.length - 1; i++) {
      segA.push({ p: jointsA[i], q: jointsA[i + 1] });
    }
    const segB = [];
    for (let i = 0; i < jointsB.length - 1; i++) {
      segB.push({ p: jointsB[i], q: jointsB[i + 1] });
    }
    function orient(a, b, c) {
      return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    }
    for (const sa of segA) {
      for (const sb of segB) {
        const o1 = orient(sa.p, sa.q, sb.p);
        const o2 = orient(sa.p, sa.q, sb.q);
        const o3 = orient(sb.p, sb.q, sa.p);
        const o4 = orient(sb.p, sb.q, sa.q);
        if (o1 * o2 < 0 && o3 * o4 < 0) {
          const sign = o1 > 0 ? 1 : -1;
          crossings += sign;
        }
      }
    }
    return crossings / 2;
  }

  /**
   * Compute a simple feature vector describing the topological relationship
   * between two chains. The current implementation returns two values:
   *  - linking number on the XY projection (integer or half integer)
   *  - Euclidean distance between the chains' centroids
   *
   * @param {Chain|{joints:Vec3[]}} cA First chain-like object
   * @param {Chain|{joints:Vec3[]}} cB Second chain-like object
   * @returns {number[]} Feature vector [link, centreDistance]
   */
  function computeFeaturesForChains(cA, cB) {
    const jointsA = cA.joints;
    const jointsB = cB.joints;
    const link = computeLinkingBetween(jointsA, jointsB);
    const centroidA = centroidVec(jointsA);
    const centroidB = centroidVec(jointsB);
    const diff = centroidA.clone().sub(centroidB);
    const centreDist = diff.length();
    return [link, centreDist];
  }

  /**
   * Compute and store feature vectors for all loaded states. Should be
   * invoked once after states are populated.
   */
  function computeStateFeatures() {
    states.forEach((st) => {
      const jA = st.chain1.map((p) => new Vec3(p.x, p.y, p.z));
      const jB = st.chain2.map((p) => new Vec3(p.x, p.y, p.z));
      st.features = computeFeaturesForChains({ joints: jA }, { joints: jB });
    });
  }

  /**
   * Find the index of the state whose feature vector is closest to the given
   * features. Uses simple Euclidean distance in the feature space.
   * @param {number[]} feat Feature vector [link, centreDist]
   * @returns {{idx:number, dist:number}}
   */
  function findNearestState(feat) {
    let bestDist = Infinity;
    let bestIdx = -1;
    states.forEach((st, idx) => {
      if (!st.features) return;
      const f = st.features;
      const d0 = f[0] - feat[0];
      const d1 = f[1] - feat[1];
      const dist = Math.sqrt(d0 * d0 + d1 * d1);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = idx;
      }
    });
    return { idx: bestIdx, dist: bestDist };
  }

  /**
   * Simple collision detection between the two current chains. Returns true
   * if any joint pair comes within a small threshold distance. The threshold
   * is tuned heuristically and may need adjustment depending on the scale
   * of the environment.
   * @returns {boolean}
   */
  function checkCollision() {
    const threshold = 0.25;
    for (const p of chain1.joints) {
      for (const q of chain2.joints) {
        const dx = p.x - q.x;
        const dy = p.y - q.y;
        const dz = p.z - q.z;
        const distSq = dx * dx + dy * dy + dz * dz;
        if (distSq < threshold * threshold) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Attempt to load the GrappleMap database from GitHub. The database is
   * a plain text file where each entry describes a grappling position and
   * includes encoded animation data. For the purpose of this demo we
   * only extract the entry names and reuse placeholder joint positions.
   */
  async function loadStates() {
    try {
      /*
       * Use precomputed states if available.  The `states.js` file
       * defines a global PRECOMPUTED_STATES variable containing an
       * array of pose definitions.  Loading these precomputed values
       * avoids CORS issues and expensive parsing on the client.
       */
      if (Array.isArray(window.PRECOMPUTED_STATES)) {
        window.PRECOMPUTED_STATES.forEach((st) => {
          states.push({
            name: st.name,
            chain1: st.chain1,
            chain2: st.chain2,
            skeleton1: st.skeleton1,
            skeleton2: st.skeleton2,
            features: st.features
          });
        });
      } else {
        throw new Error('PRECOMPUTED_STATES not found');
      }
    } catch (errPre) {
      console.warn('Precomputed states unavailable, attempting to fetch raw database:', errPre);
      try {
        const url = 'https://raw.githubusercontent.com/Eelis/GrappleMap/master/GrappleMap.txt';
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        const parsed = parseGrappleMap(text, 300);
        states.push(...parsed);
      } catch (err) {
        console.error('Failed to load GrappleMap database:', err);
      }
    }
    // Precompute feature vectors for any states lacking them
    computeStateFeatures();
    // Populate the drop‑down options
    updateStateOptions();
    // Apply the first available state if present
    const sel = document.getElementById('state-select');
    if (states.length > 0 && sel && sel.options.length > 0) {
      const firstIdx = parseInt(sel.options[0].value);
      if (!Number.isNaN(firstIdx)) {
        applyState(firstIdx);
      }
    }
  }

  /**
   * Populate the state selection drop‑down with the names of all loaded states.
   */
  // Update the options displayed in the state <select> based on a search
  // filter. An empty filter shows all states. This function is invoked
  // whenever the search input changes or after the states have been loaded.
  function updateStateOptions(filter = '') {
    const sel = document.getElementById('state-select');
    if (!sel) return;
    sel.innerHTML = '';
    const norm = filter.toLowerCase();
    states.forEach((st, idx) => {
      if (!norm || st.name.toLowerCase().includes(norm)) {
        const opt = document.createElement('option');
        opt.value = idx;
        // Replace embedded newlines in the name with slashes for readability
        opt.textContent = st.name.replace(/\n/g, ' / ');
        sel.appendChild(opt);
      }
    });
  }

  /**
   * Apply the joint positions of a given state to the current chains. This
   * function recreates the joint arrays and recomputes the segment lengths.
   *
   * @param {number} idx Index of the state in the states array.
   */
  function applyState(idx) {
    const st = states[idx];
    if (!st) return;
    // Copy positions into chain1
    chain1.joints = st.chain1.map((p) => new Vec3(p.x, p.y, p.z));
    chain1.lengths = [];
    for (let i = 0; i < chain1.joints.length - 1; i++) {
      const diff = chain1.joints[i + 1].clone().sub(chain1.joints[i]);
      chain1.lengths.push(diff.length());
    }
    // Copy positions into chain2
    chain2.joints = st.chain2.map((p) => new Vec3(p.x, p.y, p.z));
    chain2.lengths = [];
    for (let i = 0; i < chain2.joints.length - 1; i++) {
      const diff = chain2.joints[i + 1].clone().sub(chain2.joints[i]);
      chain2.lengths.push(diff.length());
    }
    currentStateIndex = idx;
    // Update dropdown selection and nearest state display if DOM elements exist
    const sel = document.getElementById('state-select');
    if (sel) {
      sel.value = idx;
    }
    const stateEl = document.getElementById('nearest-state');
    if (stateEl) {
      stateEl.textContent = st.name.replace(/\n/g, ' / ');
    }

    // If the state includes precomputed skeletons (from GrappleMap),
    // convert them into our extended 18‑point representation by
    // inserting ankle and wrist joints between existing points.  The
    // original precomputed skeletons contain 14 points in the order
    // documented in deriveSkeletonPoints() prior to our extension:
    //   0:leftFoot,1:leftKnee,2:leftHip,3:core,4:leftShoulder,
    //   5:leftElbow,6:leftHand,7:head,8:rightHand,9:rightElbow,
    //   10:rightShoulder,11:rightHip,12:rightKnee,13:rightFoot.
    // We compute leftAnkle midway between leftFoot and leftKnee, leftWrist
    // midway between leftElbow and leftHand, and similarly for the right
    // side.  If no precomputed skeleton exists, fall back to deriving
    // skeletons from the simplified chains.
    if (st.skeleton1 && st.skeleton2) {
      function convertSk(sk) {
        // ensure we have at least 14 points
        if (!Array.isArray(sk) || sk.length < 14) {
          return [];
        }
        const leftFoot = sk[0];
        const leftKnee = sk[1];
        const leftHip = sk[2];
        const core = sk[3];
        const leftShoulder = sk[4];
        const leftElbow = sk[5];
        const leftHand = sk[6];
        const head = sk[7];
        const rightHand = sk[8];
        const rightElbow = sk[9];
        const rightShoulder = sk[10];
        const rightHip = sk[11];
        const rightKnee = sk[12];
        const rightFoot = sk[13];
        // helper to midpoint
        const mid = (a,b) => {
          return { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5, z: (a.z + b.z) * 0.5 };
        };
        // Place ankle closer to the foot (70% foot, 30% knee)
        const leftAnkle = {
          x: leftFoot.x * 0.7 + leftKnee.x * 0.3,
          y: leftFoot.y * 0.7 + leftKnee.y * 0.3,
          z: leftFoot.z * 0.7 + leftKnee.z * 0.3
        };
        const rightAnkle = {
          x: rightFoot.x * 0.7 + rightKnee.x * 0.3,
          y: rightFoot.y * 0.7 + rightKnee.y * 0.3,
          z: rightFoot.z * 0.7 + rightKnee.z * 0.3
        };
        // Place wrist closer to the hand (70% hand, 30% elbow)
        const leftWrist = {
          x: leftElbow.x * 0.3 + leftHand.x * 0.7,
          y: leftElbow.y * 0.3 + leftHand.y * 0.7,
          z: leftElbow.z * 0.3 + leftHand.z * 0.7
        };
        const rightWrist = {
          x: rightElbow.x * 0.3 + rightHand.x * 0.7,
          y: rightElbow.y * 0.3 + rightHand.y * 0.7,
          z: rightElbow.z * 0.3 + rightHand.z * 0.7
        };
        // assemble extended skeleton
        return [leftFoot, leftAnkle, leftKnee, leftHip, core,
                leftShoulder, leftElbow, leftWrist, leftHand, head,
                rightHand, rightWrist, rightElbow, rightShoulder,
                rightHip, rightKnee, rightAnkle, rightFoot];
      }
      currentSkeleton1 = convertSk(st.skeleton1).map((p) => new Vec3(p.x, p.y, p.z));
      currentSkeleton2 = convertSk(st.skeleton2).map((p) => new Vec3(p.x, p.y, p.z));
    } else {
      updateSkeletonsFromChains();
    }
  }

  // Event handler for the state selection drop‑down. Wait until the DOM is
  // available before attaching the listener. The script is loaded at the
  // bottom of the document so this code will execute after the DOM is
  // constructed.
  document.addEventListener('DOMContentLoaded', () => {
    const sel = document.getElementById('state-select');
    if (sel) {
      sel.addEventListener('change', (e) => {
        const idx = parseInt(e.target.value);
        if (!Number.isNaN(idx)) {
          applyState(idx);
        }
      });
    }
    // Attach input listener to the search field to filter state options
    const search = document.getElementById('state-search');
    if (search) {
      search.addEventListener('input', (e) => {
        updateStateOptions(e.target.value);
      });
    }
  });

  // Kick off asynchronous loading of the GrappleMap database or fallback
  loadStates();

  // --- Interaction state ---
  let selected = null; // {chain: chain, index: number}
  let rotating = false;
  let lastX = 0;
  let lastY = 0;

  // Prevent context menu on right-click
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  // Picking on mousedown
  canvas.addEventListener('mousedown', (e) => {
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
        // Save previous positions for collision revert
        prevPositions = [
          chain1.joints.map((p) => p.clone()),
          chain2.joints.map((p) => p.clone())
        ];
      }
    } else if (e.button === 2) {
      // Right button: rotate camera
      rotating = true;
    }
  });

  document.addEventListener('mouseup', () => {
    selected = null;
    rotating = false;
    // If we were dragging and recorded positions, perform collision check and
    // revert if necessary. Do not snap to a nearest pose. Update the
    // skeletons from the resulting chains. This lets the user explore the
    // configuration space continuously.
    if (prevPositions) {
      if (checkCollision()) {
        chain1.joints = prevPositions[0].map((p) => p.clone());
        chain1.lengths = [];
        for (let i = 0; i < chain1.joints.length - 1; i++) {
          const diff = chain1.joints[i + 1].clone().sub(chain1.joints[i]);
          chain1.lengths.push(diff.length());
        }
        chain2.joints = prevPositions[1].map((p) => p.clone());
        chain2.lengths = [];
        for (let i = 0; i < chain2.joints.length - 1; i++) {
          const diff = chain2.joints[i + 1].clone().sub(chain2.joints[i]);
          chain2.lengths.push(diff.length());
        }
      }
      // Recompute skeletons from the final chain positions
      updateSkeletonsFromChains();
      prevPositions = null;
    }
  });

  document.addEventListener('mousemove', (e) => {
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

  canvas.addEventListener('wheel', (e) => {
    const delta = e.deltaY;
    camRadius *= Math.exp(delta * 0.001);
    camRadius = Math.max(2, Math.min(50, camRadius));
    e.preventDefault();
  });

  // --- Camera helpers ---
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
    chains.forEach((chain) => {
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
    const scale = depth / (height * 0.8);
    // Compute world delta: right and up directions scaled by screen movement
    const worldDelta = right.clone().multiplyScalar(dx * scale).add(
      up.clone().multiplyScalar(-dy * scale)
    );
    selPt.add(worldDelta);
    // Adjust chain to maintain bone lengths
    selected.chain.adjust(selected.index);

    // Update skeleton representation based on the modified chain positions.
    updateSkeletonsFromChains();
  }

  /**
   * Recompute the approximate skeletons for both avatars based on the current
   * simplified chains. This function mirrors the left‑side joints across the
   * hips to produce right‑side counterparts. It is invoked after the user
   * drags a joint so that the drawn humanoids follow the updated poses.
   */
  function updateSkeletonsFromChains() {
    // Helper to mirror a point across the hip of a given chain
    function mirrorPoint(p, hip) {
      const dx = p.x - hip.x;
      return new Vec3(hip.x - dx, p.y, p.z);
    }
    // Compute skeleton for a single chain (fallback mode).
    function computeSkeletonFromChain(chain) {
      // Each simplified chain has 6 joints: foot, knee, hip, shoulder, hand, head.
      // We derive an 18‑point skeleton by interpolating extra joints along each
      // segment and mirroring the left side across the hips for the right side.
      const footL = chain.joints[0].clone();        // toe
      const kneeL = chain.joints[1].clone();        // knee
      const hip = chain.joints[2].clone();          // hip (and core)
      const core = hip.clone();                     // use hip for core in fallback
      const shoulderL = chain.joints[3].clone();    // shoulder
      const handL = chain.joints[4].clone();        // palm
      const head = chain.joints[5].clone();         // head
      // approximate elbow halfway between shoulder and hand
      const elbowL = chain.joints[3].clone().multiplyScalar(0.5).add(chain.joints[4].clone().multiplyScalar(0.5));
      // approximate wrist roughly two‑thirds from elbow to hand
      const wristL = elbowL.clone().multiplyScalar(0.3).add(handL.clone().multiplyScalar(0.7));
      // approximate ankle roughly one‑third from foot to knee
      const ankleL = footL.clone().multiplyScalar(0.7).add(kneeL.clone().multiplyScalar(0.3));
      // Mirror helper for right side
      const mirror = (p) => {
        const dx = p.x - hip.x;
        return new Vec3(hip.x - dx, p.y, p.z);
      };
      // Mirror left side joints to obtain right side
      const handR = mirror(handL);
      const wristR = mirror(wristL);
      const elbowR = mirror(elbowL);
      const shoulderR = mirror(shoulderL);
      const hipR = mirror(hip);
      const kneeR = mirror(kneeL);
      const ankleR = mirror(ankleL);
      const footR = mirror(footL);
      return [footL, ankleL, kneeL, hip, core, shoulderL, elbowL, wristL, handL, head,
              handR, wristR, elbowR, shoulderR, hipR, kneeR, ankleR, footR];
    }
    currentSkeleton1 = computeSkeletonFromChain(chain1);
    currentSkeleton2 = computeSkeletonFromChain(chain2);
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
    for (const segA of s1) {
      for (const segB of s2) {
        const a1 = segA.p;
        const a2 = segA.q;
        const b1 = segB.p;
        const b2 = segB.q;
        const o1 = orient(a1, a2, b1);
        const o2 = orient(a1, a2, b2);
        const o3 = orient(b1, b2, a1);
        const o4 = orient(b1, b2, a2);
        if (o1 * o2 < 0 && o3 * o4 < 0) {
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
    // Draw humanoid skeletons for both avatars. Each skeleton is defined by
    // 10 points in currentSkeleton1/currentSkeleton2 and a set of edges
    // connecting them. Line thickness and joint size scale with depth to
    // convey perspective.
    const drawSkeleton = (skeleton, color) => {
      if (!skeleton || skeleton.length === 0) return;
      // Define edges connecting the skeleton joints (18‑point layout).
      // Indices: 0: left toe, 1: left ankle, 2: left knee, 3: left hip,
      // 4: core, 5: left shoulder, 6: left elbow, 7: left wrist,
      // 8: left hand, 9: head, 10: right hand, 11: right wrist,
      // 12: right elbow, 13: right shoulder, 14: right hip, 15: right knee,
      // 16: right ankle, 17: right toe.
      // We connect the toes up to the hips, hips to core, arms out from
      // the shoulders, wrists to hands, and the head connects to both
      // shoulders.
      const edges = [
        [0, 1],   // left toe to left ankle
        [1, 2],   // left ankle to left knee
        [2, 3],   // left knee to left hip
        [3, 4],   // left hip to core
        [14, 4],  // right hip to core
        [14, 15], // right hip to right knee
        [15, 16], // right knee to right ankle
        [16, 17], // right ankle to right toe
        [4, 5],   // core to left shoulder
        [5, 6],   // left shoulder to left elbow
        [6, 7],   // left elbow to left wrist
        [7, 8],   // left wrist to left hand
        [4, 13],  // core to right shoulder
        [13, 12], // right shoulder to right elbow
        [12, 11], // right elbow to right wrist
        [11, 10], // right wrist to right hand
        [5, 9],   // left shoulder to head
        [13, 9]   // right shoulder to head
      ];
      // Project all points once
      const projections = skeleton.map((pt) => projectPoint(pt, basis));
      // Draw bones
      ctx.strokeStyle = color;
      edges.forEach(([a, b]) => {
        const p = projections[a];
        const q = projections[b];
        if (!p || !q) return;
        // Use average depth to determine line width (closer = thicker)
        const depth = (p.z + q.z) * 0.5;
        const lineW = Math.max(1, 8 / depth);
        ctx.lineWidth = lineW;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(q.x, q.y);
        ctx.stroke();
      });
      // Draw joints as filled circles. Radius inversely proportional to depth.
      ctx.fillStyle = color;
      projections.forEach((proj) => {
        if (!proj) return;
        const r = Math.max(2, 10 / proj.z);
        ctx.beginPath();
        ctx.arc(proj.x, proj.y, r, 0, Math.PI * 2);
        ctx.fill();
      });
    };
    drawSkeleton(currentSkeleton1, '#e74c3c');
    drawSkeleton(currentSkeleton2, '#3498db');
    // Display linking number
    const linkValueEl = document.getElementById('link-value');
    if (linkValueEl) {
      linkValueEl.textContent = computeLinking().toFixed(1);
    }
    // Update nearest state display based on current configuration
    const stateEl = document.getElementById('nearest-state');
    if (stateEl && states.length > 0) {
      const feat = computeFeaturesForChains(chain1, chain2);
      const { idx: nearestIdx } = findNearestState(feat);
      if (nearestIdx !== -1) {
        stateEl.textContent = states[nearestIdx].name;
      }
    }
    requestAnimationFrame(render);
  }
  render();
})();
