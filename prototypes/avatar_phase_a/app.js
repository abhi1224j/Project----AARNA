/**
 * AARNA — Phase A v2: Redesigned 3D Character & Lively Animation Prototype
 * Built with Three.js + @pixiv/three-vrm (100% Local Offline ES Modules)
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

// ---------------------------------------------------------------------------
// 1. Configuration & Model Registry
// ---------------------------------------------------------------------------
const MODEL_CATALOG = {
  aarna_glb: {
    id: 'aarna_glb',
    name: 'AARNA v2: Redesigned Teen & Baby Pink Frock (GLB)',
    url: './models/aarna_teen_casual.glb?v=babypink_v2_final',
    format: 'GLB v2 (Baby Pink Frock + Hair Physics)',
    description:
      'Redesigned 15–16 yr feminine teenage face • Soft cheeks, V-taper jaw, large expressive eyes • Modest Baby Pink floral frock, rose-pink satin bow & Mary Jane shoes.',
  },
  aarna_v1_backup: {
    id: 'aarna_v1_backup',
    name: 'AARNA v1 Backup (Sweatshirt & Jeans Comparison)',
    url: './models/aarna_v1_casual_backup.glb',
    format: 'GLB v1 (Backup)',
    description:
      'Preserved v1 prototype character (unsculpted base face + casual sweatshirt & denim jeans) for before/after comparison and rollback.',
  },
  vrm1_sample: {
    id: 'vrm1_sample',
    name: 'VRM 1.0 Official Sample (VRM1_Constraint_Twist_Sample)',
    url: './models/VRM1_Constraint_Twist_Sample.vrm',
    format: 'VRM 1.0',
    description:
      'Official pixiv/three-vrm VRM 1.0 sample • 54 humanoid bones • Twist constraints & hair spring bones • 18 preset expressions.',
  },
  vrm0_sample: {
    id: 'vrm0_sample',
    name: 'VRM 0.x Official Sample (AvatarSample_A)',
    url: './models/AvatarSample_A.vrm',
    format: 'VRM 0.x',
    description:
      'Official VRoid Studio Sample A (CC0) • VRM 0.x backward-compatibility verification • 54 humanoid bones & spring physics.',
  },
};

const OUTFIT_SWATCHES = {
  babypink: { name: 'Baby Pink Frock', hex: 0xffffff, vrmHex: 0xffd4e5 },
  lavender: { name: 'Pastel Lavender Frock', hex: 0xdec9f8, vrmHex: 0xd8ccf8 },
  peach: { name: 'Soft Peach Frock', hex: 0xffdcc2, vrmHex: 0xffe0c8 },
  skyblue: { name: 'Pastel Sky Blue Frock', hex: 0xc6e6ff, vrmHex: 0xbce0fd },
  cream: { name: 'Warm Cream White', hex: 0xfff8ee, vrmHex: 0xf7f4eb },
  mint: { name: 'Soft Pastel Mint', hex: 0xd8faea, vrmHex: 0xc4f0dc },
};

// ---------------------------------------------------------------------------
// 2. Application State
// ---------------------------------------------------------------------------
const state = {
  currentModelKey: 'aarna_glb',
  assistantState: 'idle', // 'idle' | 'listening'
  activeExpression: 'cheerful',
  activeGesture: 'idle', // 'idle' | 'wave' | 'tilt' | 'nod' | 'hands' | 'speaking'
  gestureStartTime: 0,
  visemeDemoActive: false,
  visemeDemoStart: 0,
  audioLevel: 0,
  smoothAudioLevel: 0,
  autoGaze: true,
  manualPose: {
    yaw: 0,
    pitch: 0,
    roll: 0,
    torso: 0,
  },
  outfitKey: 'babypink',
  cameraPreset: 'full',
  targetFps: 60,
  rendererPaused: false,
  autoPausedByVisibility: false,
  characterHeight: 1.65,
  // Blinking state (varied human intervals + occasional double blink)
  nextBlinkTime: 1.8,
  blinkStartTime: -1,
  isBlinking: false,
  pendingDoubleBlink: false,
  // Saccade eye gaze state
  nextSaccadeTime: 1.2,
  eyeTargetX: 0,
  eyeTargetY: 0,
  eyeCurrentX: 0,
  eyeCurrentY: 0,
  // Non-periodic organic idle head drift state (2nd-order critically damped cervical motion)
  nextHeadDriftTime: 2.4,
  headDriftTargetYaw: 0.025,
  headDriftTargetPitch: 0.01,
  headDriftTargetRoll: 0.02,
  headDriftYaw: 0,
  headDriftPitch: 0,
  headDriftRoll: 0,
  headDriftVelYaw: 0,
  headDriftVelPitch: 0,
  headDriftVelRoll: 0,
  // Smoothed gesture pose state (prevents pops/snaps when transitioning between gestures)
  curRightArmLift: 0,
  curRightForeArmBend: 0,
  curRightWristWave: 0,
  curLeftArmLift: 0,
  curLeftForeArmBend: 0,
  curLeftWristTurn: 0,
  curLeftHandOpen: 0,
  curRightHandOpen: 0,
  curHeadGestureYaw: 0,
  curHeadGesturePitch: 0,
  curHeadGestureRoll: 0,
  curTorsoGesturePitch: 0,
  // 2nd-order spring-damper hair physics state (drives HeadTop_End Joint 6)
  prevHeadYaw: 0,
  prevHeadPitch: 0,
  prevHeadRoll: 0,
  hairSpringX: 0,
  hairSpringY: 0,
  hairSpringZ: 0,
  hairVelX: 0,
  hairVelY: 0,
  hairVelZ: 0,
  // Pointer tracking
  pointerX: 0,
  pointerY: 0,
  // Telemetry
  fps: 60,
  frameTimeMs: 16.6,
  frameCount: 0,
  fpsAccumTime: 0,
  lastFrameTimestamp: performance.now(),
};

// ---------------------------------------------------------------------------
// 3. Scene, Camera, Renderer & Soft Portrait Lighting
// ---------------------------------------------------------------------------
const canvas = document.getElementById('avatar-canvas');
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false,
  powerPreference: 'high-performance',
  preserveDrawingBuffer: true,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.06;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x040711);
scene.fog = new THREE.FogExp2(0x040711, 0.062);

const camera = new THREE.PerspectiveCamera(
  30,
  window.innerWidth / window.innerHeight,
  0.1,
  100
);
camera.position.set(0, 0.85, 3.85);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.55;
controls.maxDistance = 6.5;
controls.maxPolarAngle = Math.PI * 0.52;
controls.target.set(0, 0.82, 0);
controls.update();

// Soft, flattering 5-point portrait & space lighting around AARNA (true-color warm neutral balance)
const hemiLight = new THREE.HemisphereLight(0xfff6f4, 0x221b1d, 1.25);
hemiLight.position.set(0, 4, 0);
scene.add(hemiLight);

const keyLight = new THREE.DirectionalLight(0xfff8f2, 1.65);
keyLight.position.set(1.25, 2.35, 2.75);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(1024, 1024);
keyLight.shadow.bias = -0.0005;
scene.add(keyLight);

// Soft warm-ivory fill light from the left (zero blue/lavender color cast on Baby Pink fabric)
const fillLight = new THREE.DirectionalLight(0xfff5f0, 1.05);
fillLight.position.set(-1.65, 1.75, 2.1);
scene.add(fillLight);

// Gentle facial beauty bounce light near camera level for bright eye catchlights & soft skin shading
const beautyLight = new THREE.PointLight(0xfff5f5, 0.55, 3.5);
beautyLight.position.set(0.0, 1.58, 1.45);
scene.add(beautyLight);

// Champagne-gold rim light for holographic space atmosphere
const goldRimLight = new THREE.SpotLight(0xf3d078, 2.35, 8, Math.PI * 0.38, 0.6, 1.2);
goldRimLight.position.set(0.2, 2.3, -1.9);
goldRimLight.target.position.set(0, 1.0, 0);
scene.add(goldRimLight);
scene.add(goldRimLight.target);

// Warm amber ground bounce light
const groundGlowLight = new THREE.PointLight(0xe5a93c, 0.62, 2.5);
groundGlowLight.position.set(0, 0.15, 0.4);
scene.add(groundGlowLight);

// ---------------------------------------------------------------------------
// 4. Space Starfield, Champagne-Gold Particles & Pedestal
// ---------------------------------------------------------------------------
const environmentGroup = new THREE.Group();
scene.add(environmentGroup);

function createStarfield() {
  const starCount = 650;
  const positions = new Float32Array(starCount * 3);
  const colors = new Float32Array(starCount * 3);
  const palette = [
    new THREE.Color(0xf8fafc),
    new THREE.Color(0xf3d078),
    new THREE.Color(0xbfdbfe),
    new THREE.Color(0xfde68a),
  ];

  for (let i = 0; i < starCount; i++) {
    const r = 8 + Math.random() * 24;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = Math.abs(r * Math.cos(phi)) - 1.5;
    positions[i * 3 + 2] = -Math.abs(r * Math.sin(phi) * Math.sin(theta)) - 2.0;

    const c = palette[i % palette.length];
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 0.045,
    vertexColors: true,
    transparent: true,
    opacity: 0.8,
    depthWrite: false,
  });

  const stars = new THREE.Points(geo, mat);
  environmentGroup.add(stars);
  return stars;
}

function createGoldenParticles() {
  const count = 110;
  const positions = new Float32Array(count * 3);
  const phases = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const angle = Math.random() * Math.PI * 2;
    const radius = 0.55 + Math.random() * 1.85;
    const px = Math.cos(angle) * radius;
    const pz = Math.sin(angle) * radius - 0.2;
    positions[i * 3] = px;
    positions[i * 3 + 1] = Math.random() * 2.2;
    positions[i * 3 + 2] = Math.abs(px) < 0.45 && pz > -0.25 ? -Math.abs(pz) - 0.35 : pz;
    phases[i] = Math.random() * Math.PI * 2;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  const spriteCanvas = document.createElement('canvas');
  spriteCanvas.width = 64;
  spriteCanvas.height = 64;
  const ctx = spriteCanvas.getContext('2d');
  const grad = ctx.createRadialGradient(32, 32, 2, 32, 32, 30);
  grad.addColorStop(0, 'rgba(255, 243, 199, 1.0)');
  grad.addColorStop(0.4, 'rgba(243, 208, 120, 0.65)');
  grad.addColorStop(1, 'rgba(229, 169, 60, 0.0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 64, 64);
  const spriteTex = new THREE.CanvasTexture(spriteCanvas);

  const mat = new THREE.PointsMaterial({
    size: 0.065,
    map: spriteTex,
    color: 0xf3d078,
    transparent: true,
    opacity: 0.72,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  const points = new THREE.Points(geo, mat);
  environmentGroup.add(points);
  return { points, phases, count };
}

function createPedestal() {
  const group = new THREE.Group();
  group.position.set(0, 0.002, 0);

  // 1. Core cybernetic rings (neon cyan & electric arc blue)
  const innerGeo = new THREE.RingGeometry(0.42, 0.44, 64);
  innerGeo.rotateX(-Math.PI / 2);
  const innerMat = new THREE.MeshBasicMaterial({
    color: 0x00f3ff,
    transparent: true,
    opacity: 0.65,
    side: THREE.DoubleSide,
  });
  const innerRing = new THREE.Mesh(innerGeo, innerMat);
  group.add(innerRing);

  const outerGeo = new THREE.RingGeometry(0.58, 0.596, 64);
  outerGeo.rotateX(-Math.PI / 2);
  const outerMat = new THREE.MeshBasicMaterial({
    color: 0x0284c7,
    transparent: true,
    opacity: 0.45,
    side: THREE.DoubleSide,
  });
  const outerRing = new THREE.Mesh(outerGeo, outerMat);
  group.add(outerRing);

  const discGeo = new THREE.CircleGeometry(0.58, 64);
  discGeo.rotateX(-Math.PI / 2);
  const discMat = new THREE.MeshBasicMaterial({
    color: 0x082f49,
    transparent: true,
    opacity: 0.18,
  });
  const disc = new THREE.Mesh(discGeo, discMat);
  disc.position.y = -0.001;
  group.add(disc);

  // 2. Cybernetic Projection Pedestal Grid
  const gridGroup = new THREE.Group();
  const gridRadii = [0.18, 0.30, 0.42, 0.54, 0.68, 0.82];
  const gridRings = [];
  gridRadii.forEach((r, idx) => {
    const rGeo = new THREE.RingGeometry(r - 0.003, r + 0.003, 64);
    rGeo.rotateX(-Math.PI / 2);
    const rMat = new THREE.MeshBasicMaterial({
      color: idx % 2 === 0 ? 0x00f3ff : 0x38bdf8,
      transparent: true,
      opacity: 0.28,
      side: THREE.DoubleSide,
    });
    const rMesh = new THREE.Mesh(rGeo, rMat);
    gridRings.push(rMesh);
    gridGroup.add(rMesh);
  });

  // Radial grid spokes (16 radial tick lines)
  const spokeCount = 16;
  const spokePositions = [];
  for (let i = 0; i < spokeCount; i++) {
    const angle = (i / spokeCount) * Math.PI * 2;
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    spokePositions.push(cosA * 0.12, 0.001, sinA * 0.12);
    spokePositions.push(cosA * 0.82, 0.001, sinA * 0.82);
  }
  const spokeGeo = new THREE.BufferGeometry();
  spokeGeo.setAttribute('position', new THREE.Float32BufferAttribute(spokePositions, 3));
  const spokeMat = new THREE.LineBasicMaterial({
    color: 0x00f3ff,
    transparent: true,
    opacity: 0.22,
  });
  const spokeLines = new THREE.LineSegments(spokeGeo, spokeMat);
  gridGroup.add(spokeLines);
  group.add(gridGroup);

  // 3. Sound-Reactive Neon Wave Ripple Emitters
  const rippleCount = 4;
  const rippleRings = [];
  for (let i = 0; i < rippleCount; i++) {
    const ripGeo = new THREE.RingGeometry(0.04, 0.07, 64);
    ripGeo.rotateX(-Math.PI / 2);
    const ripMat = new THREE.MeshBasicMaterial({
      color: 0x00f3ff,
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const ripMesh = new THREE.Mesh(ripGeo, ripMat);
    ripMesh.position.y = 0.003;
    ripMesh.userData.phaseOffset = i / rippleCount;
    rippleRings.push(ripMesh);
    group.add(ripMesh);
  }

  environmentGroup.add(group);
  return { group, innerMat, outerMat, discMat, gridGroup, gridRings, rippleRings };
}

const starfield = createStarfield();
const goldenParticles = createGoldenParticles();
const pedestal = createPedestal();

// ---------------------------------------------------------------------------
// 5. Character Loader & Rig Abstraction (GLB + VRM 1.0 + VRM 0.x)
// ---------------------------------------------------------------------------
let currentAvatarRoot = null;
let currentVrm = null;
let glbBones = {};
let glbRestQuats = {};
let glbRestPos = {};
let glbMorphMeshes = [];
let outfitMaterials = [];
let chestBadgeMesh = null;
let chestBadgeMat = null;
let holoWireframeMeshes = [];
const holoUniforms = {
  uHoloTime: { value: 0 },
  uAudioLevel: { value: 0 },
};

function createChestScriptBadge() {
  const badgeCanvas = document.createElement('canvas');
  badgeCanvas.width = 512;
  badgeCanvas.height = 160;
  const ctx = badgeCanvas.getContext('2d');
  ctx.clearRect(0, 0, 512, 160);

  // Glowing script text "AARNA"
  ctx.font = 'bold italic 72px "Brush Script MT", "Segoe Script", "Lucida Handwriting", cursive, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  // Layered glowing cyan and white glow
  ctx.shadowColor = '#00f3ff';
  ctx.shadowBlur = 30;
  ctx.fillStyle = '#00f3ff';
  ctx.fillText('AARNA', 256, 80);

  ctx.shadowBlur = 12;
  ctx.fillStyle = '#e0faff';
  ctx.fillText('AARNA', 256, 80);

  const tex = new THREE.CanvasTexture(badgeCanvas);
  tex.colorSpace = THREE.SRGBColorSpace;

  const geo = new THREE.PlaneGeometry(0.20, 0.068);
  const mat = new THREE.MeshBasicMaterial({
    map: tex,
    transparent: true,
    opacity: 0.95,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.name = 'Cybernetic_Chest_Badge_AARNA';
  return { mesh, mat };
}

const gltfLoader = new GLTFLoader();
gltfLoader.register((parser) => new VRMLoaderPlugin(parser));

function disposeCurrentAvatar() {
  if (!currentAvatarRoot) return;
  scene.remove(currentAvatarRoot);
  currentAvatarRoot.traverse((obj) => {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      mats.forEach((m) => m && m.dispose && m.dispose());
    }
  });
  if (chestBadgeMesh && chestBadgeMesh.parent) {
    chestBadgeMesh.parent.remove(chestBadgeMesh);
    if (chestBadgeMesh.geometry) chestBadgeMesh.geometry.dispose();
  }
  if (chestBadgeMat) {
    if (chestBadgeMat.map) chestBadgeMat.map.dispose();
    chestBadgeMat.dispose();
  }
  chestBadgeMesh = null;
  chestBadgeMat = null;
  holoWireframeMeshes.forEach((wm) => {
    if (wm.parent) wm.parent.remove(wm);
    if (wm.material) wm.material.dispose();
  });
  holoWireframeMeshes = [];
  currentAvatarRoot = null;
  currentVrm = null;
  glbBones = {};
  glbRestQuats = {};
  glbRestPos = {};
  glbMorphMeshes = [];
  outfitMaterials = [];
}

async function loadCharacterModel(modelKey) {
  const entry = MODEL_CATALOG[modelKey] || MODEL_CATALOG.aarna_glb;
  state.currentModelKey = entry.id;

  const overlay = document.getElementById('loading-overlay');
  const loadingText = document.getElementById('loading-text');
  const statusMsg = document.getElementById('status-message');
  const infoNote = document.getElementById('model-info-note');
  const formatHud = document.getElementById('hud-format');

  overlay.classList.remove('hidden');
  loadingText.textContent = `Loading ${entry.name}...`;
  statusMsg.textContent = `Loading ${entry.name}...`;
  if (infoNote) infoNote.textContent = entry.description;
  if (formatHud) formatHud.textContent = entry.format;

  disposeCurrentAvatar();

  return new Promise((resolve, reject) => {
    gltfLoader.load(
      entry.url,
      (gltf) => {
        const vrm = gltf.userData.vrm || null;
        currentVrm = vrm;
        const root = vrm ? vrm.scene : gltf.scene;
        currentAvatarRoot = root;

        if (vrm) {
          VRMUtils.removeUnnecessaryVertices(root);
          if (vrm.meta && vrm.meta.metaVersion === '0') {
            VRMUtils.rotateVRM0(vrm);
          }
          setupVrmInitialPose(vrm);
        } else {
          setupGlbAvatar(root);
        }

        // Collect frock bodice, skirt, and matching shoe materials for pastel colour customization
        root.traverse((obj) => {
          if (obj.isMesh) {
            obj.castShadow = true;
            obj.receiveShadow = true;
            obj.frustumCulled = false;
            const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
            mats.forEach((m) => {
              if (!m) return;
              const n = (m.name || '').toLowerCase();
              if (
                n.includes('outfit_top') ||
                n.includes('outfit_bottom') ||
                n.includes('outfit_footwear') ||
                n.includes('tops_01')
              ) {
                outfitMaterials.push(m);
              }
            });
          }
        });

        scene.add(root);
        root.updateMatrixWorld(true);

        // Safely measure height and ground feet at y = 0.0
        let minY = Infinity;
        let maxY = -Infinity;
        root.traverse((obj) => {
          if (!obj.visible) return;
          if (obj.isBone || obj.type === 'Bone') {
            const wp = new THREE.Vector3();
            obj.getWorldPosition(wp);
            if (Number.isFinite(wp.y)) {
              minY = Math.min(minY, wp.y);
              maxY = Math.max(maxY, wp.y);
            }
          } else if (obj.isMesh && obj.geometry) {
            obj.geometry.computeBoundingBox();
            const bb = obj.geometry.boundingBox;
            if (bb && Number.isFinite(bb.min.y) && Number.isFinite(bb.max.y)) {
              const s = root.scale.y || 1;
              minY = Math.min(minY, bb.min.y * s);
              maxY = Math.max(maxY, bb.max.y * s);
            }
          }
        });

        if (!Number.isFinite(minY) || !Number.isFinite(maxY) || maxY <= minY) {
          minY = 0;
          maxY = 1.65;
        }
        if (minY > 0.04) minY = 0;
        root.position.y = -minY;
        root.updateMatrixWorld(true);

        state.characterHeight = Math.max(1.35, Math.min(1.85, maxY - minY));

        applyOutfitSwatch(state.outfitKey);
        applyCameraPreset(state.cameraPreset);
        updateGeometryMetrics();

        // Render an immediate frame so canvas & renderer.info are populated immediately
        renderer.render(scene, camera);

        overlay.classList.add('hidden');
        statusMsg.textContent = `${entry.name} ready (Full body + feet visible)`;
        resolve(entry);
      },
      undefined,
      (err) => {
        console.error('Error loading character model:', err);
        loadingText.textContent = `Error loading ${entry.name}`;
        statusMsg.textContent = 'Model load error';
        reject(err);
      }
    );
  });
}

function setupGlbAvatar(root) {
  // Youthful ~15-16 yr teenage proportions
  root.scale.set(0.95, 0.95, 0.95);

  root.traverse((node) => {
    if (node.name === 'Wolf3D_Glasses') {
      node.visible = false;
    }
    if (node.isBone || node.type === 'Bone' || node.name) {
      glbBones[node.name] = node;
      glbRestQuats[node.name] = node.quaternion.clone();
      glbRestPos[node.name] = node.position.clone();
    }
    if (node.isMesh && node.morphTargetDictionary && node.morphTargetInfluences) {
      glbMorphMeshes.push(node);
    }
  });

  if (glbBones.Head) {
    glbBones.Head.scale.set(1.045, 1.045, 1.045);
  }

  // 1. Upgrade materials to Cybernetic Holographic Female Mesh with wireframe and scanline shaders
  root.traverse((node) => {
    if (node.isMesh) {
      const mats = Array.isArray(node.material) ? node.material : [node.material];
      mats.forEach((m) => {
        if (!m) return;
        m.transparent = true;
        m.opacity = 0.88;
        // Inject cybernetic holographic shader hooks (Fresnel rim glow + scanlines)
        const prevCompile = m.onBeforeCompile;
        m.onBeforeCompile = (shader) => {
          if (prevCompile) prevCompile(shader);
          shader.uniforms.uHoloTime = holoUniforms.uHoloTime;
          shader.uniforms.uAudioLevel = holoUniforms.uAudioLevel;

          shader.vertexShader = shader.vertexShader.replace(
            '#include <common>',
            `#include <common>
             varying vec3 vWorldPosHolo;
             varying vec3 vNormalHolo;`
          );
          shader.vertexShader = shader.vertexShader.replace(
            '#include <worldpos_vertex>',
            `#include <worldpos_vertex>
             vWorldPosHolo = (modelMatrix * vec4(transformed, 1.0)).xyz;
             vNormalHolo = normalize(normalMatrix * normal);`
          );

          shader.fragmentShader = shader.fragmentShader.replace(
            '#include <common>',
            `#include <common>
             uniform float uHoloTime;
             uniform float uAudioLevel;
             varying vec3 vWorldPosHolo;
             varying vec3 vNormalHolo;`
          );
          shader.fragmentShader = shader.fragmentShader.replace(
            '#include <dithering_fragment>',
            `#include <dithering_fragment>
             vec3 holoViewDir = normalize(cameraPosition - vWorldPosHolo);
             float holoFresnel = pow(1.0 - max(0.0, dot(vNormalHolo, holoViewDir)), 2.6);
             vec3 holoGlow = vec3(0.0, 0.95, 1.0) * holoFresnel * (0.75 + 0.25 * sin(uHoloTime * 3.0) + uAudioLevel * 0.45);
             float holoScan = sin(vWorldPosHolo.y * 140.0 - uHoloTime * 3.5) * 0.08;
             gl_FragColor.rgb += holoGlow + vec3(0.0, 0.8, 1.0) * holoScan;
             gl_FragColor.a = min(gl_FragColor.a, 0.88);`
          );
        };
      });

      // 2. Holographic Cybernetic Wireframe Mesh Overlays
      const n = (node.name || '').toLowerCase();
      if (node.isSkinnedMesh && (n.includes('body') || n.includes('top') || n.includes('bottom') || n.includes('hair'))) {
        const wireMat = new THREE.MeshBasicMaterial({
          color: 0x00f3ff,
          wireframe: true,
          transparent: true,
          opacity: 0.16,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        });
        const wireMesh = new THREE.SkinnedMesh(node.geometry, wireMat);
        wireMesh.bind(node.skeleton, node.bindMatrix);
        wireMesh.name = node.name + '_HoloWireframe';
        root.add(wireMesh);
        holoWireframeMeshes.push(wireMesh);
      }
    }
  });

  // 3. Bright neon-blue glowing script text "AARNA" centered directly onto the chest
  const badge = createChestScriptBadge();
  chestBadgeMesh = badge.mesh;
  chestBadgeMat = badge.mat;
  if (glbBones.Spine2) {
    chestBadgeMesh.position.set(0, 0.13, 0.135);
    chestBadgeMesh.rotation.set(-0.06, 0, 0);
    glbBones.Spine2.add(chestBadgeMesh);
  } else if (glbBones.Spine1) {
    chestBadgeMesh.position.set(0, 0.22, 0.135);
    glbBones.Spine1.add(chestBadgeMesh);
  } else {
    chestBadgeMesh.position.set(0, 1.16, 0.14);
    root.add(chestBadgeMesh);
  }
}

function setupVrmInitialPose(vrm) {
  if (!vrm.humanoid) return;
  const leftUpperArm = vrm.humanoid.getNormalizedBoneNode('leftUpperArm');
  const rightUpperArm = vrm.humanoid.getNormalizedBoneNode('rightUpperArm');
  const leftLowerArm = vrm.humanoid.getNormalizedBoneNode('leftLowerArm');
  const rightLowerArm = vrm.humanoid.getNormalizedBoneNode('rightLowerArm');
  const leftHand = vrm.humanoid.getNormalizedBoneNode('leftHand');
  const rightHand = vrm.humanoid.getNormalizedBoneNode('rightHand');

  if (leftUpperArm) leftUpperArm.rotation.z = -1.16;
  if (rightUpperArm) rightUpperArm.rotation.z = 1.16;
  if (leftLowerArm) leftLowerArm.rotation.z = -0.1;
  if (rightLowerArm) rightLowerArm.rotation.z = 0.1;
  if (leftHand) leftHand.rotation.z = -0.08;
  if (rightHand) rightHand.rotation.z = 0.08;
}

// ---------------------------------------------------------------------------
// 6. Camera Framing & Pastel Frock Swatches
// ---------------------------------------------------------------------------
function applyCameraPreset(preset) {
  state.cameraPreset = preset;
  const H = state.characterHeight || 1.65;

  if (preset === 'full') {
    controls.target.set(0, H * 0.49, 0);
    camera.position.set(0, H * 0.52, H * 2.28);
  } else if (preset === 'side') {
    // 3/4 & Side profile framing to showcase nose/jaw profile, hair locks & A-line frock flare
    controls.target.set(0, H * 0.53, 0.02);
    camera.position.set(H * 1.45, H * 0.56, H * 1.42);
  } else if (preset === 'upper') {
    controls.target.set(0, H * 0.72, 0);
    camera.position.set(0, H * 0.75, H * 1.12);
  } else if (preset === 'face') {
    controls.target.set(0, H * 0.875, 0.04);
    camera.position.set(0, H * 0.885, H * 0.53);
  }
  controls.update();

  document.querySelectorAll('.cam-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.cam === preset);
  });
}

function ensureNeutralFrockMap(mat) {
  if (!mat || !mat.map || !mat.map.image) return null;
  if (!mat.userData.babyPinkMap) {
    mat.userData.babyPinkMap = mat.map;
  }
  if (!mat.userData.neutralMap) {
    try {
      const img = mat.userData.babyPinkMap.image;
      const w = img.width || 512;
      const h = img.height || 512;
      const cvs = document.createElement('canvas');
      cvs.width = w;
      cvs.height = h;
      const ctx = cvs.getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);
      const imgData = ctx.getImageData(0, 0, w, h);
      const d = imgData.data;
      for (let i = 0; i < d.length; i += 4) {
        const r = d[i];
        const g = d[i + 1];
        const b = d[i + 2];
        // Preserve green leaves and warm gold floral centers; normalize pink fabric to white-luminance
        const isLeaf = g > r + 8 && g > b + 4;
        const isGoldCenter = r > 225 && g > 185 && b < 165;
        if (!isLeaf && !isGoldCenter) {
          const luma = Math.min(255, Math.round((0.32 * r + 0.34 * g + 0.34 * b) * 1.13));
          d[i] = luma;
          d[i + 1] = luma;
          d[i + 2] = luma;
        }
      }
      ctx.putImageData(imgData, 0, 0);
      const tex = new THREE.CanvasTexture(cvs);
      tex.flipY = mat.userData.babyPinkMap.flipY;
      tex.colorSpace = mat.userData.babyPinkMap.colorSpace;
      tex.wrapS = mat.userData.babyPinkMap.wrapS;
      tex.wrapT = mat.userData.babyPinkMap.wrapT;
      tex.needsUpdate = true;
      mat.userData.neutralMap = tex;
    } catch (_e) {
      mat.userData.neutralMap = mat.userData.babyPinkMap;
    }
  }
  return mat.userData.neutralMap;
}

function applyOutfitSwatch(swatchKey) {
  const sw = OUTFIT_SWATCHES[swatchKey] || OUTFIT_SWATCHES.babypink;
  state.outfitKey = swatchKey;

  const isVrm = Boolean(currentVrm);
  const targetHex = isVrm ? sw.vrmHex : sw.hex;

  outfitMaterials.forEach((mat) => {
    if (!isVrm && mat.map) {
      ensureNeutralFrockMap(mat);
      if (swatchKey === 'babypink' && mat.userData.babyPinkMap) {
        mat.map = mat.userData.babyPinkMap;
      } else if (mat.userData.neutralMap) {
        mat.map = mat.userData.neutralMap;
      }
      mat.needsUpdate = true;
    }
    if (mat.color) {
      mat.color.setHex(targetHex);
    }
  });

  document.querySelectorAll('.swatch-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.outfit === swatchKey);
  });
}

// ---------------------------------------------------------------------------
// 7. Facial Expressions, Blinking & Conversational Viseme System
// ---------------------------------------------------------------------------
// Note: Because +0.60 eyeWide, +0.40 browInnerUp, and +0.24 mouthSmileLeft/Right
// are already baked into AARNA v2's sculpted rest-pose geometry, runtime morph weights
// add natural, expressive emotional nuances on top of her cute resting face!
const GLB_EXPRESSION_TARGETS = {
  cheerful: {
    mouthSmile: 0.20,
    cheekSquintLeft: 0.08,
    cheekSquintRight: 0.08,
    browInnerUp: 0.08,
  },
  neutral: {
    mouthSmile: 0.08,
    browInnerUp: 0.02,
  },
  happy: {
    mouthSmile: 0.34,
    mouthOpen: 0.08,
    cheekSquintLeft: 0.16,
    cheekSquintRight: 0.16,
    eyeSquintLeft: 0.14,
    eyeSquintRight: 0.14,
    browInnerUp: 0.14,
  },
  curious: {
    eyeWideLeft: 0.22,
    eyeWideRight: 0.22,
    browInnerUp: 0.36,
    browOuterUpLeft: 0.32,
    browOuterUpRight: 0.32,
    mouthOpen: 0.14,
    viseme_O: 0.18,
    mouthSmile: 0.12,
  },
  surprised: {
    eyeWideLeft: 0.28,
    eyeWideRight: 0.28,
    browInnerUp: 0.42,
    browOuterUpLeft: 0.38,
    browOuterUpRight: 0.38,
    mouthOpen: 0.18,
    viseme_O: 0.24,
  },
  thinking: {
    browInnerUp: 0.28,
    browOuterUpRight: 0.34,
    browDownLeft: 0.18,
    eyeSquintLeft: 0.16,
    eyeLookUpRight: 0.28,
    eyeLookOutRight: 0.18,
    eyeLookUpLeft: 0.28,
    eyeLookInLeft: 0.18,
    mouthPucker: 0.20,
    mouthRollLower: 0.16,
    mouthLeft: 0.12,
    mouthSmile: 0.08,
  },
  relaxed: {
    mouthSmile: 0.18,
    eyeSquintLeft: 0.14,
    eyeSquintRight: 0.14,
    browInnerUp: 0.06,
  },
  concerned: {
    browInnerUp: 0.52,
    browDownLeft: 0.12,
    browDownRight: 0.12,
    mouthFrownLeft: 0.24,
    mouthFrownRight: 0.24,
    mouthPressLeft: 0.14,
    mouthPressRight: 0.14,
  },
  sad: {
    browInnerUp: 0.52,
    mouthFrownLeft: 0.26,
    mouthFrownRight: 0.26,
    mouthPressLeft: 0.14,
    mouthPressRight: 0.14,
  },
};

function updateFacialAnimation(elapsed, dt) {
  // 1. Natural blinking at varied intervals (with occasional realistic double-blink)
  if (!state.isBlinking && elapsed >= state.nextBlinkTime) {
    state.isBlinking = true;
    state.blinkStartTime = elapsed;
  }

  let blinkWeight = 0;
  if (state.isBlinking) {
    const bElapsed = elapsed - state.blinkStartTime;
    const blinkDuration = 0.14;
    if (bElapsed < blinkDuration) {
      blinkWeight = Math.sin((bElapsed / blinkDuration) * Math.PI);
    } else {
      state.isBlinking = false;
      if (state.pendingDoubleBlink) {
        state.pendingDoubleBlink = false;
        state.nextBlinkTime = elapsed + 0.12;
      } else {
        state.pendingDoubleBlink = Math.random() < 0.22;
        state.nextBlinkTime = elapsed + 2.1 + Math.random() * 2.8;
      }
    }
  }

  // 2. Micro-saccade eye gaze updates
  if (elapsed >= state.nextSaccadeTime) {
    if (state.autoGaze) {
      state.eyeTargetX = (Math.random() - 0.5) * 0.22 + state.pointerX * 0.32;
      state.eyeTargetY = (Math.random() - 0.5) * 0.12 + state.pointerY * 0.22;
    } else {
      state.eyeTargetX = 0;
      state.eyeTargetY = 0;
    }
    state.nextSaccadeTime = elapsed + 1.1 + Math.random() * 1.9;
  }
  state.eyeCurrentX += (state.eyeTargetX - state.eyeCurrentX) * Math.min(1, dt * 8.5);
  state.eyeCurrentY += (state.eyeTargetY - state.eyeCurrentY) * Math.min(1, dt * 8.5);

  // 3. Speech & Mouth Viseme Articulation (active during viseme demo OR 'speaking' gesture)
  let activeViseme = null;
  let visemeWeight = 0;
  const isSpeakingGesture = state.activeGesture === 'speaking';

  // Smooth live audio amplitude from Edge-TTS playback stream
  const targetAudioLvl = Number(state.audioLevel) || 0.0;
  state.smoothAudioLevel += (targetAudioLvl - state.smoothAudioLevel) * Math.min(1.0, dt * 26.0);
  if (targetAudioLvl === 0.0 && state.smoothAudioLevel < 0.005) {
    state.smoothAudioLevel = 0.0;
  }
  const curAudioLvl = state.smoothAudioLevel;

  if (state.visemeDemoActive || isSpeakingGesture) {
    const vStart = isSpeakingGesture ? state.gestureStartTime : state.visemeDemoStart;
    const vTime = elapsed - vStart;
    const seq = ['aa', 'ee', 'ih', 'oh', 'ou', 'aa', 'ee', 'oh'];
    const segDur = 0.35;
    if (isSpeakingGesture) {
      // Loop smoothly for any speech length with natural vowel phrasing
      const idx = Math.floor(vTime / segDur) % seq.length;
      activeViseme = seq[idx];
      const localPhase = (vTime % segDur) / segDur;
      visemeWeight = Math.sin(localPhase * Math.PI) * 0.72;
    } else {
      const idx = Math.floor(vTime / segDur);
      if (idx < seq.length) {
        activeViseme = seq[idx];
        const localPhase = (vTime % segDur) / segDur;
        visemeWeight = Math.sin(localPhase * Math.PI) * 0.72;
      } else {
        state.visemeDemoActive = false;
        const btn = document.getElementById('btn-viseme-demo');
        if (btn) btn.classList.remove('active');
      }
    }
  }

  const isListening = state.assistantState === 'listening';

  if (currentVrm && currentVrm.expressionManager) {
    const em = currentVrm.expressionManager;
    const allPresets = ['happy', 'relaxed', 'surprised', 'sad', 'angry', 'neutral', 'aa', 'ee', 'ih', 'oh', 'ou'];
    allPresets.forEach((p) => em.setValue(p, 0));

    if (state.activeExpression === 'cheerful') {
      em.setValue('happy', isListening ? 0.55 : 0.42);
      em.setValue('relaxed', 0.25);
    } else if (state.activeExpression === 'happy') {
      em.setValue('happy', 0.92);
    } else if (state.activeExpression === 'curious' || state.activeExpression === 'surprised') {
      em.setValue('surprised', 0.85);
    } else if (state.activeExpression === 'thinking' || state.activeExpression === 'relaxed') {
      em.setValue('relaxed', 0.82);
    } else if (state.activeExpression === 'concerned' || state.activeExpression === 'sad') {
      em.setValue('sad', 0.78);
    } else {
      em.setValue('neutral', 1.0);
    }

    em.setValue('blink', blinkWeight);
    if (curAudioLvl > 0.001) {
      // Sound-reactive lip sync on VRM humanoid expressions
      em.setValue('aa', Math.min(1.0, curAudioLvl * 0.90));
      em.setValue('oh', Math.min(1.0, curAudioLvl * 0.45));
      em.setValue('ih', Math.min(0.6, curAudioLvl * 0.30));
    } else if (activeViseme) {
      em.setValue(activeViseme, visemeWeight);
    }
  } else if (glbMorphMeshes.length > 0) {
    const presetWeights = { ...(GLB_EXPRESSION_TARGETS[state.activeExpression] || GLB_EXPRESSION_TARGETS.cheerful) };

    if (isListening && state.activeExpression === 'cheerful') {
      presetWeights.mouthSmile = 0.34;
      presetWeights.browInnerUp = 0.24;
      presetWeights.eyeWideLeft = 0.14;
      presetWeights.eyeWideRight = 0.14;
    }

    presetWeights.eyeBlinkLeft = Math.max(presetWeights.eyeBlinkLeft || 0, blinkWeight);
    presetWeights.eyeBlinkRight = Math.max(presetWeights.eyeBlinkRight || 0, blinkWeight);
    presetWeights.eyesClosed = blinkWeight;

    // Subtle organic breathing micro-smile modulation
    presetWeights.mouthSmile = (presetWeights.mouthSmile || 0) + 0.03 * Math.sin(elapsed * 1.1);

    // Eye gaze morph targets
    if (state.eyeCurrentX > 0) {
      presetWeights.eyeLookOutLeft = Math.max(presetWeights.eyeLookOutLeft || 0, state.eyeCurrentX);
      presetWeights.eyeLookInRight = Math.max(presetWeights.eyeLookInRight || 0, state.eyeCurrentX);
    } else {
      presetWeights.eyeLookInLeft = Math.max(presetWeights.eyeLookInLeft || 0, -state.eyeCurrentX);
      presetWeights.eyeLookOutRight = Math.max(presetWeights.eyeLookOutRight || 0, -state.eyeCurrentX);
    }

    // Viseme mapping for GLB
    const glbVisemeMap = {
      aa: 'viseme_aa',
      ee: 'viseme_E',
      ih: 'viseme_I',
      oh: 'viseme_O',
      ou: 'viseme_U',
    };
    if (curAudioLvl > 0.001) {
      // Sound-Reactive Lip Sync: jaw and mouth vertex meshes deform and actuate in lockstep with live audio amplitude
      presetWeights.jawOpen = Math.min(1.0, curAudioLvl * 0.96);
      presetWeights.mouthOpen = Math.min(1.0, curAudioLvl * 0.78);
      presetWeights.viseme_aa = Math.min(1.0, curAudioLvl * 0.65);
      presetWeights.viseme_O = Math.min(1.0, curAudioLvl * 0.38);
      presetWeights.viseme_I = Math.min(0.5, curAudioLvl * 0.22);
      presetWeights.browInnerUp = Math.max(presetWeights.browInnerUp || 0, curAudioLvl * 0.18);
    } else if (activeViseme && glbVisemeMap[activeViseme]) {
      // Test fallback / simulated viseme demo
      presetWeights[glbVisemeMap[activeViseme]] = visemeWeight;
      presetWeights.mouthOpen = visemeWeight * 0.42;
      presetWeights.jawOpen = visemeWeight * 0.28;
      presetWeights.browInnerUp = Math.max(presetWeights.browInnerUp || 0, visemeWeight * 0.25);
    }

    // Smooth critically-damped expression transition
    const lerpSpeed = Math.min(1, dt * 8.5);
    glbMorphMeshes.forEach((mesh) => {
      const dict = mesh.morphTargetDictionary;
      const infl = mesh.morphTargetInfluences;
      if (!dict || !infl) return;

      for (const [morphName, idx] of Object.entries(dict)) {
        const targetVal = presetWeights[morphName] || 0;
        const speed = morphName.includes('Blink') || morphName === 'eyesClosed' ? Math.min(1, dt * 28) : lerpSpeed;
        infl[idx] += (targetVal - infl[idx]) * speed;
      }
    });
  }
}

// ---------------------------------------------------------------------------
// 8. Lively Idle, Weight Shifting, Finger Curl, Hair Physics & Gesture Controller
// ---------------------------------------------------------------------------
const _euler = new THREE.Euler();
const _quat = new THREE.Quaternion();

function applyGlbBoneOffset(boneName, rx, ry, rz) {
  const bone = glbBones[boneName];
  const rest = glbRestQuats[boneName];
  if (!bone || !rest) return;
  _euler.set(rx, ry, rz, 'XYZ');
  _quat.setFromEuler(_euler);
  bone.quaternion.copy(rest).multiply(_quat);
}

function applyNaturalFingerCurl(elapsed, leftOpenFactor = 0, rightOpenFactor = 0) {
  // Applies a natural, relaxed feminine finger curl across all 30 finger phalanges
  // so hands never look stiff or flat.
  const microBreath = Math.sin(elapsed * 1.45) * 0.03;
  const fingers = [
    { name: 'Index', baseCurl: 0.22 },
    { name: 'Middle', baseCurl: 0.28 },
    { name: 'Ring', baseCurl: 0.34 },
    { name: 'Pinky', baseCurl: 0.40 },
  ];

  for (const side of ['Left', 'Right']) {
    const sSign = side === 'Left' ? 1 : -1;
    const open = side === 'Left' ? leftOpenFactor : rightOpenFactor;
    const curlScale = Math.max(0.08, 1.0 - open * 0.72);

    for (const f of fingers) {
      for (let p = 1; p <= 3; p++) {
        const bName = `${side}Hand${f.name}${p}`;
        const c = (f.baseCurl + microBreath) * curlScale * (p === 1 ? 0.75 : p === 2 ? 1.1 : 0.85);
        applyGlbBoneOffset(bName, c * 0.15, 0, sSign * c);
      }
    }
    // Natural thumb posture
    applyGlbBoneOffset(`${side}HandThumb1`, 0.18 * curlScale, sSign * 0.22 * curlScale, sSign * 0.12);
    applyGlbBoneOffset(`${side}HandThumb2`, 0.12 * curlScale, 0, sSign * 0.18 * curlScale);
  }
}

function updateSkeletalAnimation(elapsed, dt) {
  const isListening = state.assistantState === 'listening';
  const safeDt = Math.max(0.001, Math.min(0.05, dt));

  // 1. Multi-layered organic breathing & subtle 3-harmonic contraposto weight shifting
  const breathFreq = isListening ? 1.92 : 1.44;
  const breathWave = Math.sin(elapsed * breathFreq);
  const chestLagWave = Math.sin(elapsed * breathFreq - 0.38);

  // Non-repeating 3-harmonic weight-shift wave (subtle, relaxed, never robotic)
  const weightShift =
    0.55 * Math.sin(elapsed * 0.62) +
    0.30 * Math.sin(elapsed * 0.27 + 1.15) +
    0.15 * Math.cos(elapsed * 0.43 - 0.6);
  const bodySwayZ = weightShift * 0.018;
  const hipShiftX = weightShift * 0.010; // ±10 mm subtle lateral hip weight shift

  // 2. Smooth 2nd-order critically damped cervical head turns & varied head tilts
  if (elapsed >= state.nextHeadDriftTime) {
    state.headDriftTargetYaw = (Math.random() - 0.5) * 0.11;
    state.headDriftTargetPitch = (Math.random() - 0.46) * 0.055;
    state.headDriftTargetRoll = (Math.random() - 0.5) * 0.085;
    state.nextHeadDriftTime = elapsed + 2.8 + Math.random() * 3.2;
  }
  const headStiff = 6.5;
  const headDamp = 5.1;
  state.headDriftVelYaw += ((state.headDriftTargetYaw - state.headDriftYaw) * headStiff - state.headDriftVelYaw * headDamp) * safeDt;
  state.headDriftVelPitch += ((state.headDriftTargetPitch - state.headDriftPitch) * headStiff - state.headDriftVelPitch * headDamp) * safeDt;
  state.headDriftVelRoll += ((state.headDriftTargetRoll - state.headDriftRoll) * headStiff - state.headDriftVelRoll * headDamp) * safeDt;
  state.headDriftYaw += state.headDriftVelYaw * safeDt;
  state.headDriftPitch += state.headDriftVelPitch * safeDt;
  state.headDriftRoll += state.headDriftVelRoll * safeDt;

  // 3. Combine manual sliders + organic idle head movement + expression posture nuances
  const deg2rad = Math.PI / 180;
  let headYaw = state.manualPose.yaw * deg2rad + state.headDriftYaw + (state.autoGaze ? state.pointerX * 0.12 : 0);
  let headPitch = state.manualPose.pitch * deg2rad + state.headDriftPitch + chestLagWave * 0.012 + (state.autoGaze ? -state.pointerY * 0.07 : 0);
  let headRoll = state.manualPose.roll * deg2rad + state.headDriftRoll - bodySwayZ * 0.55;
  let torsoYaw = state.manualPose.torso * deg2rad + weightShift * 0.025;
  let torsoPitch = breathWave * 0.016 + (isListening ? 0.036 : 0);

  // Subtle head inclination matching active facial expression
  if (state.activeExpression === 'curious') {
    headRoll += 0.10;
    headPitch -= 0.022;
  } else if (state.activeExpression === 'thinking') {
    headYaw += 0.075;
    headPitch -= 0.04;
    headRoll -= 0.055;
  } else if (state.activeExpression === 'concerned') {
    headRoll += 0.065;
    headPitch += 0.035;
  }

  if (isListening) {
    headRoll += 0.08; // Attentive ~4.5 deg inquisitive head tilt on "Aarna!"
    headPitch -= 0.024;
  }

  // 4. Active Gesture Controller ('wave', 'tilt', 'nod', 'hands', 'speaking') with Smooth Crossfade
  let targetRightArmLift = 0;
  let targetRightForeArmBend = 0;
  let targetRightWristWave = 0;
  let targetLeftArmLift = 0;
  let targetLeftForeArmBend = 0;
  let targetLeftWristTurn = 0;
  let targetLeftHandOpen = 0;
  let targetRightHandOpen = 0;
  let targetHeadGestureYaw = 0;
  let targetHeadGesturePitch = 0;
  let targetHeadGestureRoll = 0;
  let targetTorsoGesturePitch = 0;

  if (state.activeGesture !== 'idle') {
    const gTime = elapsed - state.gestureStartTime;

    if (state.activeGesture === 'wave') {
      // Small, warm, friendly wave near shoulder/chest height (not exaggerated overhead!)
      const totalDur = 3.2;
      if (gTime < totalDur) {
        const env = gTime < 0.52
          ? 0.5 - 0.5 * Math.cos((gTime / 0.52) * Math.PI)
          : gTime > totalDur - 0.65
          ? 0.5 - 0.5 * Math.cos(((totalDur - gTime) / 0.65) * Math.PI)
          : 1.0;
        targetRightArmLift = env * 1.08;
        targetRightForeArmBend = env * 0.88;
        targetRightWristWave = env * Math.sin(gTime * 7.6) * 0.21;
        targetRightHandOpen = env * 0.82;
        targetHeadGestureRoll = env * 0.085;
        targetHeadGestureYaw = env * 0.045;
      } else {
        setActiveGesture('idle');
      }
    } else if (state.activeGesture === 'tilt') {
      const totalDur = 2.8;
      if (gTime < totalDur) {
        const env = 0.5 - 0.5 * Math.cos((gTime / totalDur) * 2.0 * Math.PI);
        targetHeadGestureRoll = env * 0.19;
        targetHeadGestureYaw = env * 0.07;
        targetHeadGesturePitch = -env * 0.028;
        targetRightForeArmBend = env * 0.15;
      } else {
        setActiveGesture('idle');
      }
    } else if (state.activeGesture === 'nod') {
      const totalDur = 2.2;
      if (gTime < totalDur) {
        const env = Math.sin((gTime / totalDur) * Math.PI);
        targetHeadGesturePitch = env * Math.sin(gTime * 5.8) * 0.14;
        targetTorsoGesturePitch = env * Math.sin(gTime * 5.8) * 0.03;
      } else {
        setActiveGesture('idle');
      }
    } else if (state.activeGesture === 'hands') {
      // Natural two-hand conversational open-palm movement
      const totalDur = 3.2;
      if (gTime < totalDur) {
        const env = 0.5 - 0.5 * Math.cos((gTime / totalDur) * 2.0 * Math.PI);
        const pulse = 0.78 + 0.22 * Math.sin(gTime * 3.8);
        targetLeftArmLift = env * 0.34 * pulse;
        targetRightArmLift = env * 0.38 * pulse;
        targetLeftForeArmBend = env * 0.64 * pulse;
        targetRightForeArmBend = env * 0.70 * pulse;
        targetLeftWristTurn = env * 0.18 * Math.sin(gTime * 3.2);
        targetRightWristWave = env * 0.20 * Math.cos(gTime * 3.2);
        targetLeftHandOpen = env * 0.72;
        targetRightHandOpen = env * 0.72;
        targetHeadGestureRoll = env * 0.045 * Math.sin(gTime * 2.2);
      } else {
        setActiveGesture('idle');
      }
    } else if (state.activeGesture === 'speaking') {
      // Natural conversational gestures during TTS speech:
      // subtle arm movement + natural head movement, smoothly continuous until Python calls stop_speaking()
      const env = Math.min(1.0, gTime / 0.45);
      const cadence = 0.68 + 0.32 * Math.sin(gTime * 3.8);
      targetRightArmLift = env * 0.28 * cadence;
      targetRightForeArmBend = env * 0.52 * cadence;
      targetRightWristWave = env * 0.12 * Math.sin(gTime * 4.2);
      targetLeftArmLift = env * 0.16 * (0.7 + 0.3 * Math.cos(gTime * 3.0));
      targetLeftForeArmBend = env * 0.32 * (0.7 + 0.3 * Math.cos(gTime * 3.0));
      targetRightHandOpen = env * 0.55;
      targetLeftHandOpen = env * 0.42;
      targetHeadGesturePitch = env * 0.040 * Math.sin(gTime * 4.2);
      targetHeadGestureYaw = env * 0.035 * Math.cos(gTime * 2.2);
      targetHeadGestureRoll = env * 0.032 * Math.sin(gTime * 2.6);
    }
  }

  // Smooth exponential crossfade across all gesture channels so switching or ending gestures never pops
  const gLerp = 1.0 - Math.exp(-safeDt * 9.5);
  state.curRightArmLift += (targetRightArmLift - state.curRightArmLift) * gLerp;
  state.curRightForeArmBend += (targetRightForeArmBend - state.curRightForeArmBend) * gLerp;
  state.curRightWristWave += (targetRightWristWave - state.curRightWristWave) * gLerp;
  state.curLeftArmLift += (targetLeftArmLift - state.curLeftArmLift) * gLerp;
  state.curLeftForeArmBend += (targetLeftForeArmBend - state.curLeftForeArmBend) * gLerp;
  state.curLeftWristTurn += (targetLeftWristTurn - state.curLeftWristTurn) * gLerp;
  state.curLeftHandOpen += (targetLeftHandOpen - state.curLeftHandOpen) * gLerp;
  state.curRightHandOpen += (targetRightHandOpen - state.curRightHandOpen) * gLerp;
  state.curHeadGestureYaw += (targetHeadGestureYaw - state.curHeadGestureYaw) * gLerp;
  state.curHeadGesturePitch += (targetHeadGesturePitch - state.curHeadGesturePitch) * gLerp;
  state.curHeadGestureRoll += (targetHeadGestureRoll - state.curHeadGestureRoll) * gLerp;
  state.curTorsoGesturePitch += (targetTorsoGesturePitch - state.curTorsoGesturePitch) * gLerp;

  const rightArmLift = state.curRightArmLift;
  const rightForeArmBend = state.curRightForeArmBend;
  const rightWristWave = state.curRightWristWave;
  const leftArmLift = state.curLeftArmLift;
  const leftForeArmBend = state.curLeftForeArmBend;
  const leftWristTurn = state.curLeftWristTurn;
  const leftHandOpen = state.curLeftHandOpen;
  const rightHandOpen = state.curRightHandOpen;
  headYaw += state.curHeadGestureYaw;
  headPitch += state.curHeadGesturePitch;
  headRoll += state.curHeadGestureRoll;
  torsoPitch += state.curTorsoGesturePitch;

  // 5. Real-Time 2nd-Order Hooke Spring-Damper Hair Physics Solver (drives HeadTop_End Joint 6)
  const headVelYaw = (headYaw - state.prevHeadYaw) / safeDt;
  const headVelPitch = (headPitch - state.prevHeadPitch) / safeDt;
  const headVelRoll = (headRoll - state.prevHeadRoll) / safeDt;
  state.prevHeadYaw = headYaw;
  state.prevHeadPitch = headPitch;
  state.prevHeadRoll = headRoll;

  // Target hair lag from head angular velocity + gravity droop + subtle breath breeze
  const targetHairX = -headVelPitch * 0.042 - headPitch * 0.32 + Math.sin(elapsed * 1.7) * 0.020;
  const targetHairY = -headVelYaw * 0.042 - headYaw * 0.25 + Math.cos(elapsed * 1.3) * 0.016;
  const targetHairZ = -headVelRoll * 0.050 - headRoll * 0.38 - bodySwayZ * 0.7 + Math.sin(elapsed * 2.1) * 0.018;

  const stiffness = 64.0;
  const damping = 9.8;
  state.hairVelX += ((targetHairX - state.hairSpringX) * stiffness - state.hairVelX * damping) * safeDt;
  state.hairVelY += ((targetHairY - state.hairSpringY) * stiffness - state.hairVelY * damping) * safeDt;
  state.hairVelZ += ((targetHairZ - state.hairSpringZ) * stiffness - state.hairVelZ * damping) * safeDt;
  state.hairSpringX = Math.max(-0.26, Math.min(0.26, state.hairSpringX + state.hairVelX * safeDt));
  state.hairSpringY = Math.max(-0.26, Math.min(0.26, state.hairSpringY + state.hairVelY * safeDt));
  state.hairSpringZ = Math.max(-0.26, Math.min(0.26, state.hairSpringZ + state.hairVelZ * safeDt));

  // 6. Apply to VRM Humanoid or GLB Skeleton
  if (currentVrm && currentVrm.humanoid) {
    const hum = currentVrm.humanoid;
    const hips = hum.getNormalizedBoneNode('hips');
    const spine = hum.getNormalizedBoneNode('spine');
    const chest = hum.getNormalizedBoneNode('chest');
    const neck = hum.getNormalizedBoneNode('neck');
    const head = hum.getNormalizedBoneNode('head');
    const leftUpperArm = hum.getNormalizedBoneNode('leftUpperArm');
    const leftLowerArm = hum.getNormalizedBoneNode('leftLowerArm');
    const rightUpperArm = hum.getNormalizedBoneNode('rightUpperArm');
    const rightLowerArm = hum.getNormalizedBoneNode('rightLowerArm');
    const rightHand = hum.getNormalizedBoneNode('rightHand');

    if (hips) {
      hips.rotation.set(0, torsoYaw * 0.25, bodySwayZ * 0.5);
    }
    if (spine) {
      spine.rotation.set(torsoPitch * 0.5, torsoYaw * 0.45, -bodySwayZ * 0.4);
    }
    if (chest) {
      chest.rotation.set(chestLagWave * 0.018, torsoYaw * 0.35, 0);
    }
    if (neck) {
      neck.rotation.set(headPitch * 0.35, headYaw * 0.35, headRoll * 0.35);
    }
    if (head) {
      head.rotation.set(headPitch * 0.65, headYaw * 0.65, headRoll * 0.65);
    }
    if (leftUpperArm) {
      leftUpperArm.rotation.set(leftArmLift * 0.25, 0, -1.16 + leftArmLift * 0.85 + breathWave * 0.015);
    }
    if (leftLowerArm) {
      leftLowerArm.rotation.set(0, -leftForeArmBend * 0.35, -0.1 - leftForeArmBend * 0.75 + leftWristTurn);
    }
    if (rightUpperArm) {
      rightUpperArm.rotation.set(
        rightArmLift * 0.25,
        0,
        1.16 - rightArmLift * 1.05 - breathWave * 0.015
      );
    }
    if (rightLowerArm) {
      rightLowerArm.rotation.set(0, rightForeArmBend * 0.4, 0.1 + rightForeArmBend * 0.85 + rightWristWave);
    }
    if (rightHand) {
      rightHand.rotation.set(0, 0, 0.08 + rightWristWave * 0.65);
    }

    currentVrm.update(dt);
  } else if (Object.keys(glbBones).length > 0) {
    // Contraposto Hips translation & rotation + Leg counter-balance so feet stay planted
    if (glbBones.Hips && glbRestPos.Hips) {
      glbBones.Hips.position.x = glbRestPos.Hips.x + hipShiftX;
      glbBones.Hips.position.y = glbRestPos.Hips.y - Math.abs(weightShift) * 0.0022 + breathWave * 0.0016;
    }
    applyGlbBoneOffset('Hips', 0.008 * breathWave, torsoYaw * 0.22, bodySwayZ * 0.72);

    // Subtle leg & knee counter-balance for natural weight shifting
    const leftKneeRelax = Math.max(0, -weightShift) * 0.036;
    const rightKneeRelax = Math.max(0, weightShift) * 0.036;
    applyGlbBoneOffset('LeftUpLeg', -leftKneeRelax * 0.55, 0, -bodySwayZ * 0.65);
    applyGlbBoneOffset('LeftLeg', leftKneeRelax, 0, 0);
    applyGlbBoneOffset('LeftFoot', -leftKneeRelax * 0.45, 0, bodySwayZ * 0.35);

    applyGlbBoneOffset('RightUpLeg', -rightKneeRelax * 0.55, 0, -bodySwayZ * 0.65);
    applyGlbBoneOffset('RightLeg', rightKneeRelax, 0, 0);
    applyGlbBoneOffset('RightFoot', -rightKneeRelax * 0.45, 0, bodySwayZ * 0.35);

    // Spine & Chest breathing + counter-sway
    applyGlbBoneOffset('Spine', torsoPitch * 0.42, torsoYaw * 0.35, -bodySwayZ * 0.45);
    applyGlbBoneOffset('Spine1', torsoPitch * 0.38, torsoYaw * 0.28, -bodySwayZ * 0.25);
    applyGlbBoneOffset('Spine2', chestLagWave * 0.018, torsoYaw * 0.22, 0);

    // Neck & Head + Eye Bones + Hair Spring Physics Bone (HeadTop_End)
    applyGlbBoneOffset('Neck', headPitch * 0.36, headYaw * 0.36, headRoll * 0.36);
    applyGlbBoneOffset('Head', headPitch * 0.64, headYaw * 0.64, headRoll * 0.64);
    applyGlbBoneOffset('HeadTop_End', state.hairSpringX, state.hairSpringY, state.hairSpringZ);

    applyGlbBoneOffset('LeftEye', state.eyeCurrentY * 0.18, state.eyeCurrentX * 0.22, 0);
    applyGlbBoneOffset('RightEye', state.eyeCurrentY * 0.18, state.eyeCurrentX * 0.22, 0);

    // Relaxed feminine shoulder & arm posture with organic multi-frequency micro-movement
    const armSwayL = 0.018 * Math.sin(elapsed * 0.62 + 0.5) + 0.008 * Math.cos(elapsed * 1.15);
    const armSwayR = 0.018 * Math.cos(elapsed * 0.62 + 0.5) + 0.008 * Math.sin(elapsed * 1.08);
    const foreArmRelaxL = 0.016 * Math.sin(elapsed * 0.85 + 1.2);
    const foreArmRelaxR = 0.016 * Math.cos(elapsed * 0.79 + 0.8);

    applyGlbBoneOffset('LeftShoulder', 0.006 * breathWave, 0, -chestLagWave * 0.014);
    applyGlbBoneOffset(
      'LeftArm',
      0.38 - leftArmLift * 0.35 + chestLagWave * 0.014,
      -0.06 - leftArmLift * 0.32,
      0.06 + armSwayL + leftArmLift * 0.28
    );
    applyGlbBoneOffset(
      'LeftForeArm',
      0.14 + foreArmRelaxL + leftForeArmBend * 0.45,
      -leftWristTurn * 0.25,
      0.14 + leftForeArmBend * 1.15 + leftWristTurn * 0.35
    );
    applyGlbBoneOffset('LeftHand', 0.05 + foreArmRelaxL * 0.4, -0.08, 0.08 + leftWristTurn * 0.5);

    applyGlbBoneOffset('RightShoulder', 0.006 * breathWave, 0, chestLagWave * 0.014);
    applyGlbBoneOffset(
      'RightArm',
      0.38 - rightArmLift * 0.38 + chestLagWave * 0.014,
      0.06 + rightArmLift * 0.38,
      -0.06 + armSwayR - rightArmLift * 0.34
    );
    applyGlbBoneOffset(
      'RightForeArm',
      0.14 + foreArmRelaxR + rightForeArmBend * 0.52,
      rightWristWave * 0.24,
      -0.14 - rightForeArmBend * 1.55 + rightWristWave * 0.42
    );
    applyGlbBoneOffset('RightHand', 0.05 + foreArmRelaxR * 0.4, 0.08, -0.08 + rightWristWave * 0.62);

    // Graceful finger curl across both hands
    applyNaturalFingerCurl(elapsed, leftHandOpen, rightHandOpen);
  }
}

// ---------------------------------------------------------------------------
// 9. Environment & Pedestal Animation
// ---------------------------------------------------------------------------
function updateEnvironment(elapsed, dt) {
  const isListening = state.assistantState === 'listening';
  const audioLvl = Number(state.audioLevel) || 0.0;
  const speedMult = isListening ? 2.4 : (1.0 + audioLvl * 2.5);

  starfield.rotation.y = elapsed * 0.012;

  const posAttr = goldenParticles.points.geometry.attributes.position;
  for (let i = 0; i < goldenParticles.count; i++) {
    let y = posAttr.getY(i) + dt * 0.08 * speedMult;
    if (y > 2.25) y = 0.02;
    posAttr.setY(i, y);
  }
  posAttr.needsUpdate = true;

  const pulse = 0.5 + 0.5 * Math.sin(elapsed * (isListening ? 4.5 : 2.0));
  pedestal.innerMat.opacity = isListening ? 0.75 + 0.25 * pulse : 0.45 + 0.2 * pulse + audioLvl * 0.35;
  pedestal.outerMat.opacity = isListening ? 0.55 + 0.25 * pulse : 0.30 + 0.15 * pulse + audioLvl * 0.25;
  pedestal.discMat.opacity = isListening ? 0.22 + 0.08 * pulse : 0.10 + 0.06 * pulse + audioLvl * 0.20;
  goldRimLight.intensity = isListening ? 3.5 : (2.35 + audioLvl * 1.8);

  // Rotate cybernetic projection pedestal grid
  if (pedestal.gridGroup) {
    pedestal.gridGroup.rotation.y += dt * (0.15 + audioLvl * 1.6);
  }

  // Animate sound-reactive neon wave ripples
  if (pedestal.rippleRings && pedestal.rippleRings.length > 0) {
    const ripSpeed = 0.55 + audioLvl * 2.8;
    pedestal.rippleRings.forEach((rip, idx) => {
      const pOffset = rip.userData.phaseOffset || (idx / pedestal.rippleRings.length);
      const phase = (elapsed * ripSpeed + pOffset) % 1.0;
      // Exponential outward propagation from center
      const curRadius = 0.12 + Math.pow(phase, 0.85) * 0.95;
      // Dynamic wave warp matching live audio frequency amplitude
      const waveWarp = 1.0 + Math.sin(phase * Math.PI * 4.0 + elapsed * 7.0) * (0.04 + audioLvl * 0.32);
      const s = curRadius * waveWarp;
      rip.scale.set(s, s, 1.0);

      // Ripple opacity pulses brightly with audio amplitude
      const baseAlpha = Math.sin(phase * Math.PI);
      const brightAlpha = baseAlpha * (0.28 + audioLvl * 0.70 + (isListening ? 0.25 : 0.0));
      rip.material.opacity = Math.max(0.0, Math.min(1.0, brightAlpha));

      // Color shifts towards electric white-cyan at peak audio levels
      if (audioLvl > 0.4) {
        rip.material.color.setHex(0xe0faff);
      } else {
        rip.material.color.setHex(0x00f3ff);
      }
    });
  }
}

// ---------------------------------------------------------------------------
// 10. State, UI & Power-Saving Controls
// ---------------------------------------------------------------------------
function setAssistantState(newState) {
  state.assistantState = newState === 'listening' ? 'listening' : 'idle';
  const isListening = state.assistantState === 'listening';

  const badge = document.getElementById('state-indicator');
  const text = document.getElementById('state-text');
  if (badge && text) {
    badge.className = `state-badge ${isListening ? 'state-listening' : 'state-idle'}`;
    text.textContent = isListening ? 'STATE: LISTENING ("Aarna!")' : 'STATE: IDLE';
  }

  document.getElementById('btn-state-idle')?.classList.toggle('active', !isListening);
  document.getElementById('btn-state-listening')?.classList.toggle('active', isListening);
  document.getElementById('status-message').textContent = isListening
    ? 'AARNA is attentively listening ("Aarna!" active)'
    : 'AARNA in lively natural idle state';

  if (isListening) {
    setActiveGesture('wave');
  } else if (state.activeGesture === 'wave') {
    setActiveGesture('idle');
  }
}

function setActiveExpression(expr) {
  state.activeExpression = expr;
  document.querySelectorAll('.expr-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.expr === expr);
  });
}

function setActiveGesture(gesture) {
  state.activeGesture = gesture;
  state.gestureStartTime = performance.now() / 1000;
  document.querySelectorAll('.gesture-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.gesture === gesture);
  });
}

function setRendererPaused(paused, reason = 'manual') {
  if (reason === 'visibility') {
    state.autoPausedByVisibility = paused;
  } else {
    state.rendererPaused = paused;
  }
  const effectivelyPaused = state.rendererPaused || state.autoPausedByVisibility;
  const renderHud = document.getElementById('hud-render-state');
  const pauseBtn = document.getElementById('btn-pause-render');

  if (renderHud) {
    renderHud.textContent = effectivelyPaused ? 'PAUSED (0% GPU)' : 'ACTIVE';
    renderHud.className = `metric-val ${effectivelyPaused ? 'status-paused' : 'status-active'}`;
  }
  if (pauseBtn) {
    pauseBtn.classList.toggle('active', state.rendererPaused);
    pauseBtn.textContent = state.rendererPaused ? 'Resume GPU' : 'Pause GPU';
  }
  if (effectivelyPaused) {
    document.getElementById('hud-fps').textContent = '0';
  } else {
    state.lastFrameTimestamp = performance.now();
  }
}

function updateGeometryMetrics() {
  let triangles = 0;
  if (currentAvatarRoot) {
    currentAvatarRoot.traverse((obj) => {
      if (obj.isMesh && obj.geometry) {
        const idx = obj.geometry.index;
        const pos = obj.geometry.attributes.position;
        if (idx) triangles += idx.count / 3;
        else if (pos) triangles += pos.count / 3;
      }
    });
  }
  const trisHud = document.getElementById('hud-tris');
  if (trisHud) {
    trisHud.textContent = `${(triangles / 1000).toFixed(1)}k`;
  }
  state.triangleCount = Math.round(triangles);
}

// ---------------------------------------------------------------------------
// 11. Event Listeners & Visibility Power Saving
// ---------------------------------------------------------------------------
function bindUiControls() {
  document.getElementById('btn-state-idle')?.addEventListener('click', () => setAssistantState('idle'));
  document.getElementById('btn-state-listening')?.addEventListener('click', () => setAssistantState('listening'));

  document.getElementById('model-select')?.addEventListener('change', (e) => {
    loadCharacterModel(e.target.value);
  });

  document.querySelectorAll('.swatch-btn').forEach((btn) => {
    btn.addEventListener('click', () => applyOutfitSwatch(btn.dataset.outfit));
  });

  document.querySelectorAll('.cam-btn').forEach((btn) => {
    btn.addEventListener('click', () => applyCameraPreset(btn.dataset.cam));
  });

  document.querySelectorAll('.fps-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.targetFps = parseInt(btn.dataset.fps, 10) || 60;
      document.querySelectorAll('.fps-btn').forEach((b) => b.classList.toggle('active', b === btn));
    });
  });

  document.getElementById('btn-pause-render')?.addEventListener('click', () => {
    setRendererPaused(!state.rendererPaused, 'manual');
  });

  document.querySelectorAll('.expr-btn').forEach((btn) => {
    btn.addEventListener('click', () => setActiveExpression(btn.dataset.expr));
  });

  document.getElementById('btn-trigger-blink')?.addEventListener('click', () => {
    state.isBlinking = true;
    state.blinkStartTime = performance.now() / 1000;
  });

  document.getElementById('btn-viseme-demo')?.addEventListener('click', (e) => {
    state.visemeDemoActive = true;
    state.visemeDemoStart = performance.now() / 1000;
    e.currentTarget.classList.add('active');
  });

  document.querySelectorAll('.gesture-btn').forEach((btn) => {
    btn.addEventListener('click', () => setActiveGesture(btn.dataset.gesture));
  });

  const bindSlider = (id, valId, key) => {
    const el = document.getElementById(id);
    const valEl = document.getElementById(valId);
    el?.addEventListener('input', () => {
      const v = parseFloat(el.value) || 0;
      state.manualPose[key] = v;
      if (valEl) valEl.textContent = `${v}°`;
    });
  };
  bindSlider('slider-yaw', 'val-yaw', 'yaw');
  bindSlider('slider-pitch', 'val-pitch', 'pitch');
  bindSlider('slider-roll', 'val-roll', 'roll');
  bindSlider('slider-torso', 'val-torso', 'torso');

  document.getElementById('btn-gaze-toggle')?.addEventListener('click', (e) => {
    state.autoGaze = !state.autoGaze;
    e.currentTarget.classList.toggle('active', state.autoGaze);
    e.currentTarget.textContent = `Eye Gaze: ${state.autoGaze ? 'Auto' : 'Fixed'}`;
  });

  document.getElementById('btn-reset-pose')?.addEventListener('click', () => {
    state.manualPose = { yaw: 0, pitch: 0, roll: 0, torso: 0 };
    ['yaw', 'pitch', 'roll', 'torso'].forEach((k) => {
      const s = document.getElementById(`slider-${k}`);
      const v = document.getElementById(`val-${k}`);
      if (s) s.value = '0';
      if (v) v.textContent = '0°';
    });
    applyCameraPreset('full');
  });

  window.addEventListener('pointermove', (e) => {
    state.pointerX = (e.clientX / window.innerWidth) * 2 - 1;
    state.pointerY = (e.clientY / window.innerHeight) * 2 - 1;
  });

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  document.addEventListener('visibilitychange', () => {
    setRendererPaused(document.hidden, 'visibility');
  });
}

// ---------------------------------------------------------------------------
// 12. Main Render Loop & Telemetry
// ---------------------------------------------------------------------------
function animate(nowMs) {
  requestAnimationFrame(animate);

  if (state.rendererPaused || state.autoPausedByVisibility) {
    return;
  }

  const minFrameInterval = state.targetFps === 30 ? 1000 / 31 : 0;
  const deltaMs = nowMs - state.lastFrameTimestamp;
  if (minFrameInterval > 0 && deltaMs < minFrameInterval) {
    return;
  }

  state.lastFrameTimestamp = nowMs;
  const dt = Math.min(0.1, deltaMs / 1000);
  const elapsed = nowMs / 1000;

  controls.update();
  updateFacialAnimation(elapsed, dt);
  updateSkeletalAnimation(elapsed, dt);
  updateEnvironment(elapsed, dt);

  // Update cybernetic hologram uniforms & chest script badge
  holoUniforms.uHoloTime.value = elapsed;
  holoUniforms.uAudioLevel.value = state.audioLevel;
  if (chestBadgeMat) {
    const pulse = 0.5 + 0.5 * Math.sin(elapsed * 3.2);
    chestBadgeMat.opacity = 0.82 + 0.18 * pulse + state.audioLevel * 0.15;
  }
  holoWireframeMeshes.forEach((wm) => {
    if (wm && wm.material) {
      wm.material.opacity = 0.14 + state.audioLevel * 0.18;
    }
  });

  renderer.render(scene, camera);

  state.frameCount++;
  state.fpsAccumTime += deltaMs;
  if (state.fpsAccumTime >= 400) {
    state.fps = Math.round((state.frameCount * 1000) / state.fpsAccumTime);
    state.frameTimeMs = +(state.fpsAccumTime / state.frameCount).toFixed(1);
    state.frameCount = 0;
    state.fpsAccumTime = 0;

    const fpsEl = document.getElementById('hud-fps');
    const ftEl = document.getElementById('hud-frametime');
    if (fpsEl) fpsEl.textContent = String(state.fps);
    if (ftEl) ftEl.textContent = `${state.frameTimeMs} ms`;
  }
}

// Expose automation & test inspection API on window
window.AARNA_PROTO = {
  getState: () => ({
    currentModelKey: state.currentModelKey,
    assistantState: state.assistantState,
    activeExpression: state.activeExpression,
    activeGesture: state.activeGesture,
    outfitKey: state.outfitKey,
    outfitMaterialHex:
      outfitMaterials.length > 0 && outfitMaterials[0].color
        ? outfitMaterials[0].color.getHexString()
        : null,
    outfitMaterialCount: outfitMaterials.length,
    cameraPreset: state.cameraPreset,
    rendererPaused: state.rendererPaused || state.autoPausedByVisibility,
    fps: state.fps,
    frameTimeMs: state.frameTimeMs,
    triangleCount: state.triangleCount || 0,
    drawCalls: renderer.info.render.calls,
    textures: renderer.info.memory.textures,
    geometries: renderer.info.memory.geometries,
    characterLoaded: Boolean(currentAvatarRoot),
    characterHeight: state.characterHeight,
    audioLevel: state.audioLevel,
    smoothAudioLevel: +(state.smoothAudioLevel || 0).toFixed(3),
    hairSpringMagnitude: +(
      Math.abs(state.hairSpringX) +
      Math.abs(state.hairSpringY) +
      Math.abs(state.hairSpringZ)
    ).toFixed(4),
  }),
  setState: (s) => setAssistantState(s),
  setExpression: (e) => setActiveExpression(e),
  triggerGesture: (g) => setActiveGesture(g),
  setAudioLevel: (lvl) => {
    state.audioLevel = Math.max(0.0, Math.min(1.0, Number(lvl) || 0.0));
  },
  triggerVisemeDemo: () => {
    state.visemeDemoActive = true;
    state.visemeDemoStart = performance.now() / 1000;
  },
  setOutfit: (o) => applyOutfitSwatch(o),
  setCamera: (c) => applyCameraPreset(c),
  setPaused: (p) => setRendererPaused(Boolean(p), 'manual'),
  loadModel: (k) => loadCharacterModel(k),
  startVideoRecording: (durationMs = 7500) => {
    window.__aarnaRecordedVideoB64 = null;
    window.__aarnaRecordingDone = false;
    try {
      const stream = canvas.captureStream(30);
      const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9'
        : 'video/webm';
      const recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 3800000 });
      const chunks = [];
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: mime });
        const reader = new FileReader();
        reader.onloadend = () => {
          const dataUrl = reader.result || '';
          const commaIdx = dataUrl.indexOf(',');
          window.__aarnaRecordedVideoB64 = commaIdx >= 0 ? dataUrl.slice(commaIdx + 1) : '';
          window.__aarnaRecordingDone = true;
        };
        reader.readAsDataURL(blob);
      };
      recorder.start(100);
      setTimeout(() => {
        if (recorder.state !== 'inactive') recorder.stop();
      }, durationMs);
      return true;
    } catch (err) {
      console.error('Recording error:', err);
      window.__aarnaRecordingDone = true;
      return false;
    }
  },
};

// Bootstrap prototype
bindUiControls();
loadCharacterModel('aarna_glb').then(() => {
  state.lastFrameTimestamp = performance.now();
  requestAnimationFrame(animate);
});
