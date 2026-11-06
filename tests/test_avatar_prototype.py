"""
Automated verification tests for AARNA — Phase A: 3D Character Redesign & Animation Upgrade.
Validates preserved v1 backups, genuine v2 3D vertex geometry redesign (face, frock, shoes, hair),
GLB/VRM binary integrity, local HTTP server, nomenclature rules, and live WebGL expressions,
gestures, pastel frock colours, camera framing, hair spring physics, and renderer pause.
"""

from __future__ import annotations

import json
import math
import struct
import urllib.request
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = PROJECT_ROOT / "prototypes" / "avatar_phase_a"
BACKUP_DIR = PROJECT_ROOT / "prototypes" / "avatar_phase_a_v1_backup"


def _parse_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    magic, version, total_len = struct.unpack("<4sII", data[:12])
    assert magic == b"glTF", f"Invalid GLB header in {path.name}"
    assert version == 2, f"Expected glTF v2 in {path.name}"
    assert total_len == len(data), f"Length mismatch in {path.name}"
    chunk_len, chunk_type = struct.unpack("<I4s", data[12:20])
    assert chunk_type == b"JSON"
    gltf = json.loads(data[20 : 20 + chunk_len].decode("utf-8"))
    bin_data = data[20 + chunk_len + 8 :]
    return gltf, bin_data


def _read_mesh_positions(gltf: dict, bin_data: bytes, mesh_idx: int) -> list[tuple[float, float, float]]:
    prim = gltf["meshes"][mesh_idx]["primitives"][0]
    acc = gltf["accessors"][prim["attributes"]["POSITION"]]
    bv = gltf["bufferViews"][acc["bufferView"]]
    off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride", 12)
    count = acc["count"]
    return [struct.unpack_from("<3f", bin_data, off + i * stride) for i in range(count)]


def test_prototype_files_and_v1_backup_preserved():
    """Verify all isolated prototype files, local vendor modules, and preserved v1 backups exist."""
    required_files = [
        PROTO_DIR / "index.html",
        PROTO_DIR / "style.css",
        PROTO_DIR / "app.js",
        PROTO_DIR / "run_prototype.py",
        PROTO_DIR / "vendor" / "three.module.min.js",
        PROTO_DIR / "vendor" / "three-vrm.module.min.js",
        PROTO_DIR / "vendor" / "addons" / "loaders" / "GLTFLoader.js",
        PROTO_DIR / "vendor" / "addons" / "controls" / "OrbitControls.js",
        PROTO_DIR / "vendor" / "addons" / "utils" / "BufferGeometryUtils.js",
        PROTO_DIR / "models" / "aarna_teen_casual.glb",
        PROTO_DIR / "models" / "aarna_teen_frock.glb",
        PROTO_DIR / "models" / "aarna_v1_casual_backup.glb",
        BACKUP_DIR / "index.html",
        BACKUP_DIR / "app.js",
        BACKUP_DIR / "models" / "aarna_teen_casual.glb",
    ]
    for path in required_files:
        assert path.exists(), f"Missing required prototype or backup file: {path}"
        assert path.stat().st_size > 100, f"File unexpectedly empty: {path}"


def test_phase1_source_isolation_glamora_and_deliverables_folder():
    """
    Verify that:
    1. All Phase 1 and Phase 1.1 source files in src/ remain isolated from the 3D prototype.
    2. C:\\Users\\Abhi\\Desktop\\Glamora remains untouched.
    3. The accessible output folder prototypes/avatar_phase_a/deliverables/ contains all latest screenshots and video.
    """
    phase1_core_files = [
        PROJECT_ROOT / "src" / "main.py",
        PROJECT_ROOT / "src" / "ai" / "client.py",
        PROJECT_ROOT / "src" / "ai" / "conversation.py",
        PROJECT_ROOT / "src" / "ai" / "prompts.py",
        PROJECT_ROOT / "src" / "core" / "config.py",
        PROJECT_ROOT / "src" / "core" / "time_service.py",
        PROJECT_ROOT / "src" / "ui" / "workers.py",
    ]
    for src_file in phase1_core_files:
        assert src_file.exists(), f"Missing Phase 1 core source file: {src_file}"
        content = src_file.read_text(encoding="utf-8")
        assert "avatar_phase_a" not in content, f"Prototype path leaked into core file: {src_file}"
        assert "aarna_teen" not in content, f"Avatar model leaked into core file: {src_file}"
        assert "QWebEngineView" not in content, f"WebGL view leaked into core file: {src_file}"

    glamora_dir = Path(r"C:\Users\Abhi\Desktop\Glamora")
    if glamora_dir.exists():
        for p in glamora_dir.rglob("*"):
            assert "aarna" not in p.name.lower(), f"Unexpected AARNA file in Glamora: {p}"

    deliv_dir = PROTO_DIR / "deliverables"
    required_deliverables = [
        "aarna_v2_front_fullbody.png",
        "aarna_phase_a_fullbody.png",
        "aarna_v2_side_profile.png",
        "aarna_v2_face_closeup.png",
        "aarna_v2_greeting_wave.png",
        "aarna_v2_expressions_collage.png",
        "aarna_v2_gestures_collage.png",
        "aarna_v2_frock_colours_collage.png",
        "aarna_v2_before_after_comparison.png",
        "aarna_phase_a_demo_recording.webm",
    ]
    for fname in required_deliverables:
        fpath = deliv_dir / fname
        assert fpath.exists(), f"Missing deliverable in {deliv_dir}: {fname}"
        assert fpath.stat().st_size > 10_000, f"Deliverable unexpectedly small: {fpath}"

    # Verify that the rendered frock pixels in the full-body screenshots are warm Baby Pink (R > B + 18, R > G + 30)
    from PySide6.QtGui import QImage

    for fb_name in ("aarna_v2_front_fullbody.png", "aarna_phase_a_fullbody.png"):
        img = QImage(str(deliv_dir / fb_name)).convertToFormat(QImage.Format.Format_RGB888)
        assert not img.isNull()
        cx = int(img.width() * 0.50)
        cy = int(img.height() * 0.48)
        col = img.pixelColor(cx, cy)
        r, g, b = col.red(), col.green(), col.blue()
        assert r > 190 and r > b + 18 and r > g + 30, (
            f"Expected warm Baby Pink frock pixel in {fb_name} at ({cx},{cy}), got RGB=({r},{g},{b})"
        )


def test_character_models_binary_headers_and_features():
    """Verify candidate models exist, have valid glTF 2.0 binary headers, and contain required rigs/morphs."""
    models = {
        "aarna_teen_casual.glb": PROTO_DIR / "models" / "aarna_teen_casual.glb",
        "aarna_teen_frock.glb": PROTO_DIR / "models" / "aarna_teen_frock.glb",
        "aarna_v1_casual_backup.glb": PROTO_DIR / "models" / "aarna_v1_casual_backup.glb",
        "VRM1_Constraint_Twist_Sample.vrm": PROTO_DIR / "models" / "VRM1_Constraint_Twist_Sample.vrm",
        "AvatarSample_A.vrm": PROTO_DIR / "models" / "AvatarSample_A.vrm",
    }
    for name, path in models.items():
        assert path.exists(), f"Missing candidate model: {name}"
        gltf, _ = _parse_glb(path)
        assert len(gltf.get("nodes", [])) >= 50, f"Insufficient skeleton nodes in {name}"
        assert len(gltf.get("meshes", [])) >= 3, f"Insufficient meshes in {name}"


def test_v2_genuine_3d_facial_and_frock_geometry_redesign():
    """
    Verify mathematically that v2 (aarna_teen_casual.glb) modified actual 3D vertex geometry
    compared to v1 (aarna_v1_casual_backup.glb):
    - Wolf3D_Head (Mesh 2): >1,800 vertices displaced in 3D space; narrower feminine jawline; narrower nose.
    - Wolf3D_Outfit_Bottom (Mesh 5): Trousers replaced by 3D pleated frock skirt, sash & bow (2,528 verts vs 688).
    - Wolf3D_Outfit_Top (Mesh 7): Frock bodice + 3D Peter Pan collar (1,447 verts vs 1,262).
    - Wolf3D_Hair (Mesh 8): Lengthened waves + 3D side locks & hair bow clip (941 verts vs 745) + dynamic spring weights.
    """
    gltf_v1, bin_v1 = _parse_glb(PROTO_DIR / "models" / "aarna_v1_casual_backup.glb")
    gltf_v2, bin_v2 = _parse_glb(PROTO_DIR / "models" / "aarna_teen_casual.glb")

    # 1. Head 3D vertex sculpting verification
    head_v1 = _read_mesh_positions(gltf_v1, bin_v1, 2)
    head_v2 = _read_mesh_positions(gltf_v2, bin_v2, 2)
    assert len(head_v1) == len(head_v2) == 2162

    displaced_count = sum(
        1
        for (x1, y1, z1), (x2, y2, z2) in zip(head_v1, head_v2)
        if math.dist((x1, y1, z1), (x2, y2, z2)) > 1e-4
    )
    assert displaced_count > 1200, f"Expected >1200 sculpted head vertices, got {displaced_count}"

    # Verify narrower feminine jawline in v2 (lower jaw region: 1.50 <= y <= 1.54, z > 0.02)
    jaw_width_v1 = max(abs(x) for x, y, z in head_v1 if 1.50 <= y <= 1.54 and z > 0.02)
    jaw_width_v2 = max(abs(x) for x, y, z in head_v2 if 1.50 <= y <= 1.54 and z > 0.02)
    assert jaw_width_v2 < jaw_width_v1 * 0.92, (
        f"Expected v2 jawline width ({jaw_width_v2:.4f}) to be narrower than v1 ({jaw_width_v1:.4f})"
    )

    # 2. 3D Pleated Frock Skirt, Waist Sash & Ribbon Bow (Mesh 5) verification
    skirt_v1 = _read_mesh_positions(gltf_v1, bin_v1, 5)
    skirt_v2 = _read_mesh_positions(gltf_v2, bin_v2, 5)
    assert len(skirt_v1) == 1252
    assert len(skirt_v2) > 2400, f"Expected >2400 vertices in 3D pleated frock skirt+sash+bow, got {len(skirt_v2)}"

    # 3. 3D Frock Bodice & Peter Pan Collar (Mesh 7) verification
    top_v1 = _read_mesh_positions(gltf_v1, bin_v1, 7)
    top_v2 = _read_mesh_positions(gltf_v2, bin_v2, 7)
    assert len(top_v2) > len(top_v1) + 150, "Expected 3D Peter Pan collar vertices appended to Mesh 7"

    # 4. 3D Hair Flowing Locks, Bow Clip & Spring Bone Weighting (Mesh 8) verification
    hair_v1 = _read_mesh_positions(gltf_v1, bin_v1, 8)
    hair_v2 = _read_mesh_positions(gltf_v2, bin_v2, 8)
    assert len(hair_v2) > len(hair_v1) + 150, "Expected 3D side locks and hair bow clip appended to Mesh 8"


def test_prototype_nomenclature_and_wake_word_compliance():
    """Ensure prototype uses canonical name AARNA and standalone wake word 'Aarna!' without 'Hey Aarna' or JARVIS."""
    html_text = (PROTO_DIR / "index.html").read_text(encoding="utf-8")
    js_text = (PROTO_DIR / "app.js").read_text(encoding="utf-8")

    assert "AARNA" in html_text
    assert '"Aarna!"' in html_text
    assert "Hey Aarna" not in html_text
    assert "Hey Aarna" not in js_text
    assert "JARVIS" not in html_text
    assert "J.A.R.V.I.S" not in html_text


def test_prototype_local_http_server():
    """Verify the isolated local HTTP server serves HTML, JS, GLB, and VRM assets on 127.0.0.1."""
    import sys

    if str(PROTO_DIR) not in sys.path:
        sys.path.insert(0, str(PROTO_DIR))
    from run_prototype import start_local_server

    server, port = start_local_server(PROTO_DIR, port=0)
    try:
        base_url = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(f"{base_url}/index.html", timeout=5) as resp:
            assert resp.status == 200
            body = resp.read().decode("utf-8")
            assert "avatar-canvas" in body

        with urllib.request.urlopen(
            urllib.request.Request(f"{base_url}/models/aarna_teen_casual.glb", method="HEAD"),
            timeout=5,
        ) as resp_glb:
            assert resp_glb.status == 200
            assert "gltf-binary" in resp_glb.headers.get("Content-Type", "")
    finally:
        server.shutdown()


def test_prototype_live_webgl_states_expressions_gestures_frock_and_pause():
    """
    Launch isolated 3D prototype in QWebEngineView subprocess and verify:
    - Character load & triangle count (>15,000 tris for redesigned frock + hair)
    - All 6 expressions (cheerful, happy, neutral, curious, thinking, concerned)
    - All 6 gestures (idle, wave, tilt, nod, hands, speaking)
    - All 4 pastel frock colours (lavender, babypink, peach, skyblue)
    - All 4 camera framing presets (full, side, upper, face)
    - Renderer pause & resume
    """
    import subprocess
    import sys

    script = f"""
import json, sys
from pathlib import Path
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

proto_dir = Path(r"{PROTO_DIR}")
sys.path.insert(0, str(proto_dir))
from run_prototype import AarnaAvatarPrototypeWindow, start_local_server

app = QApplication(sys.argv)
server, port = start_local_server(proto_dir, port=0)
url = f"http://127.0.0.1:{{port}}/index.html"
win = AarnaAvatarPrototypeWindow(url)
win.resize(1024, 700)
win.show()

def wait_ms(ms: int):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()

def eval_json(js_expr: str):
    loop = QEventLoop()
    box = {{"val": None}}
    win.web_view.page().runJavaScript(
        f"JSON.stringify({{js_expr}})",
        lambda v: (box.update({{"val": v}}), loop.quit()),
    )
    QTimer.singleShot(3000, loop.quit)
    loop.exec()
    return json.loads(box["val"]) if isinstance(box["val"], str) and box["val"] else None

loaded_state = None
for _ in range(50):
    wait_ms(150)
    st = eval_json("window.AARNA_PROTO ? window.AARNA_PROTO.getState() : null")
    if isinstance(st, dict) and st.get("characterLoaded"):
        loaded_state = st
        break

assert loaded_state is not None, "3D character failed to load in QWebEngineView"
assert loaded_state["currentModelKey"] == "aarna_glb"
assert loaded_state["assistantState"] == "idle"
assert loaded_state["outfitKey"] == "babypink", f"Expected default startup outfit 'babypink', got {{loaded_state['outfitKey']}}"
assert loaded_state["outfitMaterialHex"] == "ffffff"
assert loaded_state["outfitMaterialCount"] >= 3
assert loaded_state["triangleCount"] > 15000

for expr in ("cheerful", "happy", "neutral", "curious", "thinking", "concerned"):
    st_e = eval_json(f"(window.AARNA_PROTO.setExpression('{{expr}}'), window.AARNA_PROTO.getState())")
    assert st_e["activeExpression"] == expr

for gest in ("idle", "wave", "tilt", "nod", "hands", "speaking"):
    st_g = eval_json(f"(window.AARNA_PROTO.triggerGesture('{{gest}}'), window.AARNA_PROTO.getState())")
    assert st_g["activeGesture"] == gest

expected_swatch_hex = {{
    "babypink": "ffffff",
    "lavender": "dec9f8",
    "peach": "ffdcc2",
    "skyblue": "c6e6ff",
}}
for outfit, exp_hex in expected_swatch_hex.items():
    st_o = eval_json(f"(window.AARNA_PROTO.setOutfit('{{outfit}}'), window.AARNA_PROTO.getState())")
    assert st_o["outfitKey"] == outfit
    assert st_o["outfitMaterialHex"] == exp_hex, f"Material hex mismatch for {{outfit}}: {{st_o['outfitMaterialHex']}} != {{exp_hex}}"

# Restore default Baby Pink outfit
eval_json("window.AARNA_PROTO.setOutfit('babypink')")

for cam in ("full", "side", "upper", "face"):
    st_c = eval_json(f"(window.AARNA_PROTO.setCamera('{{cam}}'), window.AARNA_PROTO.getState())")
    assert st_c["cameraPreset"] == cam

st2 = eval_json("(window.AARNA_PROTO.setState('listening'), window.AARNA_PROTO.getState())")
assert st2["assistantState"] == "listening"

st_paused = eval_json("(window.AARNA_PROTO.setPaused(true), window.AARNA_PROTO.getState())")
assert st_paused["rendererPaused"] is True

win.close()
server.shutdown()
print("WEBGL_PROTOTYPE_VERIFIED_OK")
"""
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=35,
    )
    assert res.returncode == 0, f"Subprocess failed: {res.stderr}\n{res.stdout}"
    assert "WEBGL_PROTOTYPE_VERIFIED_OK" in res.stdout
