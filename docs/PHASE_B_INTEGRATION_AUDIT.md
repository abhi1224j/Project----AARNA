# AARNA — Phase B: Desktop Avatar Integration Audit

**Document Status:** Phase B Engineering Audit (Pre-Implementation — Awaiting Approval)  
**Target Repository:** `C:\Users\Abhi\Desktop\Project -- AARNA`  
**Audited Baseline:** Phase 1.1 Main Desktop App (`src/`) + Approved Phase A v2 Avatar Prototype (`prototypes/avatar_phase_a/`)  
**Scope:** Architecture, integration design, failure isolation, resource budgeting, and file-by-file plan. **Zero application source files were modified during this audit.**

---

## 1. Current Architecture Findings

### 1.1 Main Desktop Application (`src/`)
The existing Phase 1 / Phase 1.1 native Windows desktop application is cleanly structured into three decoupled layers:

| Module | File Path | Current Responsibilities & Key Details |
| :--- | :--- | :--- |
| **Master Entrypoint** | `src/main.py` | Configures `Qt.HighDpiScaleFactorRoundingPolicy.PassThrough`, creates `QApplication`, sets default font (`Segoe UI`, 10pt), loads `AppConfig`, and launches `AarnaMainWindow`. |
| **Configuration** | `src/core/config.py` | Frozen dataclass `AppConfig` loading `.env` / environment variables with safe defaults: `app_name="AARNA"`, `wake_word="Aarna"`, `wake_word_activation_phrase="Aarna!"`, `require_wake_word_prefix=False`, `ollama_base_url="http://localhost:11434"`, `default_model="qwen2.5:3b"`, `request_timeout=60.0`, `temperature=0.7`, `max_context_messages=20`. |
| **System Time Service** | `src/core/time_service.py` | `TimeService` retrieving OS-local aware time via `datetime.now().astimezone()`. Resolves direct time/date/Indian-city queries deterministically via `try_resolve_exact_time_query()` and injects `[WORKSTATION GROUND TRUTH CLOCK]` into the LLM system prompt via `build_system_context()`. |
| **Ollama Client** | `src/ai/client.py` | `OllamaClient` communicating with `http://localhost:11434` (`/api/version`, `/api/tags`, `/api/chat`). Supports streaming token generation (`chat_stream`) and typed exceptions (`OllamaConnectionError`, `OllamaModelNotFoundError`, `OllamaTimeoutError`, `OllamaError`). |
| **Conversation Manager** | `src/ai/conversation.py` | `ConversationManager` maintaining in-memory `ChatMessage` history (`max_history=20`), prepending `JARVIS_SYSTEM_PROMPT` (`src/ai/prompts.py`) plus live workstation clock context. |
| **Async Workers** | `src/ui/workers.py` | `ChatWorker(QThread)` (signals: `chunk_received(str)`, `finished_response(str)`, `error_occurred(str)`, `status_update(str)`) and `StatusCheckWorker(QThread)` (signal: `status_result(bool, str, list)`). Ensures zero network/inference blocking on the main Qt GUI thread. |
| **Main Window & Theme** | `src/ui/main_window.py`, `src/ui/theme.py` | `AarnaMainWindow(QMainWindow)` (`960×720` default, `720×520` minimum). Single-column `QVBoxLayout` containing: (1) `_create_header()`, (2) `_create_banner()` (offline warning), (3) `_create_chat_area()` (`QScrollArea` with `MessageBubble` widgets), (4) `status_indicator`, (5) `_create_input_area()` (`PromptInputEdit` + `TRANSMIT` button), and (6) `_create_diagnostics_footer()`. Styled with `JARVIS_STYLESHEET` (`#070a13` / `#0b1120` / `#111c33` dark slate surfaces, `#00f0ff` cyan accents). |

### 1.2 Approved Avatar Prototype (`prototypes/avatar_phase_a/`)
The Phase A v2 prototype is a self-contained, 100% offline 3D WebGL application located in `prototypes/avatar_phase_a/`:

| Component | File Path | Architecture & Runtime Characteristics |
| :--- | :--- | :--- |
| **Prototype Launcher & HTTP Server** | `prototypes/avatar_phase_a/run_prototype.py` | Runs `QuietStaticHandler(SimpleHTTPRequestHandler)` on `ThreadingHTTPServer(("127.0.0.1", 0))` (daemon thread, ephemeral loopback port, `Cache-Control: no-store` headers, MIME types for `.js`, `.mjs`, `.glb`, `.vrm`). Hosts a standalone `QWebEngineView` with `QWebEngineProfile.HttpCacheType.NoCache` and automatic render pause on `Qt.WindowMinimized`. |
| **3D Scene & Animation Engine** | `prototypes/avatar_phase_a/app.js` | Built on bundled offline ES modules (`vendor/three.module.min.js` r169, `vendor/addons/loaders/GLTFLoader.js`, `vendor/addons/controls/OrbitControls.js`, `vendor/three-vrm.module.min.js` v3.3.2). Renders deep-space starfield (`650` stars), `55` rising champagne-gold particles, holographic pedestal rings, 5-point warm portrait + champagne-gold rim lighting, 67-bone procedural skeletal animation, 72-morph facial expression/viseme blending, 4 frock swatches (`babypink` default, `lavender`, `peach`, `skyblue`), 4 camera presets (`full`, `side`, `upper`, `face`), 30/60 FPS cap, and GPU pause. |
| **Redesigned v2 Character Model** | `prototypes/avatar_phase_a/models/aarna_teen_casual.glb` | `5.51 MB` glTF 2.0 binary (`16,954` triangles, `10` meshes, `67` humanoid joints, `72` ARKit/Oculus morph targets, `13` embedded textures). Native default appearance is the approved 15–16 yr feminine teenage girl wearing the **Baby Pink** floral frock with rose-pink waist sash/bow, rose-pink hair bow clip, and Baby Pink Mary Jane shoes. |
| **Prototype UI Shell** | `prototypes/avatar_phase_a/index.html`, `style.css` | Includes standalone diagnostic overlays (`#top-bar`, `#left-panel`, `#right-panel`, `#bottom-bar`) surrounding `<canvas id="avatar-canvas">`. |

### 1.3 How Three.js Renders and Communicates with PySide6
1. **Rendering Pipeline**:
   - `QWebEngineView` embeds Chromium's hardware-accelerated WebGL 2 canvas (`THREE.WebGLRenderer` with `ACESFilmicToneMapping` and `SRGBColorSpace`).
   - An internal `requestAnimationFrame(animate)` loop drives `controls.update()`, `updateFacialAnimation(elapsed, dt)`, `updateSkeletalAnimation(elapsed, dt)`, `updateEnvironment(elapsed, dt)`, and `renderer.render(scene, camera)` at 60 FPS (or 30 FPS eco mode).
2. **Current Python $\leftrightarrow$ JS Bridge**:
   - `app.js` exposes a global automation object `window.AARNA_PROTO` with:
     - `getState()` $\rightarrow$ returns a JSON-serializable state snapshot (`characterLoaded`, `assistantState`, `activeExpression`, `activeGesture`, `outfitKey`, `cameraPreset`, `fps`, `frameTimeMs`, `triangleCount`, `rendererPaused`, etc.).
     - `setState(s)` (`'idle'` | `'listening'`), `setExpression(e)` (`'cheerful'` | `'happy'` | `'neutral'` | `'curious'` | `'thinking'` | `'concerned'`), `triggerGesture(g)` (`'idle'` | `'wave'` | `'tilt'` | `'nod'` | `'hands'` | `'speaking'`), `triggerVisemeDemo()`, `setOutfit(o)`, `setCamera(c)`, `setPaused(p)`, and `loadModel(k)`.
   - Python invokes these methods asynchronously via `self.web_view.page().runJavaScript(...)`.

---

## 2. Environment & Qt WebEngine Compatibility Verification

We inspected the project's virtual environment (`C:\Users\Abhi\Desktop\Project -- AARNA\.venv`) directly:

- **Python Version**: `3.13.9 (tags/v3.13.9:8183fa5, Oct 14 2025, MSC v.1944 64 bit (AMD64))`
- **PySide6 Version**: `6.12.0` (`PySide6-Essentials 6.12.0`, `PySide6-Addons 6.12.0`)
- **`PySide6.QtWebEngineWidgets` (`QWebEngineView`)**: **Installed & Verified Working**
- **`PySide6.QtWebEngineCore` (`QWebEnginePage`, `QWebEngineProfile`, `QWebEngineSettings`)**: **Installed & Verified Working**
- **`PySide6.QtWebChannel` (`QWebChannel`)**: **Installed & Verified Working**
- **Hardware Acceleration**: Verified on the host **NVIDIA GeForce RTX 4050 Laptop GPU** (`6,141 MB` VRAM) at locked 60 FPS (`16.7 ms` frame time, `13–14%` GPU utilization).

### Required Dependencies
No new external packages or paid services are needed. `PySide6>=6.8.0` in `pyproject.toml` and `requirements.txt` already installs `QtWebEngineWidgets`, `QtWebEngineCore`, and `QtWebChannel` on Windows.

---

## 3. Recommended Integration Architecture

```mermaid
flowchart LR
    subgraph MainWindow ["AarnaMainWindow (PySide6 QMainWindow)"]
        Header["Top Header Bar\n(Title, Model Selector, Avatar Toggle, Status Pill, Clear)"]
        Banner["Offline Warning Banner (Conditional)"]
        subgraph Splitter ["Main Content Area (QSplitter Horizontal)"]
            subgraph LeftStage ["Left Pane: AarnaAvatarWidget (QFrame)"]
                Stack["QStackedWidget (Graceful Degradation)"]
                WebView["Tier 1: QWebEngineView\n(Full-Body 3D AARNA Stage)"]
                FallbackCard["Tier 2: Native Qt Fallback Card\n(Offline/Headless/Error Mode)"]
                MiniControls["Compact Bottom Stage Bar\n(Camera: Full/Upper/Face | Frock Swatch | 60/30 FPS | Pause)"]
            end
            subgraph RightChat ["Right Pane: Chat & Command Console"]
                ChatScroll["QScrollArea (MessageBubble Stream)"]
                StatusInd["Reasoning / State Indicator"]
                InputBar["PromptInputEdit + TRANSMIT Button"]
            end
        end
        Footer["Diagnostics Footer (GPU, Ollama Status, Avatar State & FPS)"]
    end

    subgraph Workers ["Background Threads"]
        ChatW["ChatWorker (QThread)"]
        StatusW["StatusCheckWorker (QThread)"]
        AssetSrv["LocalAssetServer (127.0.0.1:0 Daemon Thread)"]
    end

    ChatW -->|Qt Signals (Main Thread)| MainWindow
    StatusW -->|Qt Signals (Main Thread)| MainWindow
    MainWindow -->|AvatarState Transitions| LeftStage
    LeftStage -->|Async runJavaScript Queue| WebView
    AssetSrv -->|Read-Only HTTP (No-Cache)| WebView
```

### 3.1 Preserving `prototypes/avatar_phase_a/` Untouched
To guarantee that `prototypes/avatar_phase_a/` is never overwritten or damaged while avoiding duplicating the `5.51 MB` `.glb` binary and `vendor/` bundles:
1. Create a new package `src/ui/avatar/` containing:
   - `src/ui/avatar/__init__.py`
   - `src/ui/avatar/state.py` — `AvatarState` enum and state-to-animation mapping rules.
   - `src/ui/avatar/server.py` — `LocalAssetServer`: a thread-safe, loopback-only (`127.0.0.1:0`) static HTTP server that serves `prototypes/avatar_phase_a/` **strictly read-only**, plus a virtual `/embedded.html` route (or query parameter `?embedded=1`) that injects a clean embedded CSS stylesheet hiding the standalone prototype's debug sidebars (`#top-bar`, `#left-panel`, `#right-panel`, `#bottom-bar`) so only the 3D `<canvas id="avatar-canvas">` and a subtle state pill are displayed inside `AarnaAvatarWidget`.
   - `src/ui/avatar/widget.py` — `AarnaAvatarWidget(QFrame)`: self-contained PySide6 widget encapsulating the `QStackedWidget` (`QWebEngineView` + native Qt fallback widget), a compact native Qt stage control bar underneath the canvas, the command queue, readiness watchdog timer, and clean shutdown lifecycle.

> [!TIP]
> **Why Serving `?embedded=1` / `/embedded.html` over Read-Only `prototypes/avatar_phase_a/` is Ideal:**
> - **Zero modifications** are required inside `prototypes/avatar_phase_a/`.
> - **Zero file duplication** of the `5.51 MB` `.glb` models and Three.js vendor modules.
> - The standalone prototype (`python prototypes/avatar_phase_a/run_prototype.py`) continues to work identically with all of its full debug panels intact, while the embedded desktop view renders a clean, distraction-free full-body 3D stage.

---

## 4. Proposed UI Layout

### 4.1 Window Dimensions & Responsive Split View
- **Default Window Size**: Expand `AarnaMainWindow` default size from `960×720` to **`1240×780`** (minimum size `880×580`), preserving full usability on 1080p/125%-DPI laptop displays.
- **Horizontal `QSplitter` Layout**:
  - **Left Pane (`AarnaAvatarWidget`, ~42% width / `440–500 px`, `min-width: 340 px`)**:
    - Framed in `COLOR_BG_DARKEST` (`#070a13`) with a subtle `1px solid #1e2c4a` border matching `JARVIS_STYLESHEET`.
    - **Top Overlay / Header Strip**: Displays `"AARNA • VISUAL CORE"` and current state badge (`IDLE`, `LISTENING ("Aarna!")`, `THINKING`, `SPEAKING`, `OFFLINE`).
    - **Center 3D Stage (`QWebEngineView`, `stretch=1`)**: Full-body AARNA in her default **Baby Pink** frock standing on the champagne-gold pedestal against the deep space starfield. Vertical portrait aspect ratio (`~460×600`) naturally frames the full-body character large and clear from head to Mary Jane shoes.
    - **Compact Bottom Stage Toolbar (`36 px` height)**: Native Qt controls for quick user adjustments without cluttering the 3D view:
      - Camera Framing selector (`Full Body` [default], `Upper Body`, `Face Close-Up`)
      - Frock Colour selector (`Baby Pink` [default], `Lavender`, `Soft Peach`, `Sky Blue`)
      - Power mode button (`60 FPS` / `30 FPS` / `Pause 3D`)
  - **Right Pane (`Chat & Command Console`, ~58% width / `700–760 px`)**:
    - Preserves the **exact existing Phase 1 / 1.1 chat hierarchy**:
      - Connection warning banner (`self.banner_frame`)
      - Scrollable message bubble feed (`self.scroll_area` / `self.chat_container`)
      - Reasoning indicator (`self.status_indicator`)
      - Prompt input console (`self.input_edit` + `self.send_btn`)
  - **Header Toggle Button (`"Hide Avatar"` / `"Show Avatar"`)**:
    - Added to `_create_header()` next to `Clear` so the user can collapse or expand the 3D avatar pane at any time with a single click.

---

## 5. Python–JavaScript Communication Design & Avatar State Machine

### 5.1 Non-Blocking Asynchronous Command Bridge
All communication from Python (`AarnaAvatarWidget`) to JavaScript (`window.AARNA_PROTO`) follows three strict rules:
1. **Main-Thread Only**: Only methods on `AarnaAvatarWidget` (invoked on the main Qt GUI thread via signals/slots) call `page().runJavaScript(...)`.
2. **Readiness Handshake & Command Queue**:
   - While `QWebEngineView` is loading `index.html` and parsing `aarna_teen_casual.glb`, `self._is_avatar_ready` is `False`.
   - A lightweight readiness poll (`QTimer` firing every `200 ms`, up to an `8.0 s` watchdog timeout) checks `window.AARNA_PROTO && window.AARNA_PROTO.getState().characterLoaded`.
   - Any state transition requested before `_is_avatar_ready == True` is stored as `self._pending_state`, and flushed immediately once `characterLoaded` becomes `True`.
3. **Token-Stream Debouncing (Critical Bottleneck Prevention)**:
   - During `ChatWorker.chunk_received`, Python must **never** call `runJavaScript` on every token chunk (which fires 40–70 times per second).
   - Instead:
     - When `transmit_message()` starts `ChatWorker`, `AarnaAvatarWidget.set_state(AvatarState.THINKING)` is called **once**.
     - When the **first** non-empty chunk arrives in `_on_chunk_received`, `AarnaAvatarWidget.set_state(AvatarState.SPEAKING)` is called **once**.
     - While remaining in `SPEAKING`, a `QTimer` inside `AarnaAvatarWidget` re-triggers `window.AARNA_PROTO.triggerVisemeDemo()` / `'speaking'` gesture at most once every `3.0 seconds` if the streaming response is long.
     - When `_on_finished_response` fires, `AarnaAvatarWidget.set_state(AvatarState.IDLE)` is called **once** (triggering a brief `1.2 s` affirmative `nod` before settling into `idle`).

### 5.2 Complete Avatar State Machine (`AvatarState`)

| State (`AvatarState`) | Trigger Event in `AarnaMainWindow` | Expression (`setExpression`) | Gesture (`triggerGesture`) | Assistant State (`setState`) | Visual & Behavioral Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`GREETING`** | Initial model load completes on app startup (`characterLoaded == True`) | `happy` | `wave` | `idle` | Warm welcome smile + friendly right-hand greeting wave (`3.2 s`), then auto-transitions to `IDLE`. |
| **`IDLE`** | Default standby state after startup, or after `SPEAKING` / `ERROR` completes | `cheerful` | `idle` | `idle` | Lively 3-harmonic contraposto weight shifts, chest/shoulder breathing, 2nd-order head drift, stochastic double-blinks, auto eye gaze, hair spring physics. |
| **`LISTENING`** | User focuses/types in prompt input or triggers `"Aarna!"` listening mode | `curious` | `tilt` (or `wave` on wake word) | `listening` | Attentive ~4.5° head tilt, brightened eyes, golden-amber pedestal & rim-light pulse. |
| **`THINKING`** | `transmit_message()` dispatches prompt to `ChatWorker` (waiting for first token) | `thinking` | `tilt` | `listening` | Reflective upward-left eye gaze, contemplative brow/mouth posture while Qwen 2.5 3B processes prompt. |
| **`SPEAKING`** | First token chunk arrives in `_on_chunk_received()` (or deterministic `TimeService` reply) | `happy` | `speaking` (+ `triggerVisemeDemo()`) | `idle` | Conversational two-hand/arm emphasis cadence, gentle head nods, and procedural jaw/lip viseme articulation during response output. *(Note: Procedural viseme animation during text streaming; real audio-synchronized lip-sync will connect in Phase 2 TTS).* |
| **`ERROR`** | `ChatWorker.error_occurred` or Ollama status check returns `OFFLINE` | `concerned` | `tilt` | `idle` | Empathetic inner-brow lift and gentle concerned tilt; automatically relaxes back to `IDLE` after `4.0 s`. |

---

## 6. GPU/CPU Resource Usage & Performance Considerations

### 6.1 Measured Hardware Budget on User's Laptop
**Workstation Specs:** Windows 11, AMD Ryzen 7 7445HS, 16 GB RAM, **NVIDIA GeForce RTX 4050 Laptop GPU (`6,141 MB` VRAM)**.

| Subsystem | VRAM Allocation | GPU Compute | CPU Usage | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Windows 11 Desktop + DWM** | `~470 MB` | `0–1%` | `<1%` | Baseline before launching AARNA |
| **PySide6 Main Window + 3D Avatar (`QWebEngineView` @ 60 FPS)** | **`+335 to +403 MB`** | **`13–14%`** (`~6%` at 30 FPS; `0%` when paused) | `~2–3%` | `16,954` triangles, `11–16` draw calls, `1024×1024` textures |
| **Ollama `qwen2.5:3b` (Q4_K_M + 4K Context)** | `~2,200 to 2,500 MB` | `45–75%` (only during active token generation) | `~4–8%` | Idle VRAM footprint when model stays loaded in Ollama |
| **Combined Peak Total** | **`~3,070 to 3,375 MB / 6,141 MB` (`50–55%`)** | Safe concurrent operation | `<12%` | Leaves **`>2,750 MB` free VRAM** for future Phase 2 Whisper STT + Piper TTS |

### 6.2 Performance Optimizations for Integrated Desktop Mode
1. **Pixel Ratio Capping**: In embedded desktop mode (`~460×600` logical viewport), cap `renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))` so fill-rate stays ultra-low even during concurrent LLM token streaming.
2. **Automatic GPU Pause on Minimize / Hide**:
   - Hook `AarnaMainWindow.changeEvent` (`Qt.WindowMinimized`) and the `"Hide Avatar"` splitter collapse to call `avatar_widget.set_paused(True)`, dropping WebGL GPU utilization to **`0.0%`** whenever the avatar is not visible.
3. **Optional Automatic Eco-Throttling During Heavy Inference**:
   - Provide a toggle (`30 FPS Eco` / `60 FPS`) in the avatar stage bar so users on battery power can halve GPU usage with one click.

---

## 7. Risks, Threading, Lifecycle & Failure Handling

### 7.1 Identified Technical Risks & Mitigations

| Risk Area | Detailed Failure Mechanism | Engineered Mitigation |
| :--- | :--- | :--- |
| **1. Headless / Offscreen Pytest Execution (`QT_QPA_PLATFORM=offscreen`)** | `tests/test_gui.py` sets `os.environ["QT_QPA_PLATFORM"] = "offscreen"`. Instantiating a live Chromium `QWebEngineView` inside an in-process `offscreen` test session on Windows can hang on teardown or fail to create an OpenGL/DXGI surface. | `AarnaAvatarWidget` inspects `os.environ.get("QT_QPA_PLATFORM") == "offscreen"` (and `config.enable_avatar`). In offscreen test mode, it automatically runs in **Lightweight Native Mode** (updating its Python `AvatarState` state machine and rendering the native Qt Tier-2 card without spawning a Chromium subprocess), while dedicated subprocess tests in `test_avatar_prototype.py` / `test_avatar_integration.py` test live WebGL rendering. |
| **2. Threading Violations (`QThread` $\rightarrow$ `QWebEngineView`)** | Calling `QWebEngineView` methods from `ChatWorker` or `StatusCheckWorker` background threads causes immediate Qt thread-affinity crashes. | `ChatWorker` and `StatusCheckWorker` never reference `AarnaAvatarWidget`. Only `AarnaMainWindow` slots executing on the main Qt GUI thread call `self.avatar_widget.set_state(...)`. |
| **3. Application Shutdown & Orphaned Processes (`closeEvent`)** | Closing `AarnaMainWindow` while `QWebEnginePage` is rendering WebGL or while `LocalAssetServer` is listening can leave background threads alive or cause `ConnectionAbortedError` spam. | `AarnaMainWindow.closeEvent()` calls `self.avatar_widget.shutdown()`, which halts `requestAnimationFrame`, navigates `QWebEnginePage` to `about:blank`, stops the readiness `QTimer`, and shuts down `LocalAssetServer` cleanly before `event.accept()`. |
| **4. Local File Access & CORS Security** | Chromium blocks ES module `import` and `fetch('./models/aarna_teen_casual.glb')` over `file://` URLs. | `LocalAssetServer` binds strictly to `127.0.0.1:0` (loopback only, unreachable from LAN), disables directory listing, rejects `..` path traversal, sets `Cache-Control: no-store`, and `QWebEngineSettings.LocalContentCanAccessRemoteUrls` is set to `False`. |
| **5. WebGL Context Loss or Missing Model File** | GPU driver reset, corrupted `.glb`, or missing `QtWebEngineProcess.exe` could prevent the 3D scene from loading. | **Multi-Tier Graceful Degradation (`QStackedWidget`)**: If `QWebEngineView` fails to import, `loadFinished(False)` fires, or `characterLoaded` does not become `True` within `8.0 seconds`, `AarnaAvatarWidget` automatically switches its `QStackedWidget` to the **Native Qt Fallback Card** (showing AARNA's state badge, a friendly status note, and a `"Retry 3D View"` button). The chat interface, Ollama streaming, and `TimeService` continue operating with **zero interruption**. |

---

## 8. File-by-File Implementation Plan (For Phase B Execution)

Once you approve this audit, Phase B will be implemented across the following exact files:

### 8.1 New Files to Create
1. **`src/ui/avatar/__init__.py`**
   - Exports `AarnaAvatarWidget`, `AvatarState`, and `LocalAssetServer`.
2. **`src/ui/avatar/state.py`**
   - Defines `AvatarState(str, Enum)` (`GREETING`, `IDLE`, `LISTENING`, `THINKING`, `SPEAKING`, `ERROR`).
   - Defines `AVATAR_STATE_SPECS`: deterministic mapping from each `AvatarState` to `(expression, gesture, assistant_state, status_label, auto_return_seconds)`.
3. **`src/ui/avatar/server.py`**
   - Implements `LocalAssetServer`: singleton/managed daemon `ThreadingHTTPServer(("127.0.0.1", 0))` serving `prototypes/avatar_phase_a/` read-only with `Cache-Control: no-store`.
   - Provides a clean `?embedded=1` mode (via a small injected CSS/JS bootstrap or `/embedded.html` endpoint) that hides the standalone prototype's left/right/top/bottom debug panels so only the 3D character canvas is shown.
4. **`src/ui/avatar/widget.py`**
   - Implements `AarnaAvatarWidget(QFrame)`:
     - Header bar with `"AARNA • 3D COMPANION"` and live state badge (`IDLE`, `LISTENING ("Aarna!")`, `THINKING`, `SPEAKING`, `ERROR`).
     - `QStackedWidget` containing Tier 1 (`QWebEngineView`) and Tier 2 (`NativeAvatarFallbackWidget`).
     - Compact bottom toolbar with Camera preset (`Full Body`, `Upper Body`, `Face`), Frock colour (`Baby Pink` default, `Lavender`, `Soft Peach`, `Sky Blue`), and FPS/Pause toggle.
     - Public API: `set_state(state: AvatarState)`, `set_outfit(outfit_key: str)`, `set_camera(preset: str)`, `set_paused(paused: bool)`, `shutdown()`.
5. **`tests/test_avatar_integration.py`**
   - Unit and integration tests verifying `AvatarState` transitions during `transmit_message()`, `TimeService` queries, streaming token chunks, offline errors, fallback mode when WebEngine is disabled/offscreen, and live WebGL embedded rendering.

### 8.2 Existing Files to Modify (Minimally & Non-Destructively)
1. **`src/core/config.py`**
   - Add two optional configuration fields with safe defaults (preserving 100% backward compatibility with existing `test_config.py`):
     - `enable_avatar: bool = True` (`ENABLE_AVATAR` env var, defaults to `True`)
     - `avatar_default_outfit: str = "babypink"` (`AVATAR_DEFAULT_OUTFIT` env var, defaults to `"babypink"`)
2. **`src/ui/theme.py`**
   - Append QSS rules for `QFrame#avatar_stage_frame`, `QFrame#avatar_toolbar`, and `QSplitter::handle` matching the existing dark theme (`#070a13`, `#111c33`, `#1e2c4a`, `#00f0ff`).
3. **`src/ui/main_window.py`**
   - Wrap the central content area in a horizontal `QSplitter`:
     - Left widget: `self.avatar_widget = AarnaAvatarWidget(config=self.config, parent=self)`
     - Right widget: existing chat container (`banner_frame`, `scroll_area`, `status_indicator`, `input_area`)
   - Add `"Hide Avatar"` / `"Show Avatar"` toggle button to `_create_header()`.
   - Connect `transmit_message()`, `_on_chunk_received()`, `_on_finished_response()`, `_on_error_occurred()`, `_on_status_result()`, `changeEvent()`, and `closeEvent()` to `self.avatar_widget`.

### 8.3 Files Guaranteed Untouched
- `prototypes/avatar_phase_a/*` and `prototypes/avatar_phase_a_v1_backup/*` — **100% untouched**
- `src/ai/client.py`, `src/ai/conversation.py`, `src/ai/prompts.py`, `src/core/time_service.py`, `src/ui/workers.py` — **100% untouched**
- `C:\Users\Abhi\Desktop\Glamora` — **100% untouched**

---

## 9. Testing Strategy & Current Baseline Test Results

### 9.1 Current Baseline Test Execution (Audit Verification)
We ran the complete existing `pytest` suite across the repository without modifying any application code:

```text
.\.venv\Scripts\pytest.exe -v
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-9.2.0, pluggy-1.7.0
collected 30 items

tests/test_avatar_prototype.py::test_prototype_files_and_v1_backup_preserved PASSED [  3%]
tests/test_avatar_prototype.py::test_phase1_source_isolation_glamora_and_deliverables_folder PASSED [  6%]
tests/test_avatar_prototype.py::test_character_models_binary_headers_and_features PASSED [ 10%]
tests/test_avatar_prototype.py::test_v2_genuine_3d_facial_and_frock_geometry_redesign PASSED [ 13%]
tests/test_avatar_prototype.py::test_prototype_nomenclature_and_wake_word_compliance PASSED [ 16%]
tests/test_avatar_prototype.py::test_prototype_local_http_server PASSED  [ 20%]
tests/test_avatar_prototype.py::test_prototype_live_webgl_states_expressions_gestures_frock_and_pause PASSED [ 23%]
tests/test_config.py::test_default_config PASSED                         [ 26%]
tests/test_config.py::test_load_from_environment PASSED                  [ 30%]
tests/test_config.py::test_invalid_environment_fallbacks PASSED          [ 33%]
tests/test_conversation.py::test_conversation_empty_init PASSED          [ 36%]
tests/test_conversation.py::test_add_user_and_assistant_messages PASSED  [ 40%]
tests/test_conversation.py::test_reject_empty_user_message PASSED        [ 43%]
tests/test_conversation.py::test_context_window_trimming PASSED          [ 46%]
tests/test_conversation.py::test_clear_conversation PASSED               [ 50%]
tests/test_gui.py::test_main_window_lifecycle_and_offline_mode PASSED    [ 53%]
tests/test_gui.py::test_gui_time_queries_resolution PASSED               [ 56%]
tests/test_ollama_client.py::test_connection_online PASSED               [ 60%]
tests/test_ollama_client.py::test_connection_offline PASSED              [ 63%]
tests/test_ollama_client.py::test_list_models PASSED                     [ 66%]
tests/test_ollama_client.py::test_chat_streaming PASSED                  [ 70%]
tests/test_ollama_client.py::test_chat_model_not_found PASSED            [ 73%]
tests/test_ollama_client.py::test_chat_timeout PASSED                    [ 76%]
tests/test_time_service.py::test_system_local_aware_time PASSED          [ 80%]
tests/test_time_service.py::test_ist_detection_and_chennai_equivalence PASSED [ 83%]
tests/test_time_service.py::test_direct_time_query_resolution PASSED     [ 86%]
tests/test_time_service.py::test_direct_date_query_resolution PASSED     [ 90%]
tests/test_time_service.py::test_non_time_query_returns_none PASSED      [ 93%]
tests/test_time_service.py::test_conversation_injects_live_time_context PASSED [ 96%]
tests/test_time_service.py::test_system_clock_failure_handling PASSED    [100%]

============================= 30 passed in 12.03s =============================
```

> [!NOTE]
> Notice `test_phase1_source_isolation_glamora_and_deliverables_folder` in `tests/test_avatar_prototype.py` currently asserts that `src/*.py` does not contain `"QWebEngineView"` (a guard created during Phase A to ensure Phase A did not prematurely touch `src/`). During Phase B implementation, that specific Phase-A-only isolation assertion in `tests/test_avatar_prototype.py` will be updated to allow `src/ui/avatar/` while keeping `src/ai/` and `src/core/` isolated from UI/WebGL imports, and new Phase B integration tests will be added in `tests/test_avatar_integration.py`.

### 9.2 Planned Phase B Automated Tests (`tests/test_avatar_integration.py`)
1. **State Machine Transition Tests**:
   - Startup $\rightarrow$ `GREETING` $\rightarrow$ `IDLE`.
   - User submits prompt $\rightarrow$ `THINKING` $\rightarrow$ first streaming chunk $\rightarrow$ `SPEAKING` $\rightarrow$ completion $\rightarrow$ `IDLE`.
   - Direct `TimeService` query $\rightarrow$ `SPEAKING` $\rightarrow$ `IDLE`.
   - Ollama offline / `ChatWorker.error_occurred` $\rightarrow$ `ERROR` $\rightarrow$ auto-recovery to `IDLE`.
2. **Graceful Fallback & Toggle Tests**:
   - `enable_avatar=False` or simulated WebEngine load failure cleanly activates the Tier-2 fallback card without breaking `transmit_message()`, `clear_conversation()`, or `TimeService`.
   - Clicking `"Hide Avatar"` / `"Show Avatar"` pauses and resumes the 3D renderer.
3. **Live Embedded WebGL Subprocess Verification**:
   - Launches `AarnaMainWindow` in a real GUI subprocess, verifies the embedded 3D AARNA character loads with `outfitKey == "babypink"`, verifies live state transitions (`THINKING` $\rightarrow$ `SPEAKING` $\rightarrow$ `IDLE`), and captures a verification screenshot of the integrated desktop window.

---

## 10. Rollback Strategy

1. **Runtime Instant Fallback (Zero Code Changes)**:
   - Setting `ENABLE_AVATAR=false` in `.env` (or clicking `"Hide Avatar"` in the header bar) disables the 3D WebEngine view immediately and runs the exact Phase 1.1 chat interface.
2. **Code-Level Rollback**:
   - Because all avatar logic is encapsulated inside `src/ui/avatar/` and connected to `src/ui/main_window.py` through `self.avatar_widget`, reverting `src/ui/main_window.py`, `src/ui/theme.py`, and `src/core/config.py` and removing `src/ui/avatar/` restores the exact Phase 1.1 state in seconds.
   - `prototypes/avatar_phase_a/` remains completely untouched throughout Phase B.

---

## 11. Blockers, Uncertainties & Open Design Decisions

- **Technical Blockers:** **None.** `PySide6 6.12.0` with `QtWebEngineWidgets`, `QtWebEngineCore`, and `QtWebChannel` is already installed and verified at 60 FPS on the RTX 4050 GPU.
- **Honest Scope Boundary for Phase B `SPEAKING` State:**
  - Because voice TTS (`Piper`) is scheduled for the next phase, the `SPEAKING` state in Phase B will animate AARNA's conversational hand/arm gestures, head nods, and procedural mouth visemes synchronized with the **incoming text token stream** from Qwen 2.5 3B (and brief speaking cadence for deterministic `TimeService` answers). Real audio-waveform/phoneme lip-sync will hook into this exact `SPEAKING` state once TTS is added.
- **Recommended Defaults (Awaiting Your Confirmation to Proceed)**:
  1. **Layout Position**: Place the full-body 3D AARNA stage on the **left side** (`~42%` width) and the Chat & Command Console on the **right side** (`~58%` width) inside a resizable horizontal `QSplitter`, with a `"Hide/Show Avatar"` button in the top header.
  2. **Default Startup Appearance**: **Baby Pink** floral frock (`babypink`), **Full-Body** camera framing (`full`), and automatic **Greeting Wave** (`GREETING` $\rightarrow$ `IDLE`) when the desktop window opens.
