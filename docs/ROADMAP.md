# Project AARNA — Development Roadmap

This document outlines the phased engineering milestones for Project AARNA. Each phase builds upon the verified foundation of preceding phases.

---

## Phase Overview

| Phase | Title | Status | Scope |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Workspace & Architecture | **COMPLETED** | Environment setup, modular directory tree, config, docs, test harness. |
| **Phase 1** | Local AI Desktop Assistant | **COMPLETED** | Native PySide6 desktop app, Ollama client, non-blocking UI, multi-turn chat. |
| **Phase 2** | Natural Voice I/O | Planned | Speech-to-text (Faster-Whisper), text-to-speech (Piper). |
| **Phase 3** | Global Wake-Word Activation ("Aarna!") | Planned | Custom-trained local wake word for standalone "Aarna" (NO "Hey" required). |
| **Phase 4** | Persistent Personal Memory | Planned | SQLite memory store, semantic recall, user preferences. |
| **Phase 5** | Desktop Automation & Computer Control | Planned | App launcher, window manager, PyAutoGUI control with permission tiers. |
| **Phase 6** | Internet Research & Tools | Planned | Allowlisted web search, summarizer, weather, local tools. |
| **Phase 7** | Vision & Screen Understanding | Planned | Screenshot analysis, webcam input, OpenCV integration. |
| **Phase 8** | Proactive Assistance | Planned | Smart proactive notifications, scheduled check-ins. |
| **Phase 9** | Background & System Tray Integration | Planned | Minimize to tray, start with Windows, global hotkeys. |
| **Phase 10** | Executable Packaging & Installer | Planned | Standalone PyInstaller build, Inno Setup installer. |
| **Phase 11** | Mobile Companion App | Planned | Local network companion app for Android/iOS. |
| **Phase 12** | Advanced Integrations | Planned | Home automation, cross-device sync, model fine-tuning. |

---

## Acceptance Criteria Status

### Phase 0: Workspace & Architecture Setup (Verified)
- [x] Hardware verification (RTX 4050 6GB VRAM, Ryzen 7 7445HS, Windows 11).
- [x] Python virtual environment created (`.venv`).
- [x] Project structure and modular directory trees initialized.
- [x] Configuration template (`.env.example`) and `.gitignore` created.
- [x] Assistant naming and wake-word locked: "AARNA", exact wake word "Aarna", activation "Aarna!", prefix "Hey" forbidden as a requirement.
- [x] Custom wake-word training requirement researched and documented.
- [x] System Architecture document (`docs/ARCHITECTURE.md`) published.
- [x] Automated test framework initialized (`pytest`).

### Phase 1: First Working Local AI Desktop Assistant (Verified)
- [x] PySide6 native window with futuristic AARNA dark theme.
- [x] Non-blocking asynchronous worker thread (`QThread`) for AI inference.
- [x] Ollama API client with connection status detection and error recovery.
- [x] Multi-turn conversation context manager.
- [x] Message feed with distinct user vs. assistant styling.
- [x] Clean empty-input prevention and input sanitization.
- [x] Informative offline banner and troubleshooting guide when Ollama is unreachable.
- [x] Automated unit test suite passing with 15/15 tests (100% pass rate).
- [x] Live end-to-end verification with Qwen 2.5 3B accelerated on NVIDIA GeForce RTX 4050 Laptop GPU.
- [x] Clean application shutdown without orphaned threads.
