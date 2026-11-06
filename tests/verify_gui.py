import os
import sys
from pathlib import Path
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PySide6.QtWidgets import QApplication
    from src.core.config import AppConfig
    from src.ui.main_window import AarnaMainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    config = AppConfig.load()
    window = AarnaMainWindow(config=config)

    print("WINDOW_TITLE:", window.windowTitle())
    print("CONFIG_MODEL:", window.config.default_model)
    print("WAKE_WORD:", window.config.wake_word)
    print("IS_CONNECTED:", window.is_connected)

    # Test empty message
    window.input_edit.setPlainText("   ")
    window.transmit_message()
    count_1 = window.chat_layout.count()
    print("COUNT_AFTER_EMPTY:", count_1)

    # Test offline message
    window.input_edit.setPlainText("Hello AARNA")
    window.transmit_message()
    count_2 = window.chat_layout.count()
    print("COUNT_AFTER_OFFLINE:", count_2)

    # Test clear
    window.clear_conversation()
    count_3 = window.chat_layout.count()
    print("COUNT_AFTER_CLEAR:", count_3)

    window.close()
    app.processEvents()
    if window.status_worker and window.status_worker.isRunning():
        window.status_worker.wait(4000)
    print("GUI_VERIFICATION_PASSED")
    sys.exit(0)

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
