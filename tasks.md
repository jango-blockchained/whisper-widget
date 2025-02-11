# GTK Migration Tasks

## Current Findings

1. **GTK 3.0 Components Found:**
   - AppIndicator3 (version 0.1) in whisper_widget/app.py
   - GTK 3.0 bindings in virtual environment's gi module
   - No other GTK 3.0 specific widgets or functions found

2. **Current GTK 4.0 Usage:**
   - Main application window (WhisperWindow)
   - Menu system (Gio.Menu)
   - Window management and events
   - WebKit2 (version 4.1)
   - Gdk (version 4.0)

## Steps for Migration

1. **Replace AppIndicator3:**
   - Switch to Libayatana AppIndicator
   - Implementation steps:
     1. Install system dependency: `libayatana-appindicator3-dev`
     2. Uncomment and update in requirements.txt: `ayatana-appindicator3>=0.5.91`
     3. Update imports in app.py:
        ```python
        gi.require_version('AyatanaAppIndicator3', '0.1')
        from gi.repository import AyatanaAppIndicator3
        ```
     4. Replace AppIndicator3 usage with AyatanaAppIndicator3
     5. Update test mocks accordingly
   - Fallback plan:
     - If Ayatana is not available, minimize to window instead of system tray
     - Add setting to control tray icon behavior

2. **Update Dependencies:**
   - Ensure PyGObject installation uses GTK 4.0
   - Update virtual environment configuration
   - Remove or update any GTK 3.0 specific dependencies
   - Update package versions in requirements.txt

3. **Code Updates Required:**
   - Modify system tray implementation in app.py
   - Update any AppIndicator3 related test mocks
   - Verify all GTK imports use version 4.0

4. **Testing Requirements:**
   - Test system tray functionality with new implementation
   - Verify menu interactions
   - Check window transparency and dragging
   - Validate all existing tests pass with GTK 4.0

5. **Documentation Updates:**
   - Update installation instructions for new dependencies
   - Document any changes in system tray behavior
   - Update version requirements in README.md

## Considerations

- **Backward Compatibility:**
  - Current codebase mostly uses GTK 4.0
  - Main compatibility issue is system tray implementation
  - Consider providing fallback for systems without new indicator support

- **UI Changes:**
  - System tray behavior might change with new implementation
  - Verify transparency and window management still work as expected

- **Performance Improvements:**
  - GTK 4.0 should provide better rendering
  - Monitor system tray performance with new implementation

- **Community and Support:**
  - Research community solutions for GTK 4.0 system tray implementations
  - Monitor GTK 4.0 status notifications API development 