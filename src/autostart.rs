use anyhow::Result;

const APP_NAME: &str = "nic-assistant";
const RUN_KEY:  &str = r"Software\Microsoft\Windows\CurrentVersion\Run";

/// Returns true if the autostart entry exists in HKCU\...\Run.
pub fn is_enabled() -> bool {
    #[cfg(windows)]
    {
        use winreg::enums::HKEY_CURRENT_USER;
        use winreg::RegKey;
        let hkcu = RegKey::predef(HKEY_CURRENT_USER);
        if let Ok(key) = hkcu.open_subkey(RUN_KEY) {
            return key.get_value::<String, _>(APP_NAME).is_ok();
        }
        false
    }
    #[cfg(not(windows))]
    { false }
}

/// Writes `<exe_path> --headless` into HKCU\...\Run.
/// Requires no admin rights.
pub fn enable() -> Result<()> {
    #[cfg(windows)]
    {
        use winreg::enums::{HKEY_CURRENT_USER, KEY_SET_VALUE};
        use winreg::RegKey;
        let exe = std::env::current_exe()?;
        let value = format!("\"{}\" --headless", exe.display());
        let hkcu  = RegKey::predef(HKEY_CURRENT_USER);
        let (key, _) = hkcu.create_subkey_with_flags(RUN_KEY, KEY_SET_VALUE)?;
        key.set_value(APP_NAME, &value)?;
    }
    Ok(())
}

/// Removes the autostart entry if it exists.
pub fn disable() -> Result<()> {
    #[cfg(windows)]
    {
        use winreg::enums::{HKEY_CURRENT_USER, KEY_SET_VALUE};
        use winreg::RegKey;
        let hkcu = RegKey::predef(HKEY_CURRENT_USER);
        if let Ok(key) = hkcu.open_subkey_with_flags(RUN_KEY, KEY_SET_VALUE) {
            let _ = key.delete_value(APP_NAME);
        }
    }
    Ok(())
}
