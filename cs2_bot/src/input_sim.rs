//! Input simulation: raw Win32 mouse input and keyboard via enigo.
//!
//! Mouse: raw Win32 `SendInput` with `MOUSEEVENTF_MOVE` (relative) on Windows.
//! Keyboard: uses `enigo` crate for key press/release.
//!
//! On non-Windows platforms, provides stub implementations for development/testing.

use anyhow::Result;
use tracing::debug;

/// Mouse button identifiers.
#[derive(Debug, Clone, Copy)]
pub enum MouseButton {
    Left,
    Right,
}

/// Movement direction for WASD controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveDirection {
    Forward,  // W
    Backward, // S
    Left,     // A
    Right,    // D
}

/// Raw mouse input abstraction.
/// Uses Win32 SendInput on Windows for accurate relative mouse movement.
pub struct RawMouse;

impl RawMouse {
    /// Move the mouse by a relative delta.
    /// Uses `SendInput` with `MOUSEEVENTF_MOVE` on Windows to bypass mouse acceleration.
    pub fn move_relative(dx: i32, dy: i32) -> Result<()> {
        #[cfg(target_os = "windows")]
        {
            use windows::Win32::UI::Input::KeyboardAndMouse::{
                SendInput, INPUT, INPUT_0, INPUT_MOUSE, MOUSEEVENTF_MOVE, MOUSEINPUT,
            };

            let input = INPUT {
                r#type: INPUT_MOUSE,
                Anonymous: INPUT_0 {
                    mi: MOUSEINPUT {
                        dx,
                        dy,
                        mouseData: 0,
                        dwFlags: MOUSEEVENTF_MOVE,
                        time: 0,
                        dwExtraInfo: 0,
                    },
                },
            };

            // SAFETY: Calling Win32 SendInput with a valid INPUT struct.
            // This is the only safe way to send relative mouse movement
            // without mouse acceleration on Windows.
            let result = unsafe { SendInput(&[input], std::mem::size_of::<INPUT>() as i32) };

            if result == 0 {
                anyhow::bail!("SendInput failed for mouse move");
            }

            Ok(())
        }

        #[cfg(not(target_os = "windows"))]
        {
            debug!(dx, dy, "Mouse move (stub)");
            Ok(())
        }
    }

    /// Click a mouse button (press + release).
    pub fn click(button: MouseButton) -> Result<()> {
        #[cfg(target_os = "windows")]
        {
            use windows::Win32::UI::Input::KeyboardAndMouse::{
                SendInput, INPUT, INPUT_0, INPUT_MOUSE, MOUSEEVENTF_LEFTDOWN,
                MOUSEEVENTF_LEFTUP, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP,
                MOUSEINPUT,
            };

            let (down_flag, up_flag) = match button {
                MouseButton::Left => (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
                MouseButton::Right => (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
            };

            let inputs = [
                INPUT {
                    r#type: INPUT_MOUSE,
                    Anonymous: INPUT_0 {
                        mi: MOUSEINPUT {
                            dx: 0,
                            dy: 0,
                            mouseData: 0,
                            dwFlags: down_flag,
                            time: 0,
                            dwExtraInfo: 0,
                        },
                    },
                },
                INPUT {
                    r#type: INPUT_MOUSE,
                    Anonymous: INPUT_0 {
                        mi: MOUSEINPUT {
                            dx: 0,
                            dy: 0,
                            mouseData: 0,
                            dwFlags: up_flag,
                            time: 0,
                            dwExtraInfo: 0,
                        },
                    },
                },
            ];

            // SAFETY: Calling Win32 SendInput with valid INPUT structs for mouse click.
            let result = unsafe { SendInput(&inputs, std::mem::size_of::<INPUT>() as i32) };

            if result == 0 {
                anyhow::bail!("SendInput failed for mouse click");
            }

            Ok(())
        }

        #[cfg(not(target_os = "windows"))]
        {
            debug!(?button, "Mouse click (stub)");
            Ok(())
        }
    }
}

/// Keyboard input abstraction.
/// Uses enigo on Windows, stubs on other platforms.
pub struct KeyboardInput {
    #[cfg(target_os = "windows")]
    enigo: enigo::Enigo,
}

impl KeyboardInput {
    pub fn new() -> Result<Self> {
        #[cfg(target_os = "windows")]
        {
            let settings = enigo::Settings::default();
            let enigo = enigo::Enigo::new(&settings)
                .map_err(|e| anyhow::anyhow!("Failed to create enigo: {e}"))?;
            Ok(Self { enigo })
        }

        #[cfg(not(target_os = "windows"))]
        {
            Ok(Self {})
        }
    }

    /// Press a movement key.
    pub fn press_direction(&mut self, dir: MoveDirection) -> Result<()> {
        #[cfg(target_os = "windows")]
        {
            use enigo::Keyboard;
            let key = direction_to_key(dir);
            self.enigo
                .key(key, enigo::Direction::Press)
                .map_err(|e| anyhow::anyhow!("Key press failed: {e}"))?;
        }

        #[cfg(not(target_os = "windows"))]
        {
            debug!(?dir, "Key press (stub)");
        }

        Ok(())
    }

    /// Release a movement key.
    pub fn release_direction(&mut self, dir: MoveDirection) -> Result<()> {
        #[cfg(target_os = "windows")]
        {
            use enigo::Keyboard;
            let key = direction_to_key(dir);
            self.enigo
                .key(key, enigo::Direction::Release)
                .map_err(|e| anyhow::anyhow!("Key release failed: {e}"))?;
        }

        #[cfg(not(target_os = "windows"))]
        {
            debug!(?dir, "Key release (stub)");
        }

        Ok(())
    }

    /// Release all movement keys.
    pub fn release_all_movement(&mut self) -> Result<()> {
        self.release_direction(MoveDirection::Forward)?;
        self.release_direction(MoveDirection::Backward)?;
        self.release_direction(MoveDirection::Left)?;
        self.release_direction(MoveDirection::Right)?;
        Ok(())
    }
}

/// Map movement direction to keyboard key (Windows only).
#[cfg(target_os = "windows")]
fn direction_to_key(dir: MoveDirection) -> enigo::Key {
    match dir {
        MoveDirection::Forward => enigo::Key::Unicode('w'),
        MoveDirection::Backward => enigo::Key::Unicode('s'),
        MoveDirection::Left => enigo::Key::Unicode('a'),
        MoveDirection::Right => enigo::Key::Unicode('d'),
    }
}
