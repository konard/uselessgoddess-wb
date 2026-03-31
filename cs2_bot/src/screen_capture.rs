//! Screen capture module using the `scap` crate.
//!
//! Provides zero-copy frame grabbing targeting < 2ms latency.
//! On non-supported platforms, returns a placeholder frame for testing.

use anyhow::Result;
use tracing::{debug, info};

/// Raw captured frame data.
pub struct CapturedFrame {
    /// BGRA pixel data.
    pub data: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

/// Screen capturer abstraction.
pub struct ScreenCapturer {
    width: u32,
    height: u32,
    #[cfg(target_os = "windows")]
    capturer: Option<scap::Capturer>,
}

impl ScreenCapturer {
    /// Create a new screen capturer for the given resolution.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        info!(width, height, "Initializing screen capturer");

        #[cfg(target_os = "windows")]
        {
            use scap::{
                capturer::{Area, Capturer, Options, Point, Resolution, Size},
                frame::FrameType,
            };

            let options = Options {
                fps: 60,
                show_cursor: false,
                show_highlight: false,
                excluded_targets: None,
                output_type: FrameType::BGRAFrame,
                output_resolution: Resolution::Custom(width, height),
                source_rect: Some(Area {
                    origin: Point { x: 0.0, y: 0.0 },
                    size: Size {
                        width: width as f64,
                        height: height as f64,
                    },
                }),
                ..Default::default()
            };

            let mut capturer = Capturer::new(options);
            capturer.start_capture();
            info!("Screen capture started (Windows)");

            return Ok(Self {
                width,
                height,
                capturer: Some(capturer),
            });
        }

        #[cfg(not(target_os = "windows"))]
        {
            debug!("Screen capture: using placeholder (non-Windows platform)");
            Ok(Self { width, height })
        }
    }

    /// Grab a single frame. Returns the raw pixel data.
    pub fn grab_frame(&mut self) -> Result<CapturedFrame> {
        #[cfg(target_os = "windows")]
        {
            if let Some(ref mut capturer) = self.capturer {
                let frame = capturer.get_next_frame()?;
                match frame {
                    scap::frame::Frame::BGRA(bgra) => {
                        return Ok(CapturedFrame {
                            data: bgra.data,
                            width: bgra.width as u32,
                            height: bgra.height as u32,
                        });
                    }
                    _ => anyhow::bail!("Unexpected frame format"),
                }
            }
            anyhow::bail!("Capturer not initialized");
        }

        #[cfg(not(target_os = "windows"))]
        {
            // Return a blank frame for testing on non-Windows platforms
            let size = (self.width * self.height * 4) as usize;
            Ok(CapturedFrame {
                data: vec![0u8; size],
                width: self.width,
                height: self.height,
            })
        }
    }

    /// Stop the capture session.
    pub fn stop(&mut self) {
        #[cfg(target_os = "windows")]
        {
            if let Some(ref mut capturer) = self.capturer {
                capturer.stop_capture();
                info!("Screen capture stopped");
            }
        }
    }
}

impl Drop for ScreenCapturer {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Convert a BGRA frame to RGB f32 CHW tensor data normalized to [0, 1].
/// Output shape: [1, 3, height, width] flattened.
pub fn bgra_to_chw_f32(data: &[u8], width: u32, height: u32) -> Vec<f32> {
    let pixels = (width * height) as usize;
    let mut output = vec![0.0f32; 3 * pixels];

    let r_offset = 0;
    let g_offset = pixels;
    let b_offset = 2 * pixels;

    for i in 0..pixels {
        let base = i * 4;
        // BGRA -> RGB
        output[r_offset + i] = data[base + 2] as f32 / 255.0;
        output[g_offset + i] = data[base + 1] as f32 / 255.0;
        output[b_offset + i] = data[base] as f32 / 255.0;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bgra_to_chw() {
        // 2x2 image, all red pixels (BGRA: 0, 0, 255, 255)
        let data = vec![
            0, 0, 255, 255, // pixel 0: blue=0, green=0, red=255, alpha=255
            0, 0, 255, 255, // pixel 1
            0, 0, 255, 255, // pixel 2
            0, 0, 255, 255, // pixel 3
        ];
        let result = bgra_to_chw_f32(&data, 2, 2);
        assert_eq!(result.len(), 3 * 4); // 3 channels * 4 pixels

        // R channel should be all 1.0
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 1.0).abs() < 1e-5);
        // G channel should be all 0.0
        assert!((result[4] - 0.0).abs() < 1e-5);
        // B channel should be all 0.0
        assert!((result[8] - 0.0).abs() < 1e-5);
    }
}
