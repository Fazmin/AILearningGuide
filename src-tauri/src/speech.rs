use std::path::PathBuf;
use std::process::Child;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[cfg(target_os = "macos")]
use std::io::Write;
#[cfg(any(target_os = "macos", target_os = "windows"))]
use std::process::{Command, Stdio};

struct SpeechJob {
    child: Child,
    cleanup: Option<PathBuf>,
}

#[derive(Clone)]
pub struct SpeechEngine {
    job: Arc<Mutex<Option<SpeechJob>>>,
}

impl SpeechEngine {
    pub fn new() -> Self {
        Self {
            job: Arc::new(Mutex::new(None)),
        }
    }

    pub fn stop(&self) {
        if let Ok(mut slot) = self.job.lock() {
            if let Some(mut job) = slot.take() {
                let _ = job.child.kill();
                let _ = job.child.wait();
                if let Some(path) = job.cleanup {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
    }

    pub fn speak(&self, text: String) -> Result<(), String> {
        let text = text.trim().to_string();
        if text.is_empty() {
            return Err("Nothing to read.".into());
        }
        if text.chars().count() > 80_000 {
            return Err("This lesson is too long to read in one pass.".into());
        }
        self.stop();
        let mut child = spawn_speaker(&text)?;
        let cleanup = child.cleanup.take();
        {
            let mut slot = self.job.lock().map_err(|_| "Speech lock poisoned".to_string())?;
            *slot = Some(SpeechJob {
                child: child.child,
                cleanup: cleanup.clone(),
            });
        }
        loop {
            std::thread::sleep(Duration::from_millis(80));
            let mut slot = self.job.lock().map_err(|_| "Speech lock poisoned".to_string())?;
            let Some(job) = slot.as_mut() else {
                if let Some(path) = cleanup {
                    let _ = std::fs::remove_file(path);
                }
                return Ok(());
            };
            match job.child.try_wait() {
                Ok(Some(_)) => {
                    if let Some(finished) = slot.take() {
                        if let Some(path) = finished.cleanup {
                            let _ = std::fs::remove_file(path);
                        }
                    }
                    return Ok(());
                }
                Ok(None) => {}
                Err(error) => {
                    slot.take();
                    if let Some(path) = cleanup {
                        let _ = std::fs::remove_file(path);
                    }
                    return Err(error.to_string());
                }
            }
        }
    }
}

struct Spawned {
    child: Child,
    cleanup: Option<PathBuf>,
}

fn spawn_speaker(text: &str) -> Result<Spawned, String> {
    #[cfg(target_os = "macos")]
    {
        let mut child = Command::new("say")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| error.to_string())?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(text.as_bytes()).map_err(|error| error.to_string())?;
        }
        return Ok(Spawned {
            child,
            cleanup: None,
        });
    }

    #[cfg(target_os = "windows")]
    {
        let path = std::env::temp_dir().join(format!(
            "the-ai-guide-speech-{}.txt",
            std::process::id()
        ));
        std::fs::write(&path, text).map_err(|error| error.to_string())?;
        let script = format!(
            "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak([System.IO.File]::ReadAllText({}));",
            powershell_quote(&path.to_string_lossy())
        );
        let child = Command::new("powershell")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-WindowStyle",
                "Hidden",
                "-Command",
                &script,
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| error.to_string())?;
        return Ok(Spawned {
            child,
            cleanup: Some(path),
        });
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = text;
        Err("System speech is not available.".into())
    }
}

#[cfg(any(target_os = "windows", test))]
pub fn powershell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_is_rejected() {
        let engine = SpeechEngine::new();
        assert_eq!(engine.speak("   ".into()).unwrap_err(), "Nothing to read.");
    }

    #[test]
    fn powershell_paths_are_single_quoted() {
        assert_eq!(powershell_quote(r"C:\Temp\guide.txt"), r"'C:\Temp\guide.txt'");
        assert_eq!(powershell_quote("O'Hara"), "'O''Hara'");
    }
}
