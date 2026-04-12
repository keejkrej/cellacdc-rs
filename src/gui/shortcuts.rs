use super::actions::default_shortcut_binding;
use super::state::{GuiActionId, ShortcutBinding, ShortcutOverrides};
use anyhow::{bail, Result};
use eframe::egui::{self, Event, Key, KeyboardShortcut, Modifiers};
use std::collections::BTreeMap;

pub(crate) fn shortcut_for_action(
    overrides: &ShortcutOverrides,
    action: GuiActionId,
) -> Option<KeyboardShortcut> {
    let binding = overrides
        .bindings
        .iter()
        .find(|override_binding| override_binding.action == action)
        .map(|entry| entry.binding.clone())
        .or_else(|| default_shortcut_binding(action));
    binding.and_then(|binding| binding_to_shortcut(&binding))
}

pub(crate) fn binding_to_shortcut(binding: &ShortcutBinding) -> Option<KeyboardShortcut> {
    let key = key_from_name(&binding.key)?;
    let modifiers = modifiers_from_binding(binding);
    Some(KeyboardShortcut::new(modifiers, key))
}

pub(crate) fn key_from_name(name: &str) -> Option<Key> {
    match name.to_ascii_uppercase().as_str() {
        "A" => Some(Key::A),
        "B" => Some(Key::B),
        "C" => Some(Key::C),
        "D" => Some(Key::D),
        "E" => Some(Key::E),
        "F" => Some(Key::F),
        "G" => Some(Key::G),
        "H" => Some(Key::H),
        "I" => Some(Key::I),
        "J" => Some(Key::J),
        "K" => Some(Key::K),
        "L" => Some(Key::L),
        "M" => Some(Key::M),
        "N" => Some(Key::N),
        "O" => Some(Key::O),
        "P" => Some(Key::P),
        "Q" => Some(Key::Q),
        "R" => Some(Key::R),
        "S" => Some(Key::S),
        "T" => Some(Key::T),
        "U" => Some(Key::U),
        "V" => Some(Key::V),
        "W" => Some(Key::W),
        "X" => Some(Key::X),
        "Y" => Some(Key::Y),
        "Z" => Some(Key::Z),
        "0" => Some(Key::Num0),
        "1" => Some(Key::Num1),
        "2" => Some(Key::Num2),
        "3" => Some(Key::Num3),
        "4" => Some(Key::Num4),
        "5" => Some(Key::Num5),
        "6" => Some(Key::Num6),
        "7" => Some(Key::Num7),
        "8" => Some(Key::Num8),
        "9" => Some(Key::Num9),
        "ARROWUP" | "UP" => Some(Key::ArrowUp),
        "ARROWDOWN" | "DOWN" => Some(Key::ArrowDown),
        "ARROWLEFT" | "LEFT" => Some(Key::ArrowLeft),
        "ARROWRIGHT" | "RIGHT" => Some(Key::ArrowRight),
        "DELETE" => Some(Key::Delete),
        "BACKSPACE" => Some(Key::Backspace),
        "SPACE" => Some(Key::Space),
        "PLUS" => Some(Key::Plus),
        "MINUS" => Some(Key::Minus),
        _ => None,
    }
}

fn key_name(key: Key) -> &'static str {
    match key {
        Key::A => "A",
        Key::B => "B",
        Key::C => "C",
        Key::D => "D",
        Key::E => "E",
        Key::F => "F",
        Key::G => "G",
        Key::H => "H",
        Key::I => "I",
        Key::J => "J",
        Key::K => "K",
        Key::L => "L",
        Key::M => "M",
        Key::N => "N",
        Key::O => "O",
        Key::P => "P",
        Key::Q => "Q",
        Key::R => "R",
        Key::S => "S",
        Key::T => "T",
        Key::U => "U",
        Key::V => "V",
        Key::W => "W",
        Key::X => "X",
        Key::Y => "Y",
        Key::Z => "Z",
        Key::Num0 => "0",
        Key::Num1 => "1",
        Key::Num2 => "2",
        Key::Num3 => "3",
        Key::Num4 => "4",
        Key::Num5 => "5",
        Key::Num6 => "6",
        Key::Num7 => "7",
        Key::Num8 => "8",
        Key::Num9 => "9",
        Key::ArrowUp => "Up",
        Key::ArrowDown => "Down",
        Key::ArrowLeft => "Left",
        Key::ArrowRight => "Right",
        Key::Delete => "Delete",
        Key::Backspace => "Backspace",
        Key::Space => "Space",
        Key::Plus => "Plus",
        Key::Minus => "Minus",
        _ => "Unknown",
    }
}

fn modifiers_from_binding(binding: &ShortcutBinding) -> Modifiers {
    Modifiers {
        alt: binding.alt,
        ctrl: binding.command && !cfg!(target_os = "macos"),
        shift: binding.shift,
        mac_cmd: binding.command && cfg!(target_os = "macos"),
        command: binding.command,
    }
}

pub(crate) fn display_binding(binding: &ShortcutBinding) -> String {
    let mut parts = Vec::new();
    if binding.command {
        parts.push(if cfg!(target_os = "macos") {
            "Cmd"
        } else {
            "Ctrl"
        });
    }
    if binding.alt {
        parts.push("Alt");
    }
    if binding.shift {
        parts.push("Shift");
    }
    parts.push(binding.key.as_str());
    parts.join("+")
}

pub(crate) fn shortcut_label(overrides: &ShortcutOverrides, action: GuiActionId) -> Option<String> {
    let binding = overrides
        .bindings
        .iter()
        .find(|entry| entry.action == action)
        .map(|entry| entry.binding.clone())
        .or_else(|| default_shortcut_binding(action))?;
    Some(display_binding(&binding))
}

pub(crate) fn detect_shortcut_capture(ctx: &egui::Context) -> Option<ShortcutBinding> {
    let events = ctx.input(|input| input.events.clone());
    for event in events {
        if let Event::Key {
            key,
            pressed: true,
            repeat: false,
            modifiers,
            ..
        } = event
        {
            if matches!(
                key,
                Key::ArrowDown
                    | Key::ArrowLeft
                    | Key::ArrowRight
                    | Key::ArrowUp
                    | Key::Backspace
                    | Key::Delete
                    | Key::Space
                    | Key::Plus
                    | Key::Minus
                    | Key::A
                    | Key::B
                    | Key::C
                    | Key::D
                    | Key::E
                    | Key::F
                    | Key::G
                    | Key::H
                    | Key::I
                    | Key::J
                    | Key::K
                    | Key::L
                    | Key::M
                    | Key::N
                    | Key::O
                    | Key::P
                    | Key::Q
                    | Key::R
                    | Key::S
                    | Key::T
                    | Key::U
                    | Key::V
                    | Key::W
                    | Key::X
                    | Key::Y
                    | Key::Z
                    | Key::Num0
                    | Key::Num1
                    | Key::Num2
                    | Key::Num3
                    | Key::Num4
                    | Key::Num5
                    | Key::Num6
                    | Key::Num7
                    | Key::Num8
                    | Key::Num9
            ) {
                return Some(ShortcutBinding {
                    key: key_name(key).to_string(),
                    command: modifiers.command,
                    shift: modifiers.shift,
                    alt: modifiers.alt,
                });
            }
        }
    }
    None
}

pub(crate) fn set_override(
    overrides: &mut ShortcutOverrides,
    action: GuiActionId,
    binding: ShortcutBinding,
) {
    if let Some(entry) = overrides
        .bindings
        .iter_mut()
        .find(|entry| entry.action == action)
    {
        entry.binding = binding;
    } else {
        overrides
            .bindings
            .push(super::state::ShortcutOverride { action, binding });
    }
}

pub(crate) fn validate_shortcut_overrides(overrides: &ShortcutOverrides) -> Result<()> {
    let mut seen = BTreeMap::<String, GuiActionId>::new();
    for entry in &overrides.bindings {
        if binding_to_shortcut(&entry.binding).is_none() {
            bail!("Unsupported shortcut binding for {:?}", entry.action);
        }
        let key = display_binding(&entry.binding);
        if let Some(existing) = seen.insert(key.clone(), entry.action) {
            bail!(
                "Shortcut collision: {} is assigned to both {:?} and {:?}",
                key,
                existing,
                entry.action
            );
        }
    }
    Ok(())
}

pub(crate) fn triggered_action(
    ctx: &egui::Context,
    overrides: &ShortcutOverrides,
    actions: &[GuiActionId],
) -> Option<GuiActionId> {
    for action in actions {
        let shortcut = shortcut_for_action(overrides, *action)?;
        let consumed = ctx.input_mut(|input| input.consume_shortcut(&shortcut));
        if consumed {
            return Some(*action);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::state::{ShortcutOverride, ShortcutOverrides};

    #[test]
    fn rejects_duplicate_overrides() {
        let overrides = ShortcutOverrides {
            bindings: vec![
                ShortcutOverride {
                    action: GuiActionId::Save,
                    binding: ShortcutBinding {
                        key: "S".to_string(),
                        command: true,
                        shift: false,
                        alt: false,
                    },
                },
                ShortcutOverride {
                    action: GuiActionId::QuickSave,
                    binding: ShortcutBinding {
                        key: "S".to_string(),
                        command: true,
                        shift: false,
                        alt: false,
                    },
                },
            ],
        };
        assert!(validate_shortcut_overrides(&overrides).is_err());
    }
}
