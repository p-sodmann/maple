//! Global hotkey manager — register named actions with key bindings,
//! attach a single `EventControllerKey` to a widget, and dispatch on press.
//!
//! ```ignore
//! let hk = HotkeyManager::new();
//! hk.register("prev-image", gdk::Key::Left, move || { /* … */ });
//! hk.register("prev-image-alt", gdk::Key::Up, {
//!     let cb = prev_cb.clone();
//!     move || cb()
//! });
//! hk.attach(&some_widget);
//! ```

use gtk4::gdk;
use gtk4::glib;
use gtk4::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

type ActionCallback = Box<dyn Fn() -> glib::Propagation>;

struct Binding {
    name: String,
    callback: ActionCallback,
}

/// A lightweight hotkey dispatcher that maps `gdk::Key` values to named
/// callbacks.  Clone-friendly (cheap `Rc` clone).
#[derive(Clone)]
pub struct HotkeyManager {
    bindings: Rc<RefCell<HashMap<gdk::Key, Binding>>>,
}

impl HotkeyManager {
    pub fn new() -> Self {
        Self {
            bindings: Rc::new(RefCell::new(HashMap::new())),
        }
    }

    /// Register a named action for the given key.
    ///
    /// If the key already has a binding it is silently replaced.
    pub fn register(
        &self,
        name: impl Into<String>,
        key: gdk::Key,
        callback: impl Fn() -> glib::Propagation + 'static,
    ) {
        self.bindings.borrow_mut().insert(
            key,
            Binding {
                name: name.into(),
                callback: Box::new(callback),
            },
        );
    }

    /// Remove a binding by its key.
    pub fn unregister(&self, key: &gdk::Key) {
        self.bindings.borrow_mut().remove(key);
    }

    /// Remove all bindings that were registered under `name`.
    pub fn unregister_by_name(&self, name: &str) {
        self.bindings
            .borrow_mut()
            .retain(|_, b| b.name != name);
    }

    /// Create an `EventControllerKey` wired to this manager and add it
    /// to `widget`.
    pub fn attach(&self, widget: &impl IsA<gtk4::Widget>) {
        let ctrl = gtk4::EventControllerKey::new();
        let bindings = self.bindings.clone();
        ctrl.connect_key_pressed(move |_, keyval, _, _| {
            let map = bindings.borrow();
            if let Some(binding) = map.get(&keyval) {
                (binding.callback)()
            } else {
                glib::Propagation::Proceed
            }
        });
        widget.add_controller(ctrl);
    }
}
