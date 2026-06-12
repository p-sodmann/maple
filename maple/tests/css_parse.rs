//! Regression test: the application stylesheet must parse cleanly.
//!
//! GTK silently skips invalid CSS rules at runtime, so a typo in
//! `style.css` would otherwise only show up as a visual glitch.

use std::cell::RefCell;
use std::rc::Rc;

#[test]
fn style_css_parses_without_errors() {
    if gtk4::init().is_err() {
        eprintln!("skipping: no display available for GTK");
        return;
    }

    let errors: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let provider = gtk4::CssProvider::new();
    provider.connect_parsing_error({
        let errors = errors.clone();
        move |_, section, error| {
            errors.borrow_mut().push(format!("{section}: {error}"));
        }
    });
    provider.load_from_string(include_str!("../src/style.css"));

    assert!(
        errors.borrow().is_empty(),
        "style.css has parse errors:\n{}",
        errors.borrow().join("\n")
    );
}
