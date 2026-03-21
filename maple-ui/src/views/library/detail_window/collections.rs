//! Collection chips bar and "add to collection" window for the detail view.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use gtk4::gdk;
use gtk4::prelude::*;
use libadwaita as adw;
use adw::prelude::*;

use maple_db::LibraryImage;

/// Shared state needed by the collection UI pieces.
#[derive(Clone)]
pub(super) struct CollectionBar {
    pub chips: gtk4::FlowBox,
    pub last_collection_id: Rc<Cell<Option<i64>>>,
    current_image: Rc<RefCell<LibraryImage>>,
    db: Arc<Mutex<maple_db::Database>>,
    toast_overlay: adw::ToastOverlay,
    /// Reference to the open picker's list box (if any).
    picker_list: Rc<RefCell<Option<gtk4::ListBox>>>,
}

impl CollectionBar {
    pub fn new(
        current_image: Rc<RefCell<LibraryImage>>,
        db: Arc<Mutex<maple_db::Database>>,
        toast_overlay: adw::ToastOverlay,
    ) -> Self {
        let chips = gtk4::FlowBox::builder()
            .selection_mode(gtk4::SelectionMode::None)
            .max_children_per_line(20)
            .column_spacing(6)
            .row_spacing(4)
            .margin_start(8)
            .margin_end(8)
            .margin_top(4)
            .margin_bottom(4)
            .homogeneous(false)
            .build();

        Self {
            chips,
            last_collection_id: Rc::new(Cell::new(None)),
            current_image,
            db,
            toast_overlay,
            picker_list: Rc::new(RefCell::new(None)),
        }
    }

    /// Reload collection chips for the current image.
    pub fn reload(&self) {
        // Remove all children.
        while let Some(child) = self.chips.first_child() {
            self.chips.remove(&child);
        }

        let image_id = self.current_image.borrow().id;
        let collections = self
            .db
            .lock()
            .ok()
            .and_then(|d| d.collections_for_image(image_id).ok())
            .unwrap_or_default();

        for coll in &collections {
            let chip = self.build_chip(coll);
            self.chips.insert(&chip, -1);
        }

        self.rebuild_add_button();

        // Refresh the picker window if it's open.
        if let Some(ref list_box) = *self.picker_list.borrow() {
            if list_box.is_mapped() {
                self.populate_picker_list(list_box);
            }
        }
    }

    /// Add the image to a collection by id, show a toast, and refresh chips.
    pub fn add_to_collection(&self, coll_id: i64) {
        let image_id = self.current_image.borrow().id;
        let coll_name = self
            .db
            .lock()
            .ok()
            .and_then(|d| d.collection_by_id(coll_id).ok().flatten())
            .map(|c| c.name);

        if let Ok(d) = self.db.lock() {
            let _ = d.add_image_to_collection(coll_id, image_id);
        }

        let name = coll_name.unwrap_or_else(|| "collection".into());
        let toast = adw::Toast::new(&format!("Added to \"{name}\""));
        toast.set_timeout(2);
        self.toast_overlay.add_toast(toast);

        self.reload();
    }

    /// Populate (or repopulate) a picker list box with collection rows for the
    /// current image.
    fn populate_picker_list(&self, list_box: &gtk4::ListBox) {
        // Clear existing rows.
        while let Some(child) = list_box.first_child() {
            list_box.remove(&child);
        }

        let all_collections = self
            .db
            .lock()
            .ok()
            .and_then(|d| d.all_collections().ok())
            .unwrap_or_default();

        let image_id = self.current_image.borrow().id;
        let current_ids: Vec<i64> = self
            .db
            .lock()
            .ok()
            .and_then(|d| d.collections_for_image(image_id).ok())
            .unwrap_or_default()
            .iter()
            .map(|c| c.id)
            .collect();

        for coll in &all_collections {
            let check = gtk4::CheckButton::builder()
                .active(current_ids.contains(&coll.id))
                .build();

            let row = adw::ActionRow::builder()
                .title(&coll.name)
                .activatable_widget(&check)
                .build();
            row.add_prefix(&check);
            list_box.append(&row);

            let coll_id = coll.id;
            let coll_name = coll.name.clone();
            let bar_clone = self.clone();
            check.connect_toggled(move |btn| {
                let image_id = bar_clone.current_image.borrow().id;
                if let Ok(d) = bar_clone.db.lock() {
                    if btn.is_active() {
                        let _ = d.add_image_to_collection(coll_id, image_id);
                        bar_clone.last_collection_id.set(Some(coll_id));
                        let toast = adw::Toast::new(&format!("Added to \"{}\"", coll_name));
                        toast.set_timeout(2);
                        bar_clone.toast_overlay.add_toast(toast);
                    } else {
                        let _ = d.remove_image_from_collection(coll_id, image_id);
                    }
                }
                bar_clone.reload();
            });
        }

        // "Manage Collections…" row at bottom.
        if !all_collections.is_empty() {
            let sep = gtk4::Separator::new(gtk4::Orientation::Horizontal);
            list_box.append(&sep);
        }

        let manage_row = adw::ActionRow::builder()
            .title("Manage Collections…")
            .activatable(true)
            .build();
        list_box.append(&manage_row);

        let bar_clone = self.clone();
        list_box.connect_row_activated(move |_, row| {
            if let Some(action_row) = row.downcast_ref::<adw::ActionRow>() {
                if action_row.title() == "Manage Collections…" {
                    let db = bar_clone.db.clone();
                    let bar_inner = bar_clone.clone();
                    if let Some(win) = row.root().and_then(|r| r.downcast::<gtk4::Window>().ok()) {
                        super::super::collection_manager::open_manager(
                            &win,
                            &db,
                            move || {
                                bar_inner.reload();
                            },
                        );
                    }
                }
            }
        });
    }

    fn build_chip(&self, coll: &maple_db::Collection) -> gtk4::Box {
        let chip_box = gtk4::Box::builder()
            .orientation(gtk4::Orientation::Horizontal)
            .spacing(4)
            .css_classes(["card"])
            .build();
        chip_box.set_margin_start(2);
        chip_box.set_margin_end(2);
        chip_box.set_margin_top(2);
        chip_box.set_margin_bottom(2);

        // Color dot.
        let dot = gtk4::DrawingArea::builder()
            .content_width(10)
            .content_height(10)
            .margin_start(10)
            .valign(gtk4::Align::Center)
            .build();
        let hex = coll.color.clone();
        dot.set_draw_func(move |_, cr, w, h| {
            if let Ok(rgba) = gdk::RGBA::parse(&hex) {
                cr.set_source_rgba(
                    rgba.red() as f64,
                    rgba.green() as f64,
                    rgba.blue() as f64,
                    1.0,
                );
                let r = w.min(h) as f64 / 2.0;
                cr.arc(w as f64 / 2.0, h as f64 / 2.0, r, 0.0, 2.0 * std::f64::consts::PI);
                let _ = cr.fill();
            }
        });

        let label = gtk4::Label::new(Some(&coll.name));
        label.set_margin_start(4);
        label.set_margin_end(2);
        label.add_css_class("caption");

        // Remove "x" button.
        let remove_btn = gtk4::Button::builder()
            .icon_name("window-close-symbolic")
            .css_classes(["flat", "circular"])
            .valign(gtk4::Align::Center)
            .build();
        remove_btn.set_margin_end(2);

        chip_box.append(&dot);
        chip_box.append(&label);
        chip_box.append(&remove_btn);

        let coll_id = coll.id;
        let bar = self.clone();
        remove_btn.connect_clicked(move |_| {
            let image_id = bar.current_image.borrow().id;
            if let Ok(d) = bar.db.lock() {
                let _ = d.remove_image_from_collection(coll_id, image_id);
            }
            bar.reload();
        });

        chip_box
    }

    /// Add a trailing "+" button that opens the collection window.
    fn rebuild_add_button(&self) {
        // Remove old add button if present.
        let mut child = self.chips.first_child();
        while let Some(c) = child {
            let next = c.next_sibling();
            if c.widget_name() == "add-chip-wrapper" {
                self.chips.remove(&c);
            }
            child = next;
        }

        let add_btn = gtk4::Button::builder()
            .icon_name("list-add-symbolic")
            .css_classes(["flat", "circular"])
            .tooltip_text("Add to collection")
            .build();

        let bar = self.clone();
        add_btn.connect_clicked(move |btn| {
            let parent_window = btn
                .root()
                .and_then(|r| r.downcast::<gtk4::Window>().ok());
            if let Some(win) = parent_window {
                open_picker(&bar, &win);
            }
        });

        let wrapper = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
        wrapper.set_widget_name("add-chip-wrapper");
        wrapper.append(&add_btn);

        self.chips.insert(&wrapper, -1);
    }
}

/// Open a window listing all collections with checkboxes for the current image.
pub(super) fn open_picker(bar: &CollectionBar, parent: &gtk4::Window) {
    // If an existing picker window is still open, just present it.
    if let Some(ref list_box) = *bar.picker_list.borrow() {
        if list_box.is_mapped() {
            if let Some(win) = list_box.root().and_then(|r| r.downcast::<gtk4::Window>().ok()) {
                win.present();
                return;
            }
        }
    }

    let list_box = gtk4::ListBox::builder()
        .selection_mode(gtk4::SelectionMode::None)
        .css_classes(["boxed-list"])
        .build();

    bar.populate_picker_list(&list_box);

    // Store reference so reload() can refresh it.
    *bar.picker_list.borrow_mut() = Some(list_box.clone());

    let scroll = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .max_content_height(300)
        .propagate_natural_height(true)
        .vexpand(true)
        .build();
    scroll.set_child(Some(&list_box));

    let header = adw::HeaderBar::new();

    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&scroll));

    let win = adw::Window::builder()
        .title("Collections")
        .default_width(320)
        .default_height(400)
        .transient_for(parent)
        .build();
    win.set_content(Some(&toolbar_view));

    // Clear the stored reference when the picker window closes.
    let picker_list = bar.picker_list.clone();
    win.connect_destroy(move |_| {
        *picker_list.borrow_mut() = None;
    });

    win.present();
}
