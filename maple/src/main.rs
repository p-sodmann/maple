use tracing_subscriber::EnvFilter;

fn main() {
    // Logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    tracing::info!("Maple starting");

    // Boot the Slint UI (blocks until the window closes).
    if let Err(e) = maple_ui::run() {
        tracing::error!("Maple exited with error: {e:#}");
        std::process::exit(1);
    }
}
