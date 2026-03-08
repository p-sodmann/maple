mod app;

use tracing_subscriber::EnvFilter;

fn main() {
    // Logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    tracing::info!("Maple starting");

    // Boot the GTK application (blocks until window closes)
    let exit_code = app::run();
    std::process::exit(exit_code);
}
