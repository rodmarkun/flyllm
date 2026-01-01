//! HTTP server for serving Grafana dashboard JSON
//!
//! This module provides an optional HTTP endpoint that serves a pre-built
//! Grafana dashboard JSON for monitoring LLM operations.

use axum::{routing::get, Json, Router};
use std::net::SocketAddr;
use tokio::net::TcpListener;

/// Grafana dashboard JSON (embedded at compile time)
const DASHBOARD_JSON: &str = include_str!("../assets/grafana_dashboard.json");

/// Dashboard server configuration
#[derive(Debug, Clone)]
pub struct DashboardServerConfig {
    /// Address to bind the HTTP server to
    pub bind_address: SocketAddr,
}

impl Default for DashboardServerConfig {
    fn default() -> Self {
        Self {
            bind_address: ([127, 0, 0, 1], 9898).into(),
        }
    }
}

impl DashboardServerConfig {
    /// Create a new configuration with a custom address
    pub fn new(addr: impl Into<SocketAddr>) -> Self {
        Self {
            bind_address: addr.into(),
        }
    }

    /// Create a configuration binding to all interfaces on the specified port
    pub fn with_port(port: u16) -> Self {
        Self {
            bind_address: ([0, 0, 0, 0], port).into(),
        }
    }
}

/// Start the dashboard HTTP server
///
/// This function starts an HTTP server that serves:
/// - `GET /dashboard` - Returns the Grafana dashboard JSON
/// - `GET /health` - Returns "OK" for health checks
///
/// # Example
///
/// ```no_run
/// use flyllm::metrics::dashboard::{DashboardServerConfig, start_dashboard_server};
///
/// #[tokio::main]
/// async fn main() {
///     let config = DashboardServerConfig::default();
///     start_dashboard_server(config).await.unwrap();
/// }
/// ```
pub async fn start_dashboard_server(config: DashboardServerConfig) -> std::io::Result<()> {
    let app = Router::new()
        .route("/dashboard", get(serve_dashboard))
        .route("/health", get(health_check));

    let listener = TcpListener::bind(config.bind_address).await?;
    log::info!(
        "FlyLLM dashboard server listening on http://{}",
        config.bind_address
    );
    log::info!(
        "Dashboard JSON available at http://{}/dashboard",
        config.bind_address
    );

    axum::serve(listener, app).await?;
    Ok(())
}

async fn serve_dashboard() -> Json<serde_json::Value> {
    match serde_json::from_str(DASHBOARD_JSON) {
        Ok(json) => Json(json),
        Err(e) => {
            log::error!("Failed to parse dashboard JSON: {}", e);
            Json(serde_json::json!({
                "error": "Failed to parse dashboard JSON",
                "details": e.to_string()
            }))
        }
    }
}

async fn health_check() -> &'static str {
    "OK"
}
