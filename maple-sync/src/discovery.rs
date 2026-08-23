//! Finding a master on the LAN, so nobody has to type an IP address (§2.4).
//!
//! A master advertises `_maple-sync._tcp.local.` with its device id, its
//! display name and the protocol version in TXT records; a servant browses
//! for that service, both while pairing (a pick-list instead of a text
//! field) and afterwards, so a master that changed address heals the link
//! without being re-paired.
//!
//! # Discovery is a convenience, never an authority
//!
//! Anyone on the network can answer a browse, so everything here is
//! *unauthenticated hearsay*: a name and an address, nothing more. It picks
//! who to dial and never who to trust — a fake master advertising a real
//! one's device id gets a connection and then fails the pairing proof or the
//! request MAC, exactly as it would if the user had typed its address by
//! hand. That is why the resolver may only move an **already paired** peer
//! to a new address: the credential does not travel with the record.
//!
//! Manual `host:port` entry stays available and stays the fallback, because
//! plenty of networks block multicast outright.
//!
//! # Threading
//!
//! `mdns-sd` runs its own daemon thread and — this is the part worth
//! remembering — has **no `Drop`**: a `ServiceDaemon` that is simply dropped
//! keeps its thread and its sockets forever. Both handles here own their
//! daemon and shut it down in their own `Drop`, so a role switch cannot
//! leave a stack of listeners multicasting behind the app's back.
//!
//! # Testing
//!
//! Nothing that needs a network is in the interesting code. Reading a TXT
//! record into a [`DiscoveredDevice`] and ranking its addresses are plain
//! functions over plain data, and the consumers — the pairing modal, the
//! sync worker — reach discovery through [`DeviceSource`], which a test
//! implements in five lines. The daemon-backed [`Browser`] and
//! [`Advertiser`] are the only parts that touch a socket, and they are
//! deliberately thin.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};

use crate::protocol::PROTOCOL_VERSION;

/// The service type both halves agree on. Trailing dot included, as DNS-SD
/// requires.
pub const SERVICE_TYPE: &str = "_maple-sync._tcp.local.";

/// TXT record keys. Read by name rather than by position, so a future
/// version can add one without breaking an older browser.
pub mod txt {
    /// The advertiser's `sync_identity.device_id` — the same string the
    /// trust file and `sync_peers` key off.
    pub const DEVICE_ID: &str = "device_id";
    /// Its display name, for the pick-list.
    pub const NAME: &str = "name";
    /// [`crate::protocol::PROTOCOL_VERSION`], as decimal digits.
    pub const PROTOCOL: &str = "protocol";
}

/// How long a shutting-down advertiser waits for its goodbye packet.
///
/// Unregistering multicasts a TTL-0 record that takes the service off every
/// browser on the network at once; without it they keep showing a device
/// that stopped listening until the record expires on its own. Short,
/// because this runs on the UI thread during a role switch and on the way
/// out of the app.
const GOODBYE_WAIT: Duration = Duration::from_millis(400);

/// How long the browse thread blocks before re-checking its stop flag.
const BROWSE_TICK: Duration = Duration::from_millis(400);

/// A master seen on the network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredDevice {
    pub device_id: String,
    pub name: String,
    /// The protocol version it announced. `0` when the record carried none,
    /// which means something older than this field.
    pub protocol: u32,
    /// Every address it can be reached at, best first — see [`rank`]. Each
    /// is a `host:port` ready to hand to [`crate::SyncClient::new`], IPv6
    /// bracketed.
    pub addresses: Vec<String>,
}

impl DiscoveredDevice {
    /// The address to dial. `None` for a record that resolved with nothing
    /// usable in it.
    pub fn address(&self) -> Option<&str> {
        self.addresses.first().map(String::as_str)
    }

    /// Whether this device speaks the version we do.
    ///
    /// An incompatible master is still listed rather than hidden: it is on
    /// the network, the user can see it in every other mDNS tool, and
    /// "Maple shows nothing" is a worse answer than "that one needs
    /// updating".
    pub fn compatible(&self) -> bool {
        self.protocol == PROTOCOL_VERSION
    }

    /// What the pick-list shows.
    pub fn label(&self) -> String {
        let address = self.address().unwrap_or("no address");
        if self.compatible() {
            format!("{} · {address}", self.name)
        } else {
            format!("{} · {address} · needs updating", self.name)
        }
    }
}

/// Where a consumer gets the current list from.
///
/// A trait rather than the concrete [`Browser`] so the pairing modal and the
/// sync worker can be driven by a test that never opens a socket, and so a
/// device with no discovery at all is `None` rather than a stub daemon.
pub trait DeviceSource: Send + Sync {
    /// Everything seen recently, best-labelled first.
    fn devices(&self) -> Vec<DiscoveredDevice>;

    /// Where one particular device is right now.
    ///
    /// This is the reconnect path (§1.4): the worker knows *which* master it
    /// is paired to and needs its current address, not a list.
    fn address_of(&self, device_id: &str) -> Option<String> {
        self.devices()
            .into_iter()
            .find(|device| device.device_id == device_id)
            .and_then(|device| device.addresses.into_iter().next())
    }
}

// ── Reading a record ────────────────────────────────────────────

/// Turn a resolved service's TXT properties and addresses into a device.
///
/// `None` when the record carries no device id: that is the one field with
/// no sensible default, since it is what a pairing is keyed on and what the
/// worker re-resolves against. A missing *name* is survivable and gets the
/// same `maple-xxxxxx` fallback the settings card uses.
pub fn describe(
    properties: &HashMap<String, String>,
    addresses: &[IpAddr],
    port: u16,
) -> Option<DiscoveredDevice> {
    let device_id = properties.get(txt::DEVICE_ID)?.trim();
    if device_id.is_empty() {
        return None;
    }
    let name = properties
        .get(txt::NAME)
        .map(|name| name.trim())
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("maple-{}", &device_id[..device_id.len().min(6)]));

    Some(DiscoveredDevice {
        device_id: device_id.to_owned(),
        name,
        protocol: properties
            .get(txt::PROTOCOL)
            .and_then(|version| version.trim().parse().ok())
            .unwrap_or(0),
        addresses: dialable(addresses, port),
    })
}

/// The addresses worth trying, best first.
///
/// Two things happen here. **IPv6 link-local is dropped**: `fe80::` is only
/// meaningful with a zone index (`%en0`), which neither a stored `host:port`
/// nor `ureq`'s parser carries, so keeping it would produce an address that
/// can only fail. Everything else is *ranked* rather than filtered, because
/// on a machine running two instances for testing, loopback is the address
/// that works — it is simply the last one to try on a real network.
fn dialable(addresses: &[IpAddr], port: u16) -> Vec<String> {
    let mut usable: Vec<&IpAddr> = addresses
        .iter()
        .filter(|ip| !is_link_local_v6(ip))
        .collect();
    usable.sort_by_key(|ip| (rank(ip), ip.to_string()));
    usable.dedup();
    usable.iter().map(|ip| socket_addr(ip, port)).collect()
}

/// Lower sorts first. Ordinary LAN addresses beat autoconfigured ones, which
/// beat loopback.
fn rank(ip: &IpAddr) -> u8 {
    match ip {
        IpAddr::V4(v4) if v4.is_loopback() => 3,
        IpAddr::V4(v4) if v4.is_link_local() => 2,
        IpAddr::V4(_) => 0,
        IpAddr::V6(v6) if v6.is_loopback() => 4,
        IpAddr::V6(_) => 1,
    }
}

fn is_link_local_v6(ip: &IpAddr) -> bool {
    match ip {
        // `Ipv6Addr::is_unicast_link_local` is still unstable, and the test
        // is one mask: fe80::/10.
        IpAddr::V6(v6) => v6.segments()[0] & 0xffc0 == 0xfe80,
        IpAddr::V4(_) => false,
    }
}

/// `host:port`, with the brackets IPv6 needs in a URL.
fn socket_addr(ip: &IpAddr, port: u16) -> String {
    match ip {
        IpAddr::V4(v4) => format!("{v4}:{port}"),
        IpAddr::V6(v6) => format!("[{v6}]:{port}"),
    }
}

/// The DNS-SD instance name this device registers under.
///
/// The display name is what a person recognises in someone else's mDNS
/// browser, and the id suffix is what keeps two laptops both called
/// "MacBook" from colliding — DNS-SD instance names have to be unique within
/// a service type, and a collision is resolved by the *other* device
/// renaming itself, which is not a negotiation to leave to chance.
pub fn instance_name(device_id: &str, name: &str) -> String {
    let name = name.trim();
    let short = &device_id[..device_id.len().min(6)];
    if name.is_empty() {
        format!("maple-{short}")
    } else {
        format!("{name} ({short})")
    }
}

/// The host name the A record is published under.
///
/// Derived from the device id rather than the machine's own host name: it is
/// guaranteed unique, it is stable across a rename, and it does not leak
/// whatever the user called their laptop to the whole network.
pub fn host_name(device_id: &str) -> String {
    format!("maple-{}.local.", &device_id[..device_id.len().min(12)])
}

// ── Advertising ─────────────────────────────────────────────────

/// A master's registration. Dropping it takes the service off the network.
pub struct Advertiser {
    daemon: ServiceDaemon,
    fullname: String,
}

impl Advertiser {
    /// Announce this master on the LAN.
    ///
    /// `port` is the port the listener actually bound, not the one in
    /// settings: `listen_addr` may name port 0, and advertising a port
    /// nothing is listening on is worse than not advertising at all.
    pub fn start(device_id: &str, name: &str, port: u16) -> anyhow::Result<Self> {
        let daemon = ServiceDaemon::new()?;
        let properties = HashMap::from([
            (txt::DEVICE_ID.to_owned(), device_id.to_owned()),
            (txt::NAME.to_owned(), name.to_owned()),
            (txt::PROTOCOL.to_owned(), PROTOCOL_VERSION.to_string()),
        ]);
        let info = ServiceInfo::new(
            SERVICE_TYPE,
            &instance_name(device_id, name),
            &host_name(device_id),
            // Empty, because `enable_addr_auto` fills in every local address
            // and keeps them current. The listener binds `0.0.0.0`, so this
            // crate genuinely does not know which address a servant should
            // use — the daemon does.
            "",
            port,
            properties,
        )?
        .enable_addr_auto();
        let fullname = info.get_fullname().to_owned();
        daemon.register(info)?;
        tracing::info!("sync discovery: advertising {fullname} on port {port}");
        Ok(Self { daemon, fullname })
    }

    pub fn fullname(&self) -> &str {
        &self.fullname
    }
}

impl Drop for Advertiser {
    fn drop(&mut self) {
        // Unregister first and give the goodbye a moment to go out, then
        // shut the daemon down — `ServiceDaemon` has no `Drop` of its own,
        // so skipping this leaks a thread and two sockets per role switch.
        match self.daemon.unregister(&self.fullname) {
            Ok(done) => {
                let _ = done.recv_timeout(GOODBYE_WAIT);
            }
            Err(e) => tracing::warn!("sync discovery: could not unregister: {e}"),
        }
        if let Err(e) = self.daemon.shutdown() {
            tracing::warn!("sync discovery: could not stop advertiser: {e}");
        }
    }
}

// ── Browsing ────────────────────────────────────────────────────

/// A running browse. Dropping it stops the thread and the daemon.
pub struct Browser {
    daemon: ServiceDaemon,
    /// Keyed by DNS-SD fullname, because that is what a removal names. Two
    /// entries can carry one device id — a master that was renamed and
    /// re-registered before its old record expired — and [`devices`] is
    /// where that is collapsed.
    ///
    /// [`devices`]: DeviceSource::devices
    found: Arc<Mutex<HashMap<String, DiscoveredDevice>>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Browser {
    /// Start looking for masters.
    pub fn start() -> anyhow::Result<Self> {
        let daemon = ServiceDaemon::new()?;
        let events = daemon.browse(SERVICE_TYPE)?;
        let found: Arc<Mutex<HashMap<String, DiscoveredDevice>>> = Arc::default();
        let stop = Arc::new(AtomicBool::new(false));

        let thread = std::thread::Builder::new()
            .name("maple-sync-browse".into())
            .spawn({
                let found = found.clone();
                let stop = stop.clone();
                move || {
                    while !stop.load(Ordering::Relaxed) {
                        match events.recv_timeout(BROWSE_TICK) {
                            Ok(ServiceEvent::ServiceResolved(service)) => {
                                let properties = service
                                    .get_properties()
                                    .iter()
                                    .map(|p| (p.key().to_owned(), p.val_str().to_owned()))
                                    .collect();
                                let addresses: Vec<IpAddr> = service
                                    .get_addresses()
                                    .iter()
                                    .map(|ip| ip.to_ip_addr())
                                    .collect();
                                if let Some(device) =
                                    describe(&properties, &addresses, service.get_port())
                                {
                                    tracing::debug!(
                                        "sync discovery: found {} at {}",
                                        device.name,
                                        device.address().unwrap_or("nowhere")
                                    );
                                    lock(&found)
                                        .insert(service.get_fullname().to_owned(), device);
                                }
                            }
                            Ok(ServiceEvent::ServiceRemoved(_, fullname)) => {
                                lock(&found).remove(&fullname);
                            }
                            Ok(_) => {}
                            // The daemon is gone; nothing further will
                            // arrive, so the thread is done.
                            Err(mdns_sd::RecvTimeoutError::Disconnected) => break,
                            Err(mdns_sd::RecvTimeoutError::Timeout) => {}
                        }
                    }
                }
            })?;

        Ok(Self {
            daemon,
            found,
            stop,
            thread: Some(thread),
        })
    }
}

impl DeviceSource for Browser {
    fn devices(&self) -> Vec<DiscoveredDevice> {
        let mut devices: Vec<DiscoveredDevice> = lock(&self.found).values().cloned().collect();
        devices.sort_by(|a, b| {
            (&a.name, &a.device_id)
                .cmp(&(&b.name, &b.device_id))
        });
        devices.dedup_by(|a, b| a.device_id == b.device_id);
        devices
    }
}

impl Drop for Browser {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Err(e) = self.daemon.shutdown() {
            tracing::warn!("sync discovery: could not stop browser: {e}");
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn properties(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    fn v4(text: &str) -> IpAddr {
        text.parse().unwrap()
    }

    #[test]
    fn a_record_becomes_a_device() {
        let device = describe(
            &properties(&[
                (txt::DEVICE_ID, "dev-master"),
                (txt::NAME, "Studio"),
                (txt::PROTOCOL, "1"),
            ]),
            &[v4("192.168.1.20")],
            7645,
        )
        .expect("a record with a device id describes a device");

        assert_eq!(device.device_id, "dev-master");
        assert_eq!(device.name, "Studio");
        assert_eq!(device.address(), Some("192.168.1.20:7645"));
        assert!(device.compatible());
    }

    #[test]
    fn a_record_without_a_device_id_is_not_a_device() {
        // Nothing can be paired with, or re-resolved to, an anonymous
        // record — it is somebody else's `_maple-sync` or a broken one.
        assert!(describe(&properties(&[(txt::NAME, "Studio")]), &[v4("192.168.1.20")], 7645).is_none());
        assert!(describe(
            &properties(&[(txt::DEVICE_ID, "  "), (txt::NAME, "Studio")]),
            &[v4("192.168.1.20")],
            7645
        )
        .is_none());
    }

    #[test]
    fn a_nameless_device_still_lists() {
        let device = describe(
            &properties(&[(txt::DEVICE_ID, "abcdef012345"), (txt::PROTOCOL, "1")]),
            &[v4("10.0.0.4")],
            7645,
        )
        .unwrap();
        assert_eq!(device.name, "maple-abcdef");
    }

    #[test]
    fn an_older_protocol_is_listed_but_flagged() {
        let device = describe(
            &properties(&[(txt::DEVICE_ID, "dev-old"), (txt::NAME, "Attic"), (txt::PROTOCOL, "0")]),
            &[v4("192.168.1.9")],
            7645,
        )
        .unwrap();
        assert!(!device.compatible());
        assert!(device.label().contains("needs updating"), "{}", device.label());
    }

    #[test]
    fn a_lan_address_beats_loopback_and_link_local() {
        let device = describe(
            &properties(&[(txt::DEVICE_ID, "dev-master")]),
            &[
                v4("127.0.0.1"),
                v4("169.254.3.4"),
                v4("192.168.1.20"),
            ],
            7645,
        )
        .unwrap();
        assert_eq!(
            device.addresses,
            vec![
                "192.168.1.20:7645".to_owned(),
                "169.254.3.4:7645".to_owned(),
                "127.0.0.1:7645".to_owned(),
            ]
        );
    }

    #[test]
    fn a_link_local_v6_address_is_dropped_and_a_routable_one_is_bracketed() {
        // fe80:: needs a zone index no `host:port` string carries, so it can
        // only ever fail; a routable v6 address has to survive the trip
        // through a URL, which means brackets.
        let device = describe(
            &properties(&[(txt::DEVICE_ID, "dev-master")]),
            &[
                "fe80::1".parse().unwrap(),
                "2001:db8::7334".parse().unwrap(),
            ],
            7645,
        )
        .unwrap();
        assert_eq!(device.addresses, vec!["[2001:db8::7334]:7645".to_owned()]);
    }

    #[test]
    fn a_device_with_no_usable_address_has_none() {
        let device = describe(
            &properties(&[(txt::DEVICE_ID, "dev-master")]),
            &["fe80::1".parse().unwrap()],
            7645,
        )
        .unwrap();
        assert_eq!(device.address(), None);
        assert!(device.label().contains("no address"));
    }

    #[test]
    fn two_devices_of_the_same_name_get_different_instances() {
        assert_ne!(
            instance_name("aaaaaa111111", "MacBook"),
            instance_name("bbbbbb222222", "MacBook")
        );
        assert_eq!(instance_name("aaaaaa111111", "MacBook"), "MacBook (aaaaaa)");
        assert_eq!(instance_name("aaaaaa111111", "   "), "maple-aaaaaa");
        assert_eq!(host_name("aaaaaa111111zzz"), "maple-aaaaaa111111.local.");
    }

    struct Fake(Vec<DiscoveredDevice>);

    impl DeviceSource for Fake {
        fn devices(&self) -> Vec<DiscoveredDevice> {
            self.0.clone()
        }
    }

    #[test]
    #[ignore = "needs a network that carries multicast; run with --ignored"]
    fn an_advertiser_is_found_and_then_unfound_by_a_browser() {
        // The one test that puts a packet on the wire. Ignored by default —
        // a CI box with multicast filtered would fail it for reasons that
        // have nothing to do with this code — but it is what proves the two
        // daemon-backed halves agree, including that dropping an advertiser
        // really does take it off the network rather than leaving a record
        // to expire.
        let device_id = "dev-live-0123456789";
        let advertiser = Advertiser::start(device_id, "Live Test", 7645).expect("advertise");
        let browser = Browser::start().expect("browse");

        let found = wait_for(&browser, |devices| {
            devices.iter().find(|d| d.device_id == device_id).cloned()
        })
        .expect("the advertiser to be discovered");
        assert_eq!(found.name, "Live Test");
        assert!(found.compatible());
        assert!(
            found.address().is_some_and(|a| a.ends_with(":7645")),
            "resolved with no usable address: {:?}",
            found.addresses
        );

        drop(advertiser);
        assert!(
            wait_for(&browser, |devices| devices
                .iter()
                .all(|d| d.device_id != device_id)
                .then_some(()))
            .is_some(),
            "a goodbye should take the record off every browser at once"
        );
    }

    #[cfg(test)]
    fn wait_for<T>(
        browser: &Browser,
        mut f: impl FnMut(&[DiscoveredDevice]) -> Option<T>,
    ) -> Option<T> {
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        while std::time::Instant::now() < deadline {
            if let Some(hit) = f(&browser.devices()) {
                return Some(hit);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        None
    }

    #[test]
    fn a_source_resolves_one_device_by_id() {
        let source = Fake(vec![
            DiscoveredDevice {
                device_id: "dev-a".into(),
                name: "A".into(),
                protocol: PROTOCOL_VERSION,
                addresses: vec!["192.168.1.4:7645".into()],
            },
            DiscoveredDevice {
                device_id: "dev-b".into(),
                name: "B".into(),
                protocol: PROTOCOL_VERSION,
                addresses: vec!["192.168.1.5:7645".into(), "127.0.0.1:7645".into()],
            },
        ]);
        assert_eq!(source.address_of("dev-b").as_deref(), Some("192.168.1.5:7645"));
        assert_eq!(source.address_of("dev-c"), None);
    }
}
