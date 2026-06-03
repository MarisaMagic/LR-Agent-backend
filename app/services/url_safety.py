import ipaddress
import socket
from urllib.parse import urlparse

from app.core.config import Settings

_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
    },
)


def _is_blocked_ip(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
    )


def _resolve_host_ips(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(hostname, None):
            if family == socket.AF_INET:
                ips.append(ipaddress.ip_address(sockaddr[0]))
            elif family == socket.AF_INET6:
                ips.append(ipaddress.ip_address(sockaddr[0]))
    except socket.gaierror as exc:
        raise ValueError("invalid_base_url_host") from exc
    if not ips:
        raise ValueError("invalid_base_url_host")
    return ips


def validate_llm_base_url(base_url: str, settings: Settings) -> str:
    parsed = urlparse(base_url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("invalid_base_url")

    scheme = parsed.scheme.lower()
    if scheme == "https":
        pass
    elif scheme == "http" and (settings.llm_base_url_allow_http or settings.is_development):
        pass
    else:
        raise ValueError("invalid_base_url_scheme")

    if parsed.username or parsed.password:
        raise ValueError("invalid_base_url_credentials")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("invalid_base_url_host")

    host_lower = hostname.lower()
    if host_lower in _BLOCKED_HOSTNAMES:
        raise ValueError("invalid_base_url_host")

    try:
        literal = ipaddress.ip_address(host_lower)
        if _is_blocked_ip(literal):
            raise ValueError("invalid_base_url_host")
    except ValueError:
        if host_lower.endswith(".localhost"):
            raise ValueError("invalid_base_url_host")
        for addr in _resolve_host_ips(hostname):
            if _is_blocked_ip(addr):
                raise ValueError("invalid_base_url_host")

    path = parsed.path.rstrip("/")
    normalized = f"{scheme}://{parsed.netloc}{path}"
    return normalized
