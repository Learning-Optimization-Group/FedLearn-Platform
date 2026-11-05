"""Operator-selected IP binding for the gradient training listener."""
import ipaddress


def grpc_bind_host(value):
    """Keep the existing default; refuse malformed explicit values instead of widening scope."""
    if value is None:
        return "[::]"
    try:
        address = value.strip()
        if address.startswith("[") and address.endswith("]"):
            address = address[1:-1]
        parsed = ipaddress.ip_address(address)
    except (AttributeError, ValueError) as error:
        raise ValueError("FEDLEARN_GRPC_BIND_ADDRESS must be a numeric bind IP without a port") from error
    return f"[{parsed}]" if parsed.version == 6 else str(parsed)
