"""Block network connections for the length of a run (docs/user_models.md, section 5).

Testing a user's model never sends anything anywhere: the model and images stay on the user's machine. Inside
`with no_network() as attempts:`, every connection or name lookup made through Python's socket module raises
NetworkBlocked, and is also noted in `attempts` (so an attempt that some library catches and hides is still
counted). The block covers Python's socket module; native code that bypasses it is not intercepted, so this is
a check backed by a test, not a sandbox.
"""

import contextlib
import socket


class NetworkBlocked(OSError):
    """Raised for any connection attempt while the network is blocked."""


# Module functions that open connections or look up names, and socket methods that send or connect.
MODULE_FUNCTIONS = ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex")
SOCKET_METHODS = ("connect", "connect_ex", "sendto", "sendmsg")


@contextlib.contextmanager
def no_network():
    attempts = []

    def refuse(name):
        def refused(*args, **kwargs):
            attempts.append(name)
            raise NetworkBlocked(f"brokkr-edge opens no network connections (blocked: {name})")
        return refused

    saved_functions = {name: getattr(socket, name) for name in MODULE_FUNCTIONS}
    # Only the methods this platform has (Windows sockets have no sendmsg).
    saved_methods = {name: getattr(socket.socket, name)
                     for name in SOCKET_METHODS if hasattr(socket.socket, name)}
    own_methods = {name for name in saved_methods if name in vars(socket.socket)}  # the rest are inherited
    try:
        for name in MODULE_FUNCTIONS:
            setattr(socket, name, refuse(f"socket.{name}"))
        for name in saved_methods:
            setattr(socket.socket, name, refuse(f"socket.socket.{name}"))
        yield attempts
    finally:
        for name, f in saved_functions.items():
            setattr(socket, name, f)
        for name, f in saved_methods.items():
            if name in own_methods:
                setattr(socket.socket, name, f)
            else:
                delattr(socket.socket, name)  # inherited again, exactly as before
