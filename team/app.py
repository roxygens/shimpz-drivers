"""Hosted Team driver entrypoint."""

from http_boundary.hosted_controller import Handler, _BoundedThreadingHTTPServer, main

__all__ = ["Handler", "_BoundedThreadingHTTPServer", "main"]


if __name__ == "__main__":
    main()
