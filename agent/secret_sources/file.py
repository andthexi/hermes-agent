"""Read a private dotenv file as a bulk secret source; never execute it."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from agent.secret_sources.base import ErrorKind, FetchResult, SecretSource
from agent.secret_scope import _decode_env_bytes, _parse_env_text

MAX_FILE_BYTES = 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024


class _UnsafeSecretFile(ValueError):
    """The configured file is not a private, bounded regular file owned by this user."""


def _resolve_path(raw_path: object, home_path: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("path is empty")
    value = raw_path.strip()
    if value == "~":
        return Path.home()
    if value.startswith("~/"):
        return Path.home() / value[2:]
    if value.startswith("~"):
        raise ValueError("named-user paths are not supported")
    path = Path(value)
    return path if path.is_absolute() else home_path / path


def _check_private_regular_file(before: os.stat_result, opened: os.stat_result) -> None:
    if not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(opened.st_mode):
        raise _UnsafeSecretFile("not a regular file")
    if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino) or not opened.st_ino:
        raise _UnsafeSecretFile("file identity changed while opening")
    if opened.st_uid != os.geteuid() or stat.S_IMODE(opened.st_mode) & 0o077:  # windows-footgun: ok — caller rejects Windows
        raise _UnsafeSecretFile("file is not private to the current user")
    if opened.st_size > MAX_FILE_BYTES:
        raise _UnsafeSecretFile("file exceeds the byte limit")


def _read_private_dotenv(path: Path) -> bytes:
    if os.name == "nt":
        raise _UnsafeSecretFile("private ACL validation is not available on this platform")
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(before.st_mode):
        raise _UnsafeSecretFile("symbolic links are not accepted")
    if not stat.S_ISREG(before.st_mode):
        raise _UnsafeSecretFile("not a regular file")

    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        _check_private_regular_file(before, opened)
        contents = bytearray()
        while len(contents) <= MAX_FILE_BYTES:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, MAX_FILE_BYTES + 1 - len(contents)))
            if not chunk:
                break
            contents.extend(chunk)
        if len(contents) > MAX_FILE_BYTES:
            raise _UnsafeSecretFile("file exceeds the byte limit")
        return bytes(contents)
    finally:
        os.close(descriptor)


class FileSource(SecretSource):
    """Bulk secrets from a user-selected, private dotenv file.

    Relative paths are resolved under the owning profile home. ``~/`` always
    means the operating-system user's home, so profiles may share one source.
    """

    name = "file"
    label = "dotenv file"
    shape = "bulk"
    override_existing_default = False
    remediation_hints = {
        ErrorKind.NOT_CONFIGURED: (
            "Set secrets.file.path to a private dotenv file and verify it is readable by this user."
        ),
        ErrorKind.INTERNAL: (
            "Use an owned regular file with mode 0600 or stricter; Hermes does not follow file symlinks."
        ),
    }

    def config_schema(self) -> dict:
        return {
            "enabled": {"description": "Master switch", "default": False},
            "path": {"description": "Private dotenv file; relative paths use the profile home, ~/ uses the OS home",
                     "default": ""},
            "override_existing": {"description": "File values overwrite .env/shell values", "default": False},
        }

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        result = FetchResult()
        cfg = cfg if isinstance(cfg, dict) else {}
        if not self.is_enabled(cfg):
            return result
        try:
            path = _resolve_path(cfg.get("path"), Path(home_path))
        except (OSError, RuntimeError, TypeError, ValueError):
            return result.fail("Configured secret file path is missing or invalid.", ErrorKind.NOT_CONFIGURED)
        try:
            contents = _read_private_dotenv(path)
        except FileNotFoundError:
            return result.fail("Configured secret file does not exist.", ErrorKind.NOT_CONFIGURED)
        except (_UnsafeSecretFile, OSError, ValueError):
            return result.fail("Configured secret file is unreadable or violates the private-file policy.",
                               ErrorKind.INTERNAL)
        result.secrets = _parse_env_text(_decode_env_bytes(contents))
        return result
