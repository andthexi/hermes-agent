# Native dotenv File Secret Source

The native `file` source reads a dotenv file as data. It does not execute the file, invoke a shell, start a subprocess, or make a network request. This is useful when one shared file supplies common credentials to several Hermes profiles.

```yaml
secrets:
  file:
    enabled: true
    path: ~/.config/hermes-shared/secrets.env
    override_existing: false
```

Example file:

```dotenv
OPENAI_API_KEY=your-secret-value
ANTHROPIC_API_KEY='another-secret-value'
```

Keep the file outside Git and plugin directories. On supported POSIX hosts it must be a regular file owned by the current user with no group/other permission bits (normally mode `0600` or stricter). Hermes rejects symlinks, FIFOs, devices, non-private files, and files larger than 1 MiB. The current implementation fails closed on Windows because Hermes does not yet validate Windows ACL privacy for this source; no Windows support is implied.

`path` may be absolute, relative to the owning profile's Hermes home, or start with `~/` to use the operating-system user's home. A `~/` path is therefore shared independently of `HERMES_HOME`, while relative paths are profile-specific. The source uses Hermes' regular dotenv parsing semantics, including `export`, comments, and quoted values.

The default `override_existing: false` lets each profile's `.env` or existing shell value win; file values fill missing names and other values remain additive. Set it to `true` only when the shared file should replace existing values. `secrets.preserve_existing` remains stronger than any source override. Sources still obey the normal precedence, bootstrap-variable protection, and source-attribution rules described in [Secrets](./index.md#multiple-sources-at-once).

The file contents are read only during the normal secret-source hydration and refresh lifecycle. Hermes does not watch the file continuously or provision it. Restart Hermes or use the application's normal secret reload path to pick up rotations. Disabling the source removes its config contribution; it never deletes or edits the external file. Only the path and source settings belong in `config.yaml`; secret values do not.
