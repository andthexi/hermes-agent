# Secrets

Hermes can resolve API keys from external secret sources at process startup instead of storing them in `~/.hermes/.env`. Vendor-manager bootstrap tokens live in `.env`; provider keys can stay in the manager or a private local dotenv file and rotate centrally.

Supported:

- [Bitwarden Secrets Manager](./bitwarden) — `bws` CLI, lazy-installed, free tier works.
- [1Password](./onepassword) — `op://` references via the official `op` CLI; service-account or desktop session auth.
- [Native dotenv file](./file) — read a private, bounded dotenv file without a shell or network access.
- [Command helper](./command) — an explicitly user-configured helper that prints `KEY=VALUE` lines.

## Multiple sources at once

You can enable more than one secret source at the same time — for example a team Bitwarden project alongside a personal vault plugin. Sources compose per env var with a deterministic precedence ladder:

1. **Your `.env` / shell wins by default.** A source only replaces a pre-existing value when its own `override_existing: true` is set (Bitwarden defaults to true so central rotation works).
2. **Mapped sources beat bulk sources.** A source where you explicitly bind env vars to references (an `env:` map) outranks a source that injects a whole project of secrets implicitly, regardless of ordering.
3. **First source wins.** Within the same shape, the order of the optional `secrets.sources` list (or registration order) decides. Later claims on an already-claimed var are skipped — with a startup warning, never silently.

`override_existing` never lets one source overwrite a var another source already claimed, and no source can ever overwrite another source's bootstrap token (e.g. `BWS_ACCESS_TOKEN`).

A plugin may declare non-secret `secrets.file` defaults so a shared dotenv source can be configured once without repeating its path in every profile. This is read-time config only: secret values stay in the external file, never in `plugin.yaml`, config YAML, or credential-pool rows. With `override_existing: false`, each profile's `.env` wins and other shared values are additive. `~/` resolves against the OS user's home, independently of `HERMES_HOME`; relative paths resolve under the owning profile home. The file is still plaintext: keep it owner-readable only (mode 0600 or stricter), outside Git/plugin directories, and back it up separately from a Hermes home. Hermes reads it only during the normal secret-source refresh; there is no watcher or automatic provisioning.

```yaml
secrets:
  sources: [bitwarden]     # optional explicit ordering
  bitwarden:
    enabled: true
    project_id: "..."
```

Every credential injected by a source is labelled with its origin — setup flows and `hermes model` show `(from Bitwarden)` next to detected keys so you always know where a value came from.

## Profiles and shared vaults

Two orchestrator-level knobs make one shared vault safe across [profiles](../profiles):

- **`secrets.preserve_existing`** — a list of env var names whose existing `.env` / shell value always wins, even against a source with `override_existing: true`. Use it for per-profile platform secrets (e.g. `FEISHU_APP_SECRET`) that intentionally differ across profiles while everything else rotates centrally:

  ```yaml
  secrets:
    preserve_existing: [FEISHU_APP_SECRET, TELEGRAM_BOT_TOKEN]
  ```

- **Profile aliasing** (on by default, `secrets.profile_alias: false` to disable) — when Hermes runs under a named profile, a vault secret named `FOO_<PROFILE>` (credential-shaped suffixes only: `*_API_KEY`, `*_TOKEN`, `*_SECRET`, `*_KEY`, `*_PASSWORD`) also hydrates the canonical `FOO`. Store `TELEGRAM_BOT_TOKEN_MILLA` in the shared project and the `milla` profile's adapters — which read the fixed name `TELEGRAM_BOT_TOKEN` — get the right value automatically. A var the vault supplies directly under its canonical name always beats an alias.

Both apply to every source — bundled and plugin — because they live in the orchestrator, not the backends.

## Secrets in child processes

Terminal commands, `execute_code` sandboxes and [`no_agent` cron scripts](../features/cron.md#giving-a-script-a-credential) run with a sanitized environment: Hermes-managed credentials are stripped, and only variables you declare in `terminal.env_passthrough` (or a loaded skill's `required_environment_variables`) are forwarded. A declared variable is forwarded with the **owning profile's** value — from that profile's `.env` or its secret sources — even when the profile is served by a multi-profile gateway or the Desktop/dashboard backend and its secrets never entered the process environment. A profile's declared value never reaches another profile's children, and the launch profile's `.env` credentials are dropped from children that run for a served profile. Provider credentials cannot be declared; see [Security → Credential scoping](../security.md).

## Adding your own backend

Third-party secret managers ship as standalone plugins, not core PRs. A backend subclasses `agent.secret_sources.base.SecretSource` (one required method: `fetch(cfg, home_path) -> FetchResult`) and registers via `ctx.register_secret_source(MySource())` in the plugin's `register(ctx)`. The orchestrator owns precedence, conflict handling, timeouts, and provenance — your source only fetches. Full guide with the contract rules, subprocess-safety helper, and conformance kit: [Building a Secret Source Plugin](../../developer-guide/secret-source-plugin.md).

The bundled set is deliberately closed (same policy as memory providers): Bitwarden and 1Password are the in-tree vendor integrations. The local dotenv-file reader and command helper are generic inputs, not vendor backends. Everything else — Infisical, Proton Pass, HashiCorp Vault, AWS Secrets Manager, OS keystores — belongs in plugin repos; share them in the Nous Research Discord (`#plugins-skills-and-skins`).
