# Clouds

Which clouds `yeto shape` can plan across and `yeto launch` can provision
on, where each one's credentials live, what "region" means to it, and
which behaviours have been verified on real machines. Yeto keeps no
credentials of its own: every cloud's own CLI writes them to a standard
place on the **submitting machine**, SkyPilot and Yeto read them there,
and in head controller mode the launcher copies only the files of the
clouds a fleet actually touches onto the head VM. Learner islands never
receive cloud credentials (only the Hugging Face token, and W&B /
CyberGym keys when those features are on).

On Windows, run everything from WSL: SkyPilot imports the POSIX-only
`resource` module and fails on native Windows. Credential files must be in
the WSL home directory (`/home/<user>/...`), not `C:\Users\...`.

## Credentials

| Cloud | Set up with | File Yeto/sky read | Env-var alternative |
|---|---|---|---|
| AWS | `aws configure` | `~/.aws/` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |
| RunPod | `runpod config` | `~/.runpod/config.toml` | `RUNPOD_API_KEY` |
| Nebius | `nebius iam get-access-token > ~/.nebius/NEBIUS_IAM_TOKEN.txt` and `nebius --format json iam whoami \| jq -r '.user_profile.tenants[0].tenant_id' > ~/.nebius/NEBIUS_TENANT_ID.txt` | `~/.nebius/` | `NEBIUS_IAM_TOKEN` + `NEBIUS_TENANT_ID` |
| Verda | console → Credentials → Cloud API Credentials, saved as JSON | `~/.verda/config.json` (`client_id`, `client_secret`) | `VERDA_CLIENT_ID` + `VERDA_CLIENT_SECRET` |
| Modal | `modal token new` | `~/.modal.toml` | `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` |
| GCP (optional, `gs://` outputs only) | `gcloud auth application-default login` | `~/.config/gcloud/` | — |

`sky check` reports each sky cloud as enabled or explains what is missing.
Modal is not a sky cloud; `modal token` shows the active token.

**Nebius binds one project to one region.** Every Nebius region a fleet
touches needs `nebius.region_configs.<region>.project_id` in
`~/.sky/config.yaml`; `yeto launch` refuses a fleet that names a region
without one, before any VM is created. The same project ids let the
planner ask Nebius for live preemptible prices.

## What each cloud means by "region"

| Cloud | `--regions` entry | Meaning | Notes |
|---|---|---|---|
| AWS | `aws:us-east-1` (or bare `us-east-1`) | data-center region | quotas, placement scores and prices are per region; default `us-east-1,us-east-2,us-west-1,us-west-2` |
| Nebius | `nebius:eu-north1` | region | catalog (2026-09-23): eu-north1, eu-west1, me-west1, uk-south1, uk-south2, us-central1; H100×8 only in eu-north1, B200 only in me-west1/us-central1 |
| Verda | `verda:FIN-03` | location code | FIN-01/02/03 (Helsinki), ICL-01 (Iceland); the planner reads the live list, falling back to these |
| RunPod | `runpod:CA` | country code | one global pool; stock is not per country |
| Modal | `modal:us` / `modal:us-west` | placement hint with a surcharge | broad (`us`, `eu`, `ap`) ×1.15, narrow (e.g. `us-west`, `eu-north`) ×1.75; leave it out to run unpinned at the base price |

Clouds you do not name in `--regions` are unrestricted, except AWS, which
stays on its default list.

## What the planner knows per cloud

| Cloud | Catalog and price | Capacity signal | Spot |
|---|---|---|---|
| AWS | sky catalog (refreshed every few hours) | vCPU quota − usage, spot placement score (AWS APIs) | yes |
| RunPod | sky catalog (no H200 rows — the planner says so) | stock High/Medium/Low (RunPod GraphQL) | catalog spot = on-demand |
| Nebius | sky catalog with spot column; live preemptible price from the billing calculator when a project id is configured | Capacity API resource-advice, per region/platform/preset | yes; **dynamic spot pricing from 2026-10-08** — the live price overrides the catalog |
| Verda | Verda's public `/v1/instance-types` (sky's catalog is too thin) × locations | one bulk `/v1/instance-availability` call | flat 50% of on-demand |
| Modal | static table in `yeto/shape/providers.py` dated 2026-09-23 (GPU + reserved CPU + memory); the planner warns after 90 days | none; constant "will run, may queue" | none |

## Modal islands

Modal has no VMs and no SSH, so a `modal:` island is a Modal function call
(`yeto/modal_runner.py`), not a sky cluster. It runs the same `run`
script the sky task would run, with the `SKYPILOT_*` variables provided
from Modal's cluster info. Facts to keep in mind:

- The syncer must be reachable from Modal's network. Head controller mode
  (the default) puts it on a public head VM; under `--controller local`
  pass `--syncer-public-addr HOST:PORT` when the syncer address is private.
- Multi-container islands (`modal:2x8xh100`) must use whole nodes
  (`H100:8` per container, Modal rule since 2026-05-31) and get RDMA.
- RL islands need `--rl-image docker:<repo>@sha256:<digest>` — the same
  digest the sky islands use.
- Object-store data (`s3://`, `gs://`) cannot be mounted; use an HF dataset
  id or a local path.
- `yeto down <run>` stops the run's Modal app (`yeto-<prefix>`); every
  island's containers end with it.
- An all-Modal SFT fleet's model is not fetchable over ssh; recover it
  from the syncer checkpoint with `yeto-export`, or keep at least one sky
  island in the fleet.
- Manual join: launch with `--external-learners 1`, then
  `python -m yeto.modal_runner --learner-id N --num-learners M --syncer-addr H:P --run-script <file> ...`.

## Verified on real machines

Verification results decide runtime behaviour: the planner only plans RL
islands on clouds in `VERIFIED_DOCKER_IMAGE_CLOUDS`, prices RL spot only
on clouds in `VERIFIED_SPOT_STORAGE_CLOUDS`, and assumes an RDMA fabric
for multi-node islands only on `RDMA_CLOUDS` (all in
`yeto/shape/catalog.py`). Update the constants and this table together.

| Cloud | SFT island, 2 sync rounds | RL island via `docker:` image | Spot checkpoint store | Multi-node fabric | Notes |
|---|---|---|---|---|---|
| AWS | yes (pre-existing) | yes (pre-existing) | yes (`sky.Storage`) | EFA on p4/p5 | |
| RunPod | yes (pre-existing) | yes (pre-existing) | not verified | single node | |
| Nebius | pending | pending | pending (S3-compatible store needs `aws configure --profile nebius`) | pending (expect InfiniBand on 8-GPU SXM) | first run: eu-north1, 1×H100 |
| Verda | pending | pending | not available in sky | single node | first run: FIN-03, 1×H100 |
| Modal | pending | pending (image via registry digest) | pending (Modal Volume) | pending (RoCE, whole nodes) | measure WAN sync time |

Record each run's syncer-tape summary and the exact command here when a
row changes.
