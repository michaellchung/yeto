"""Local launcher for the RC-6 kernel probe (Modal Sandbox, H100!:2, hard timeout; independent watchdog = go_k.sh).

    python kernel_probe_run.py <outdir> <app-name>
"""
import base64
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modal_run import IMAGE, MILES_COMMIT  # noqa: E402

TIMEOUT_S = int(os.environ.get("KPROBE_TIMEOUT_S", "2400"))
SCRIPT = f"""set -uo pipefail
exec 2>&1
mkdir -p /work/k
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader | tee /work/k/gpus.txt
n=$(grep -c . /work/k/gpus.txt); bad=$(grep -vc '^NVIDIA H100 80GB HBM3,' /work/k/gpus.txt || true)
if [ "$n" != 2 ] || [ "$bad" != 0 ]; then echo "GPU assertion failed"; exit 3; fi
test "$(git --git-dir=/root/miles/.git rev-parse HEAD)" = {MILES_COMMIT} || {{ echo 'miles pin mismatch'; exit 4; }}
nvidia-smi -q > /work/k/nvidia_smi_q.txt 2>&1; nvidia-smi topo -m > /work/k/topo.txt 2>&1; env | grep -E 'TORCH|TRITON|INDUCTOR|CUDA|NVTE' > /work/k/env.txt
pack() {{ tar czf /work/k.tgz -C /work k 2>/dev/null; echo "=== EVIDENCE_B64 ==="; base64 -w0 /work/k.tgz; echo; }}
trap pack EXIT
python /yeto/tools/probes/e3_reshard/kernel_probe.py driver /work/k/out
echo "driver rc=$?"
"""


def main(argv):
    import modal
    out, app_name = Path(argv[0]), argv[1]
    out.mkdir(parents=True, exist_ok=True)
    import base64 as b64, json
    auth = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
    user, token = b64.b64decode(auth).decode().split(":", 1)
    secret = modal.Secret.from_dict({"REGISTRY_USERNAME": user, "REGISTRY_PASSWORD": token})
    image = (modal.Image.from_registry(IMAGE, secret=secret).entrypoint([])
             .add_local_dir(str(HERE), "/yeto/tools/probes/e3_reshard", copy=False, ignore=["**/__pycache__"]))
    app = modal.App.lookup(app_name, create_if_missing=True)
    try:
        sb = modal.Sandbox.create("bash", "-c", SCRIPT, app=app, image=image, gpu="H100!:2", cpu=8.0, memory=65536,
                                  timeout=TIMEOUT_S)
        (out / "resources.txt").open("a").write(f"{app.app_id} {sb.object_id} kprobe {time.strftime('%FT%TZ', time.gmtime())}\n")
        chunks = []
        with open(out / "container.log", "a", encoding="utf-8") as log:
            for line in sb.stdout:
                chunks.append(line)
                if "=== EVIDENCE_B64 ===" not in line and len(line) < 100000:
                    log.write(line); log.flush()
        sb.wait(raise_on_termination=False)
        text = "".join(chunks)
        if "=== EVIDENCE_B64 ===" in text:
            (out / "evidence.tgz").write_bytes(base64.b64decode(text.split("=== EVIDENCE_B64 ===", 1)[1].strip().split()[0]))
        (out / "returncode.txt").write_text(str(sb.returncode))
    finally:
        subprocess.run([str(Path(sys.executable).with_name("modal")), "app", "stop", "-y", app.app_id], check=False)
        (out / "app_stopped.txt").write_text(f"{app.app_id} {time.strftime('%FT%TZ', time.gmtime())}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
