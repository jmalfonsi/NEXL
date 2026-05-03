# modal_proto_sglang_v16.py
# SGLang + Qwen2.5-7B-Instruct-AWQ sur Modal L4
# Corrections v16 :
# - Image CUDA via modal.Image.from_registry(...)
# - libnuma1 pour sgl_kernel
# - ninja-build/build-essential pour FlashInfer JIT
# - cache persistant Modal pour HF / Torch / FlashInfer
# - entrée recommandée : sglang serve
# - --model-path au lieu de --model
# - attente robuste avec /model_info puis warmup /generate
# - timeout long pour première compilation JIT FlashInfer

import modal
import time
import sys
import subprocess
from multiprocessing import Process


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-AWQ"
PORT = 30000
BASE_URL = f"http://localhost:{PORT}"


cache_vol = modal.Volume.from_name("sglang-cache-v16", create_if_missing=True)


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "libc6-dev",
        "libnuma1",
        "numactl",
        "ninja-build",
        "ccache",
    )
    .pip_install(
        "torch",
        "sglang[all]>=0.4.6.post1",
        "sglang-kernel",
        "transformers",
        "huggingface_hub",
        "outlines>=0.0.44",
        "pyairports",
        "requests",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/lib",
            "HF_HOME": "/root/.cache/huggingface",
            "HUGGINGFACE_HUB_CACHE": "/root/.cache/huggingface/hub",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface/transformers",
            "TORCH_EXTENSIONS_DIR": "/root/.cache/torch_extensions",
            "FLASHINFER_CACHE_DIR": "/root/.cache/flashinfer",
            "CCACHE_DIR": "/root/.cache/ccache",
        }
    )
)

app = modal.App("sglang-l4-proto-v16", image=image)


def run_diag() -> None:
    print("\n========== DIAGNOSTIC CUDA / SYSTEM ==========")

    checks = [
        "python --version",
        "which python",
        "nvidia-smi || true",
        "ls -l /usr/local | grep cuda || true",
        "echo CUDA_HOME=$CUDA_HOME",
        "which nvcc || true",
        "nvcc --version || true",
        "ldconfig -p | grep libnuma || true",
        "which ninja || true",
        "ninja --version || true",
        "python - <<'PY'\nimport torch\nprint('torch:', torch.__version__)\nprint('cuda available:', torch.cuda.is_available())\nprint('cuda version:', torch.version.cuda)\nif torch.cuda.is_available():\n    print('device:', torch.cuda.get_device_name(0))\n    print('capability:', torch.cuda.get_device_capability(0))\nPY",
        "python - <<'PY'\ntry:\n    import sgl_kernel\n    print('sgl_kernel OK')\nexcept Exception as e:\n    print('sgl_kernel ERROR:', repr(e))\nPY",
    ]

    for cmd in checks:
        print(f"\n$ {cmd}")
        subprocess.run(["bash", "-lc", cmd], check=False)

    print("\n========== FIN DIAGNOSTIC ==========\n")


def start_server() -> None:
    cmd = [
        "sglang",
        "serve",
        "--model-path",
        MODEL_ID,
        "--quantization",
        "awq_marlin",
        "--max-total-tokens",
        "6000",
        "--context-length",
        "8192",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--attention-backend",
        "flashinfer",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
    ]

    print("Démarrage du serveur :")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def wait_for_http_api(requests_module, timeout_s: int = 900) -> None:
    print("\n--- Attente de l'API HTTP SGLang via /model_info ---")

    deadline = time.time() + timeout_s
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        try:
            r = requests_module.get(f"{BASE_URL}/model_info", timeout=5)
            print(f"[model_info #{attempt}] HTTP {r.status_code}")

            if r.status_code == 200:
                print("✅ API HTTP SGLang disponible.")
                return

        except requests_module.exceptions.RequestException as e:
            print(f"[model_info #{attempt}] pas encore prêt : {type(e).__name__}")

        time.sleep(5)

    raise RuntimeError("L'API HTTP SGLang n'a pas démarré dans le délai imparti.")


def warmup_generation(requests_module) -> None:
    print("\n--- Warmup génération / FlashInfer JIT ---")
    print("Le premier warmup peut prendre plusieurs minutes à cause de la compilation JIT.")

    payload = {
        "text": "Hello",
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 1,
        },
        "stream": False,
    }

    last_error = None

    for attempt in range(1, 4):
        try:
            print(f"Warmup tentative {attempt}/3...")
            r = requests_module.post(
                f"{BASE_URL}/generate",
                json=payload,
                timeout=900,
            )

            print("warmup status:", r.status_code)
            print("warmup body:", r.text[:500])

            if r.status_code == 200:
                print("✅ Warmup terminé.")
                return

            last_error = RuntimeError(f"Warmup HTTP {r.status_code}: {r.text[:500]}")

        except requests_module.exceptions.RequestException as e:
            print(f"warmup exception: {repr(e)}")
            last_error = e

        time.sleep(10)

    raise RuntimeError(f"Warmup SGLang échoué: {repr(last_error)}")


def wait_for_health_optional(requests_module, timeout_s: int = 180) -> None:
    print("\n--- Vérification optionnelle /health ---")

    deadline = time.time() + timeout_s
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        try:
            r = requests_module.get(f"{BASE_URL}/health", timeout=5)
            print(f"[health #{attempt}] HTTP {r.status_code}")

            if r.status_code == 200:
                print("✅ /health OK.")
                return

        except requests_module.exceptions.RequestException as e:
            print(f"[health #{attempt}] erreur : {type(e).__name__}")

        time.sleep(5)

    print("⚠️ /health n'est pas passé à 200, mais le warmup /generate a déjà réussi.")
    print("On continue quand même les tests.")


def simple_generation_test(requests_module) -> None:
    print("\n--- Test de génération simple ---")

    t0 = time.time()

    resp = requests_module.post(
        f"{BASE_URL}/generate",
        json={
            "text": "Hello, world! " * 10,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 30,
            },
            "stream": True,
        },
        stream=True,
        timeout=300,
    )

    resp.raise_for_status()

    tokens = 0
    for line in resp.iter_lines():
        if line:
            tokens += 1

    elapsed = time.time() - t0
    print(f"✅ Génération simple : {tokens} lignes/tokens streamés en {elapsed:.2f}s")


def saturation_test(requests_module) -> None:
    print("\n--- Test de saturation du budget de tokens ---")

    import threading

    def send_long(idx: int) -> None:
        try:
            prompt = "Once upon a time, " * 1200

            r = requests_module.post(
                f"{BASE_URL}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": 100,
                    },
                    "stream": True,
                },
                stream=True,
                timeout=600,
            )

            r.raise_for_status()

            tokens = 0
            for line in r.iter_lines():
                if line:
                    tokens += 1

            print(f"✅ Requête longue {idx}: {tokens} lignes/tokens streamés")

        except Exception as e:
            print(f"❌ Requête longue {idx} échouée: {repr(e)}")

    threads = [
        threading.Thread(target=send_long, args=(i,), daemon=True)
        for i in range(3)
    ]

    for t in threads:
        t.start()

    time.sleep(3)

    print("Injection d'une requête rapide Fast Lane...")

    try:
        t0 = time.time()

        r_short = requests_module.post(
            f"{BASE_URL}/generate",
            json={
                "text": "Hello",
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 20,
                },
                "stream": True,
            },
            stream=True,
            timeout=300,
        )

        r_short.raise_for_status()

        short_tokens = 0
        for line in r_short.iter_lines():
            if line:
                short_tokens += 1

        elapsed = time.time() - t0
        print(f"✅ Requête courte : {short_tokens} lignes/tokens streamés en {elapsed:.2f}s")

    except Exception as e:
        print(f"❌ Requête courte échouée: {repr(e)}")

    for t in threads:
        t.join()

    print("✅ Test de saturation terminé.")


@app.function(
    gpu="L4",
    timeout=3600,
    volumes={
        "/root/.cache": cache_vol,
    },
)
def full_test():
    import requests

    run_diag()

    server_proc = Process(target=start_server, daemon=True)
    server_proc.start()

    try:
        wait_for_http_api(requests, timeout_s=900)
        warmup_generation(requests)

        # /health peut parfois rester 503 selon l'état interne SGLang.
        # Après warmup réussi, on ne bloque pas strictement dessus.
        wait_for_health_optional(requests, timeout_s=180)

        simple_generation_test(requests)
        saturation_test(requests)

        print("\n✅ Tous les tests v16 sont terminés.")

    finally:
        print("\nArrêt du serveur SGLang...")
        if server_proc.is_alive():
            server_proc.terminate()
            server_proc.join(timeout=30)

        if server_proc.is_alive():
            print("Le serveur ne s'est pas arrêté proprement, kill forcé.")
            server_proc.kill()
            server_proc.join(timeout=10)

        print("Commit du volume cache Modal...")
        cache_vol.commit()
        print("✅ Cache sauvegardé.")


@app.local_entrypoint()
def main():
    full_test.remote()