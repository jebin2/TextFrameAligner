import subprocess

def _is_running_in_vm():
    try:
        output = subprocess.check_output(['whoami']).decode().strip()
        if "vboxuser" in output:
            return True
    except Exception:
        pass
    return False

isVM = _is_running_in_vm()

OLLAMA_REQ_URL="http://172.28.156.132:11434" if isVM else "http://127.0.0.1:11434"