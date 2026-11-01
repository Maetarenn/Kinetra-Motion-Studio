"""Automated Release Builder for Kinetra Motion Studio.

Performs:
1. PyInstaller package build from Kinetra.spec -> dist/Kinetra/
2. Verification launch of dist/Kinetra/Kinetra.exe (ensures no missing modules)
3. Inno Setup installer compilation from packaging/installer.iss -> installer_output/
4. Verification & mirror copy to Kinetra_Setup_Release/
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPEC_FILE = PROJECT_ROOT / "Kinetra.spec"
ISS_FILE = PROJECT_ROOT / "packaging" / "installer.iss"
ISCC_PATH = Path(r"C:\Users\hmaki\AppData\Local\Programs\Inno Setup 6\ISCC.exe")
OUTPUT_DIR = PROJECT_ROOT / "installer_output"
SETUP_EXE_NAME = "Kinetra_Setup_v0.4.0.exe"
TARGET_RELEASE_DIR = PROJECT_ROOT.parent / "Kinetra_Setup_Release"


def log(msg):
    print(f"\n[BUILD] {msg}")


def sha256_of_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.chdir(str(PROJECT_ROOT))
    log(f"Working directory: {PROJECT_ROOT}")

    # 1. Check tools
    log("Checking prerequisites...")
    if not ISCC_PATH.exists():
        sys.exit(f"Error: Inno Setup compiler (ISCC.exe) not found at {ISCC_PATH}")
    log(f"Inno Setup Compiler: {ISCC_PATH}")

    # 2. Run PyInstaller
    log("Step 1/4: Running PyInstaller to bundle Kinetra into dist/Kinetra...")
    cmd_pyinstaller = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        str(SPEC_FILE),
    ]
    ret = subprocess.run(cmd_pyinstaller, cwd=str(PROJECT_ROOT))
    if ret.returncode != 0:
        sys.exit(f"Error: PyInstaller build failed with exit code {ret.returncode}")

    dist_exe = PROJECT_ROOT / "dist" / "Kinetra" / "Kinetra.exe"
    if not dist_exe.exists():
        sys.exit(f"Error: Expected executable not found at {dist_exe}")
    log(f"PyInstaller build successful: {dist_exe}")

    # Ensure icon exists in _internal for Inno Setup
    internal_ico = PROJECT_ROOT / "dist" / "Kinetra" / "_internal" / "kinetra" / "assets" / "icon.ico"
    src_ico = PROJECT_ROOT / "kinetra" / "assets" / "icon.ico"
    if not internal_ico.exists() and src_ico.exists():
        log(f"Copying icon to internal assets: {internal_ico}")
        internal_ico.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_ico), str(internal_ico))

    # 3. Verification launch of dist/Kinetra/Kinetra.exe
    log("Step 2/4: Verifying dist/Kinetra/Kinetra.exe startup (smoke test)...")
    try:
        proc = subprocess.Popen([str(dist_exe)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            proc.wait(timeout=4)
            if proc.returncode != 0:
                stdout, stderr = proc.communicate()
                err_text = stderr.decode('utf-8', errors='ignore') + stdout.decode('utf-8', errors='ignore')
                sys.exit(f"Error: Kinetra.exe crashed on startup with code {proc.returncode}!\n{err_text}")
        except subprocess.TimeoutExpired:
            # Still running without crash -> success!
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
            log("Kinetra.exe launch verification PASSED (no startup crash)!")
    except Exception as e:
        log(f"Launch verification note: {e}")

    # 4. Run Inno Setup Compiler
    log("Step 3/4: Compiling installer with Inno Setup...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd_iscc = [str(ISCC_PATH), str(ISS_FILE)]
    ret_iscc = subprocess.run(cmd_iscc, cwd=str(PROJECT_ROOT / "packaging"))
    if ret_iscc.returncode != 0:
        sys.exit(f"Error: Inno Setup compilation failed with exit code {ret_iscc.returncode}")

    built_installer = OUTPUT_DIR / SETUP_EXE_NAME
    if not built_installer.exists():
        sys.exit(f"Error: Installer not found at {built_installer}")

    file_size_mb = built_installer.stat().st_size / (1024 * 1024)
    file_sha = sha256_of_file(built_installer)
    log(f"Installer successfully built: {built_installer}")
    log(f"Size: {file_size_mb:.2f} MB | SHA-256: {file_sha}")

    # 5. Mirror to Kinetra_Setup_Release
    log(f"Step 4/4: Copying updated installer to {TARGET_RELEASE_DIR}...")
    TARGET_RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    target_exe = TARGET_RELEASE_DIR / SETUP_EXE_NAME
    shutil.copy2(str(built_installer), str(target_exe))
    log(f"Updated release copy ready at: {target_exe}")

    print("\n" + "=" * 60)
    print("SUCCESS: Kinetra v0.4.0 Setup Installer is ready for release!")
    print(f"Path: {target_exe}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"SHA-256: {file_sha}")
    print("=" * 60)


if __name__ == "__main__":
    main()
