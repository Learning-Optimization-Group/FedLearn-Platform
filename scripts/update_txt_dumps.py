import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DUMPS_DIR = os.path.join(PROJECT_ROOT, "docs", "llm_dumps")

os.makedirs(DUMPS_DIR, exist_ok=True)

mappings = {
    os.path.join(PROJECT_ROOT, "backend"): os.path.join(DUMPS_DIR, "fedlearn_backend.txt"),
    os.path.join(PROJECT_ROOT, "client-docker"): os.path.join(DUMPS_DIR, "fedlearn_client_docker.txt"),
    os.path.join(PROJECT_ROOT, "framework"): os.path.join(DUMPS_DIR, "fedlearn_framework.txt"),
    os.path.join(PROJECT_ROOT, "frontend"): os.path.join(DUMPS_DIR, "fedlearn_frontend.txt")
}

allowed_extensions = {".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".sh", ".md", ".txt", ".yaml", ".yml"}
allowed_files = {"Dockerfile", "package.json", "pom.xml", "application.properties", "requirements.txt", "CONTRIBUTING.md"}

skip_dirs = {".git", "__pycache__", "venv", ".venv", "node_modules", "target", "build", "dist", ".pytest_cache", "egg-info", "FedLearn.egg-info", ".idea", ".vscode"}

def build_txt(target_dir, output_file):
    print(f"Updating {output_file} from {target_dir}...")
    with open(output_file, "w") as out:
        for root, dirs, files in os.walk(target_dir):
            if any(x in root.split(os.sep) for x in skip_dirs):
                continue
            for file in sorted(files):
                ext = os.path.splitext(file)[1]
                # Allow files that either have a matching extension or match an allowed exact filename
                if ext in allowed_extensions or file in allowed_files:
                    file_path = os.path.join(root, file)
                    out.write("="*80 + "\n")
                    out.write(f"File: {file_path}\n")
                    out.write("="*80 + "\n\n")
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            out.write(f.read())
                            out.write("\n\n")
                    except Exception as e:
                        out.write(f"# Error reading file: {e}\n\n")

if __name__ == "__main__":
    for target_dir, output_file in mappings.items():
        if os.path.exists(target_dir):
            build_txt(target_dir, output_file)
        else:
            print(f"Warning: Directory {target_dir} not found.")
    print("Done updating text dumps.")
