import os
import argparse
import logging
from typing import Set, Dict, Tuple

# Setup default logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DEFAULT_DUMPS_DIR: str = os.path.join(PROJECT_ROOT, "docs", "llm_dumps")

# Default mappings
MAPPINGS: Dict[str, str] = {
    os.path.join(PROJECT_ROOT, "backend"): "fedlearn_backend.txt",
    os.path.join(PROJECT_ROOT, "client-docker"): "fedlearn_client_docker.txt",
    os.path.join(PROJECT_ROOT, "framework"): "fedlearn_framework.txt",
    os.path.join(PROJECT_ROOT, "frontend"): "fedlearn_frontend.txt"
}

ALLOWED_EXTENSIONS: Set[str] = {
    ".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".sh", ".md", ".txt", 
    ".yaml", ".yml", ".toml", ".json", ".sql"
}

ALLOWED_FILES: Set[str] = {
    "Dockerfile", "package.json", "pom.xml", "application.properties", 
    "requirements.txt", "CONTRIBUTING.md", "README.md", "gradlew", ".env.example"
}

# Directories to absolutely ignore during traversal
SKIP_DIRS: Set[str] = {
    ".git", "__pycache__", "venv", ".venv", "node_modules", "target", "build", 
    "dist", ".pytest_cache", "egg-info", "FedLearn.egg-info", ".idea", ".vscode",
    "apache-maven-3.9.6", "gradle-8.7"
}

def build_txt(target_dir: str, output_file: str) -> Tuple[int, int]:
    """
    Crawls target_dir and writes concatenated code files to output_file.
    Returns (num_files_processed, total_lines_written).
    """
    logging.info(f"Updating {output_file} from {target_dir}...")
    
    files_processed = 0
    total_lines = 0
    
    with open(output_file, "w", encoding="utf-8") as out:
        for root, dirs, files in os.walk(target_dir):
            # Prune skipped directories in-place to avoid deep traversal operations
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            
            for file in sorted(files):
                ext = os.path.splitext(file)[1].lower()
                
                # Check if it's an allowed file
                if ext in ALLOWED_EXTENSIONS or file in ALLOWED_FILES:
                    # Explicit exclusion for lock files
                    if file in ("package-lock.json", "yarn.lock"):
                        continue
                        
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, PROJECT_ROOT)
                    
                    # Markdown code fence identifier
                    fence = ext[1:] if len(ext) > 1 else 'text'
                    
                    out.write("="*80 + "\n")
                    out.write(f"File: {rel_path}\n")
                    out.write("="*80 + "\n\n")
                    out.write(f"```{fence}\n")
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            files_processed += 1
                            total_lines += len(lines)
                            
                            content = "".join(lines)
                            # Ensure trailing newline prevents mangled backtick fences
                            if content and not content.endswith('\n'):
                                content += '\n'
                            out.write(content)
                    except Exception as e:
                        logging.warning(f"Error reading file {rel_path}: {e}")
                        out.write(f"# Error reading file: {e}\n")
                        
                    out.write("```\n\n")
                    
    return files_processed, total_lines


def main():
    parser = argparse.ArgumentParser(description="Update LLM text dumps for context feeding.")
    parser.add_argument("--outdir", type=str, default=DEFAULT_DUMPS_DIR, 
                        help="Output directory for the text dumps.")
    parser.add_argument("--target", type=str, default=None, 
                        help="Optional specific target directory relative to project root to dump.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    mappings_to_run = {}
    if args.target:
        target_path = os.path.abspath(os.path.join(PROJECT_ROOT, args.target))
        if not os.path.exists(target_path):
            logging.error(f"Provided target {args.target} does not exist.")
            return
        output_name = f"fedlearn_{args.target.replace('/', '_')}.txt"
        mappings_to_run[target_path] = os.path.join(args.outdir, output_name)
    else:
        # Default all mappings
        mappings_to_run = {
            src: os.path.join(args.outdir, dest) 
            for src, dest in MAPPINGS.items()
        }

    total_files = 0
    total_lines = 0

    for target_dir, output_file in mappings_to_run.items():
        if os.path.exists(target_dir):
            files, lines = build_txt(target_dir, output_file)
            logging.info(f" -> Processed {files} files ({lines} lines).")
            total_files += files
            total_lines += lines
        else:
            logging.warning(f"Directory {target_dir} not found. Skipping.")

    logging.info(f"Done. Processed a total of {total_files} files and {total_lines} lines.")

if __name__ == "__main__":
    main()
